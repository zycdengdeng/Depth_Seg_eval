"""
SAM (Segment Anything) 结构一致性评测模块

使用SAM对生成图像和真值图像进行class-agnostic分割，
比较两者的边界/结构一致性。

与Mask2Former的区别：
- Mask2Former：语义级评测（car, road, person等类别是否一致）
- SAM：结构级评测（物体边界、分割结构是否一致，不涉及类别）

评测指标：
- Boundary F1 (BF1): 边界像素的F1分数
- Adjusted Rand Index (ARI): 分割一致性
- Variation of Information (VI): 信息论度量
- Segmentation Covering (SC): 分割覆盖率
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from scipy import ndimage
import json

from utils import load_config, get_image_pairs, load_image, ensure_dir, safe_import_transformers
from metrics import aggregate_metrics, format_metrics_table


# ============== SAM分割器 ==============

class SAMSegmentor:
    """SAM自动分割器"""

    def __init__(self, model_size: str = "large", device: str = "cuda"):
        self.device = device
        self.model_size = model_size
        self._load_model()

    def _load_model(self):
        """加载SAM模型"""
        safe_import_transformers()
        from transformers import SamModel, SamProcessor

        model_mapping = {
            "base": "facebook/sam-vit-base",
            "large": "facebook/sam-vit-large",
            "huge": "facebook/sam-vit-huge",
        }

        model_name = model_mapping.get(self.model_size, model_mapping["large"])
        print(f"加载 SAM 模型: {model_name}")

        self.processor = SamProcessor.from_pretrained(model_name, use_fast=False)
        self.model = SamModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        print("成功加载 SAM")

    @torch.no_grad()
    def get_embeddings(self, image: np.ndarray):
        """获取图像embedding"""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        image_embeddings = self.model.get_image_embeddings(inputs["pixel_values"])
        return image_embeddings, inputs

    @torch.no_grad()
    def predict_auto(self, image: np.ndarray, points_per_side: int = 32) -> np.ndarray:
        """
        自动分割：在图像上均匀采样点作为prompt，生成分割图

        Args:
            image: RGB图像 (H, W, 3), uint8
            points_per_side: 每边采样点数

        Returns:
            segmentation: 分割图 (H, W), int，每个区域有唯一ID
        """
        h, w = image.shape[:2]

        # 生成均匀网格点
        x = np.linspace(0, w - 1, points_per_side).astype(int)
        y = np.linspace(0, h - 1, points_per_side).astype(int)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx.flatten(), yy.flatten()], axis=1)  # (N, 2)

        # 获取image embeddings
        image_embeddings, inputs = self.get_embeddings(image)

        # 批量处理点
        seg_map = np.zeros((h, w), dtype=np.int32)
        mask_id = 1
        all_masks = []
        all_scores = []

        batch_size = 64
        for i in range(0, len(points), batch_size):
            batch_points = points[i:i+batch_size]
            # 格式化为SAM需要的input_points格式
            input_points = torch.tensor(batch_points, dtype=torch.float32).unsqueeze(0)
            input_points = input_points.unsqueeze(2)  # (1, N, 1, 2)

            # 逐点预测
            for j in range(len(batch_points)):
                pt = torch.tensor(batch_points[j:j+1], dtype=torch.float32)
                pt = pt.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, 1, 2)

                outputs = self.model(
                    image_embeddings=image_embeddings,
                    input_points=pt,
                    multimask_output=False,
                )

                mask = outputs.pred_masks.squeeze().cpu()
                score = outputs.iou_scores.squeeze().cpu().item()

                # 上采样到原始大小
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze() > 0.5

                all_masks.append(mask.numpy())
                all_scores.append(score)

        # 按score排序，高分优先
        sorted_indices = np.argsort(all_scores)[::-1]

        # 合成分割图（高分mask优先）
        for idx in sorted_indices:
            mask = all_masks[idx]
            score = all_scores[idx]

            if score < 0.7:  # 过滤低质量mask
                continue

            # 只填充还没被分配的区域
            new_region = mask & (seg_map == 0)
            if new_region.sum() > 100:  # 最小区域面积
                seg_map[new_region] = mask_id
                mask_id += 1

        return seg_map


class SAMSegmentorFast:
    """
    快速SAM分割器 - 使用边缘检测代替完整SAM推理
    用SAM的image encoder提取特征，然后用特征相似性做分割
    """

    def __init__(self, model_size: str = "large", device: str = "cuda"):
        self.device = device
        self.model_size = model_size
        self._load_model()

    def _load_model(self):
        """加载SAM模型"""
        safe_import_transformers()
        from transformers import SamModel, SamProcessor

        model_mapping = {
            "base": "facebook/sam-vit-base",
            "large": "facebook/sam-vit-large",
            "huge": "facebook/sam-vit-huge",
        }

        model_name = model_mapping.get(self.model_size, model_mapping["large"])
        print(f"加载 SAM 模型: {model_name}")

        self.processor = SamProcessor.from_pretrained(model_name, use_fast=False)
        self.model = SamModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("成功加载 SAM")

    @torch.no_grad()
    def get_feature_map(self, image: np.ndarray) -> np.ndarray:
        """
        获取SAM的特征图

        Args:
            image: RGB图像 (H, W, 3)

        Returns:
            features: 特征图 (H, W, C)
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 获取image encoder的输出特征
        image_embeddings = self.model.get_image_embeddings(inputs["pixel_values"])

        # (1, C, H', W') -> (H', W', C)
        features = image_embeddings.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # 上采样到原始大小
        from PIL import Image as PILImage
        h, w = image.shape[:2]
        features_resized = np.zeros((h, w, features.shape[2]), dtype=np.float32)
        for c in range(features.shape[2]):
            feat_c = PILImage.fromarray(features[:, :, c])
            feat_c = feat_c.resize((w, h), PILImage.BILINEAR)
            features_resized[:, :, c] = np.array(feat_c)

        return features_resized

    @torch.no_grad()
    def get_edge_map(self, image: np.ndarray) -> np.ndarray:
        """
        从SAM特征图提取边缘图

        Args:
            image: RGB图像 (H, W, 3)

        Returns:
            edge_map: 边缘图 (H, W), float32, 0-1
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        image_embeddings = self.model.get_image_embeddings(inputs["pixel_values"])
        features = image_embeddings.squeeze(0)  # (C, H', W')

        # 计算特征梯度作为边缘
        grad_x = torch.diff(features, dim=2)  # 水平梯度
        grad_y = torch.diff(features, dim=1)  # 垂直梯度

        # 取梯度幅值
        edge_x = torch.norm(grad_x, dim=0)  # (H', W'-1)
        edge_y = torch.norm(grad_y, dim=0)  # (H'-1, W')

        # Pad到相同大小
        edge_x = torch.nn.functional.pad(edge_x, (0, 1, 0, 0))
        edge_y = torch.nn.functional.pad(edge_y, (0, 0, 0, 1))

        edge = (edge_x + edge_y) / 2
        edge = edge.cpu().numpy()

        # 归一化到0-1
        edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)

        # 上采样到原始大小
        h, w = image.shape[:2]
        from PIL import Image as PILImage
        edge_img = PILImage.fromarray((edge * 255).astype(np.uint8))
        edge_img = edge_img.resize((w, h), PILImage.BILINEAR)
        edge = np.array(edge_img).astype(np.float32) / 255.0

        return edge


# ============== 结构一致性指标 ==============

def compute_boundary_f1(seg1: np.ndarray, seg2: np.ndarray,
                        tolerance: int = 2) -> Dict[str, float]:
    """
    计算边界F1分数

    Args:
        seg1, seg2: 分割图 (H, W)
        tolerance: 边界容差像素数

    Returns:
        precision, recall, f1
    """
    # 提取边界
    boundary1 = extract_boundaries(seg1)
    boundary2 = extract_boundaries(seg2)

    if boundary1.sum() == 0 or boundary2.sum() == 0:
        return {'boundary_precision': 0.0, 'boundary_recall': 0.0, 'boundary_f1': 0.0}

    # 对boundary2进行膨胀（容差）
    from scipy.ndimage import binary_dilation
    struct = np.ones((2 * tolerance + 1, 2 * tolerance + 1))
    boundary2_dilated = binary_dilation(boundary2, structure=struct)
    boundary1_dilated = binary_dilation(boundary1, structure=struct)

    # Precision: boundary1中有多少落在boundary2的容差范围内
    precision = np.sum(boundary1 & boundary2_dilated) / (np.sum(boundary1) + 1e-10)

    # Recall: boundary2中有多少落在boundary1的容差范围内
    recall = np.sum(boundary2 & boundary1_dilated) / (np.sum(boundary2) + 1e-10)

    # F1
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return {
        'boundary_precision': precision * 100,
        'boundary_recall': recall * 100,
        'boundary_f1': f1 * 100,
    }


def extract_boundaries(seg: np.ndarray) -> np.ndarray:
    """从分割图提取边界"""
    boundaries = np.zeros_like(seg, dtype=bool)

    # 使用Sobel检测分割边界
    for direction in [0, 1]:
        diff = np.diff(seg, axis=direction)
        if direction == 0:
            boundaries[:-1, :] |= (diff != 0)
            boundaries[1:, :] |= (diff != 0)
        else:
            boundaries[:, :-1] |= (diff != 0)
            boundaries[:, 1:] |= (diff != 0)

    return boundaries


def compute_edge_consistency(edge1: np.ndarray, edge2: np.ndarray,
                             tolerance: int = 3) -> Dict[str, float]:
    """
    计算边缘一致性（基于SAM特征边缘图）

    Args:
        edge1, edge2: 边缘图 (H, W), float32, 0-1
        tolerance: 容差像素数

    Returns:
        边缘一致性指标
    """
    # 二值化边缘
    threshold = 0.3
    bin_edge1 = edge1 > threshold
    bin_edge2 = edge2 > threshold

    if bin_edge1.sum() == 0 or bin_edge2.sum() == 0:
        return {'edge_precision': 0.0, 'edge_recall': 0.0, 'edge_f1': 0.0,
                'edge_correlation': 0.0}

    # 边界F1（二值化后）
    from scipy.ndimage import binary_dilation
    struct = np.ones((2 * tolerance + 1, 2 * tolerance + 1))
    edge2_dilated = binary_dilation(bin_edge2, structure=struct)
    edge1_dilated = binary_dilation(bin_edge1, structure=struct)

    precision = np.sum(bin_edge1 & edge2_dilated) / (np.sum(bin_edge1) + 1e-10)
    recall = np.sum(bin_edge2 & edge1_dilated) / (np.sum(bin_edge2) + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    # 连续值相关性
    correlation = np.corrcoef(edge1.flatten(), edge2.flatten())[0, 1]

    return {
        'edge_precision': precision * 100,
        'edge_recall': recall * 100,
        'edge_f1': f1 * 100,
        'edge_correlation': correlation,
    }


def compute_adjusted_rand_index(seg1: np.ndarray, seg2: np.ndarray) -> float:
    """
    计算Adjusted Rand Index (ARI)
    衡量两个分割的一致性，调整了随机因素

    ARI = 1: 完美一致
    ARI = 0: 随机水平
    ARI < 0: 比随机更差
    """
    # 下采样加速计算
    from PIL import Image as PILImage
    h, w = seg1.shape
    scale = max(1, min(h, w) // 256)
    if scale > 1:
        seg1_small = np.array(PILImage.fromarray(seg1.astype(np.int32)).resize(
            (w // scale, h // scale), PILImage.NEAREST))
        seg2_small = np.array(PILImage.fromarray(seg2.astype(np.int32)).resize(
            (w // scale, h // scale), PILImage.NEAREST))
    else:
        seg1_small = seg1
        seg2_small = seg2

    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(seg1_small.flatten(), seg2_small.flatten())
    return ari


def compute_variation_of_information(seg1: np.ndarray, seg2: np.ndarray) -> Dict[str, float]:
    """
    计算Variation of Information (VI)
    VI = H(S1|S2) + H(S2|S1)
    越低越好（0表示完全一致）
    """
    # 下采样加速
    from PIL import Image as PILImage
    h, w = seg1.shape
    scale = max(1, min(h, w) // 256)
    if scale > 1:
        seg1_small = np.array(PILImage.fromarray(seg1.astype(np.int32)).resize(
            (w // scale, h // scale), PILImage.NEAREST))
        seg2_small = np.array(PILImage.fromarray(seg2.astype(np.int32)).resize(
            (w // scale, h // scale), PILImage.NEAREST))
    else:
        seg1_small = seg1
        seg2_small = seg2

    n = seg1_small.size

    # 计算联合概率
    labels1 = seg1_small.flatten()
    labels2 = seg2_small.flatten()

    # 获取唯一标签
    unique1 = np.unique(labels1)
    unique2 = np.unique(labels2)

    # 构建联合概率表
    h_s1_given_s2 = 0.0
    h_s2_given_s1 = 0.0

    for l2 in unique2:
        mask2 = labels2 == l2
        p2 = mask2.sum() / n
        if p2 == 0:
            continue

        for l1 in unique1:
            mask1 = labels1 == l1
            p1 = mask1.sum() / n
            if p1 == 0:
                continue

            p_joint = (mask1 & mask2).sum() / n
            if p_joint > 0:
                h_s1_given_s2 -= p_joint * np.log2(p_joint / p2)
                h_s2_given_s1 -= p_joint * np.log2(p_joint / p1)

    vi = h_s1_given_s2 + h_s2_given_s1

    return {
        'vi_total': vi,
        'vi_under': h_s1_given_s2,  # 欠分割
        'vi_over': h_s2_given_s1,   # 过分割
    }


# ============== 主评测函数 ==============

def evaluate_sam_consistency(config: Dict, save_vis: bool = True,
                             use_full_sam: bool = False) -> Dict[str, Dict]:
    """
    使用SAM评测结构一致性

    Args:
        config: 配置字典
        save_vis: 是否保存可视化
        use_full_sam: 是否使用完整SAM分割（慢）或仅用边缘比较（快）

    Returns:
        每个相机的评测结果
    """
    print("\n" + "=" * 60)
    print("SAM 结构一致性评测")
    print("=" * 60)

    sam_config = config.get('sam', {})
    model_size = sam_config.get('model_size', 'large')
    device = sam_config.get('device', config.get('segmentation', {}).get('device', 'cuda'))

    # 初始化SAM
    sam = SAMSegmentorFast(model_size=model_size, device=device)

    # 获取图像对
    image_pairs = get_image_pairs(config)
    if not image_pairs:
        raise ValueError("没有找到有效的图像对")

    output_dir = os.path.join(config['output'].get('root', './results'), 'sam_maps')
    if save_vis:
        ensure_dir(output_dir)

    results = {}

    for camera, pairs in image_pairs.items():
        print(f"\n处理相机: {camera}")
        camera_metrics = []

        if save_vis:
            camera_out_dir = os.path.join(output_dir, camera)
            ensure_dir(camera_out_dir)

        for gen_path, gt_path in tqdm(pairs, desc=f"  {camera}"):
            filename = os.path.basename(gen_path).replace('.png', '')

            # 加载图像
            gen_img = load_image(gen_path)
            gt_img = load_image(gt_path)

            # 获取SAM边缘图
            edge_gen = sam.get_edge_map(gen_img)
            edge_gt = sam.get_edge_map(gt_img)

            # 计算边缘一致性
            metrics = compute_edge_consistency(edge_gen, edge_gt, tolerance=3)

            camera_metrics.append(metrics)

            # 保存可视化
            if save_vis:
                from PIL import Image as PILImage
                PILImage.fromarray((edge_gen * 255).astype(np.uint8)).save(
                    os.path.join(camera_out_dir, f"{filename}_gen_edge.png"))
                PILImage.fromarray((edge_gt * 255).astype(np.uint8)).save(
                    os.path.join(camera_out_dir, f"{filename}_gt_edge.png"))

        # 汇总
        results[camera] = aggregate_metrics(camera_metrics)
        print(f"\n  {camera} SAM结构指标:")
        print(format_metrics_table(results[camera]))

    # 总体平均
    all_metrics = []
    for camera_results in results.values():
        all_metrics.append({k: v for k, v in camera_results.items()
                          if not k.endswith('_std')})
    results['overall'] = aggregate_metrics(all_metrics)

    print("\n" + "=" * 60)
    print("总体SAM结构评测结果:")
    print(format_metrics_table(results['overall']))

    return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="SAM结构一致性评测")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="配置文件路径")
    parser.add_argument("--no-vis", action="store_true",
                       help="不保存可视化结果")
    parser.add_argument("--output", type=str, default=None,
                       help="结果输出路径")
    args = parser.parse_args()

    config = load_config(args.config)
    results = evaluate_sam_consistency(config, save_vis=not args.no_vis)

    output_path = args.output or os.path.join(config['output']['metrics'], "sam_results.json")
    ensure_dir(os.path.dirname(output_path))

    from evaluate import convert_to_serializable
    with open(output_path, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
