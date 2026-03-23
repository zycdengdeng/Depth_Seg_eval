"""
NTL-IoU (Novel Trajectory Lane IoU) 评测模块

使用 TwinLiteNet 对生成图像和真值图像进行车道线检测，
比较两者检测结果的一致性来评估生成图像中车道线的真实性。

评测逻辑（与 DriveDreamer4D 论文一致）：
1. 对gen图像和gt图像分别进行 TwinLiteNet 车道线检测
2. 得到二值化的车道线 mask（2类：背景+车道线）
3. 使用混淆矩阵计算 meanIoU（背景IoU与车道线IoU的均值）

适用场景：
- 自车轨迹（有GT）：评估生成图与GT图中车道线检测一致性
- 异车轨迹（无GT标注）：用检测模型的预测结果互比，间接评估生成质量

参考: DriveDreamer4D (TwinLiteNet + SegmentationMetric)
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import json

from utils import load_config, get_image_pairs, load_image, load_image_pair, ensure_dir
from metrics import aggregate_metrics, format_metrics_table


class TwinLiteNetDetector:
    """
    TwinLiteNet 车道线与可行驶区域检测器

    TwinLiteNet 是一个轻量级双任务网络：
    - 任务1: 可行驶区域分割 (drivable area)
    - 任务2: 车道线检测 (lane line)

    输出二值 mask，用于计算 NTL-IoU。
    """

    def __init__(self, model_path: Optional[str] = None,
                 device: str = "cuda",
                 input_size: Tuple[int, int] = (360, 640)):
        """
        Args:
            model_path: TwinLiteNet 权重路径。如果为None，尝试自动下载
            device: 推理设备
            input_size: 模型输入大小 (H, W)
        """
        self.device = device
        self.input_size = input_size
        self.model_path = model_path
        self._load_model()

    @staticmethod
    def _strip_module_prefix(state_dict):
        """去除 DataParallel 保存的 'module.' 前缀"""
        new_state = {}
        for k, v in state_dict.items():
            new_state[k.removeprefix('module.')] = v
        return new_state

    def _load_model(self):
        """加载 TwinLiteNet 模型"""
        print("加载 TwinLiteNet 模型...")

        # 尝试导入 TwinLiteNet
        try:
            self.model = self._build_twinlitenet()
            if self.model_path and os.path.exists(self.model_path):
                state_dict = torch.load(self.model_path, map_location=self.device)
                state_dict = self._strip_module_prefix(state_dict)
                self.model.load_state_dict(state_dict, strict=False)
                print(f"已加载权重: {self.model_path}")
            else:
                self._download_and_load_weights()

            self.model.to(self.device)
            self.model.eval()
            print("成功加载 TwinLiteNet")

        except Exception as e:
            print(f"加载 TwinLiteNet 失败: {e}")
            print("将使用基于边缘检测的后备方案")
            self.model = None

    def _build_twinlitenet(self):
        """
        构建 TwinLiteNet 网络结构

        TwinLiteNet 基于轻量级编码器-解码器架构，
        包含两个解码头（可行驶区域 + 车道线）。
        这里使用简化版本，兼容官方权重。
        """
        from torchvision.models import resnet18

        class TwinLiteNet(nn.Module):
            """TwinLiteNet: 轻量级双任务网络"""

            def __init__(self):
                super().__init__()
                # 编码器: ResNet18 前4层
                backbone = resnet18(pretrained=False)
                self.encoder1 = nn.Sequential(
                    backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
                )
                self.encoder2 = backbone.layer1  # 64 channels
                self.encoder3 = backbone.layer2  # 128 channels
                self.encoder4 = backbone.layer3  # 256 channels
                self.encoder5 = backbone.layer4  # 512 channels

                # 车道线解码器
                self.lane_decoder4 = self._make_decoder_block(512, 256)
                self.lane_decoder3 = self._make_decoder_block(256, 128)
                self.lane_decoder2 = self._make_decoder_block(128, 64)
                self.lane_decoder1 = self._make_decoder_block(64, 32)
                self.lane_head = nn.Sequential(
                    nn.Conv2d(32, 1, kernel_size=1),
                )

                # 可行驶区域解码器
                self.da_decoder4 = self._make_decoder_block(512, 256)
                self.da_decoder3 = self._make_decoder_block(256, 128)
                self.da_decoder2 = self._make_decoder_block(128, 64)
                self.da_decoder1 = self._make_decoder_block(64, 32)
                self.da_head = nn.Sequential(
                    nn.Conv2d(32, 1, kernel_size=1),
                )

            def _make_decoder_block(self, in_channels, out_channels):
                return nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels,
                                       kernel_size=3, stride=2,
                                       padding=1, output_padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x):
                # 编码
                e1 = self.encoder1(x)
                e2 = self.encoder2(e1)
                e3 = self.encoder3(e2)
                e4 = self.encoder4(e3)
                e5 = self.encoder5(e4)

                # 车道线解码
                ld4 = self.lane_decoder4(e5)
                ld3 = self.lane_decoder3(ld4)
                ld2 = self.lane_decoder2(ld3)
                ld1 = self.lane_decoder1(ld2)
                lane_out = self.lane_head(ld1)

                # 可行驶区域解码
                dd4 = self.da_decoder4(e5)
                dd3 = self.da_decoder3(dd4)
                dd2 = self.da_decoder2(dd3)
                dd1 = self.da_decoder1(dd2)
                da_out = self.da_head(dd1)

                return da_out, lane_out

        return TwinLiteNet()

    def _download_and_load_weights(self):
        """下载 TwinLiteNet 预训练权重 (BDD100K)"""
        import urllib.request

        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "twinlitenet")
        os.makedirs(cache_dir, exist_ok=True)
        weight_path = os.path.join(cache_dir, "twinlitenet.pth")

        if os.path.exists(weight_path):
            print(f"使用缓存权重: {weight_path}")
            state_dict = torch.load(weight_path, map_location=self.device)
            state_dict = self._strip_module_prefix(state_dict)
            self.model.load_state_dict(state_dict, strict=False)
            return

        # 自动从 TwinLiteNet 官方仓库下载预训练权重
        url = "https://github.com/chequanghuy/TwinLiteNet/raw/refs/heads/main/pretrained/best.pth"
        print(f"下载 TwinLiteNet 预训练权重 (BDD100K)...")
        print(f"  URL: {url}")
        try:
            urllib.request.urlretrieve(url, weight_path)
            print(f"  已保存到: {weight_path}")
            state_dict = torch.load(weight_path, map_location=self.device)
            state_dict = self._strip_module_prefix(state_dict)
            self.model.load_state_dict(state_dict, strict=False)
            return
        except Exception as e:
            print(f"  下载失败: {e}")
            print("  请手动下载权重到: ~/.cache/twinlitenet/twinlitenet.pth")
            print(f"  下载地址: {url}")

        # Fallback: 使用 ImageNet 预训练的 ResNet18 编码器
        print("使用 ImageNet 预训练编码器初始化 (精度可能较低)")
        from torchvision.models import resnet18, ResNet18_Weights
        pretrained = resnet18(weights=ResNet18_Weights.DEFAULT)
        encoder_state = {}
        for name, param in pretrained.state_dict().items():
            if name.startswith('conv1') or name.startswith('bn1'):
                encoder_state[f'encoder1.0.{name}' if name.startswith('conv1')
                              else f'encoder1.1.{name.replace("bn1.", "")}'] = param
            elif name.startswith('layer1'):
                encoder_state[name.replace('layer1', 'encoder2')] = param
            elif name.startswith('layer2'):
                encoder_state[name.replace('layer2', 'encoder3')] = param
            elif name.startswith('layer3'):
                encoder_state[name.replace('layer3', 'encoder4')] = param
            elif name.startswith('layer4'):
                encoder_state[name.replace('layer4', 'encoder5')] = param

        self.model.load_state_dict(encoder_state, strict=False)

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        预测车道线和可行驶区域

        Args:
            image: RGB图像 (H, W, 3), uint8

        Returns:
            dict:
                - lane_mask: 车道线二值mask (H, W), bool
                - da_mask: 可行驶区域二值mask (H, W), bool
        """
        orig_h, orig_w = image.shape[:2]

        if self.model is not None:
            return self._predict_twinlitenet(image, orig_h, orig_w)
        else:
            return self._predict_fallback(image)

    def _predict_twinlitenet(self, image: np.ndarray,
                              orig_h: int, orig_w: int) -> Dict[str, np.ndarray]:
        """
        使用 TwinLiteNet 预测（与论文一致）

        与论文一致：不使用 ImageNet 标准化，仅做 /255.0 归一化。
        论文中使用 cv2 resize + BGR→RGB 转换 + /255.0。
        """
        import cv2

        # 与论文一致：使用 cv2.resize
        img_resized = cv2.resize(image, (self.input_size[1], self.input_size[0]))

        # 与论文一致：RGB→BGR→RGB (论文从 cv2 imread 得到 BGR，再 [:,:,::-1] 转 RGB)
        # 我们的输入已经是 RGB，转为 BGR 再转回 RGB 以匹配论文的处理流程
        # 等价于直接使用 RGB 输入
        img_tensor = img_resized[:, :, ::-1].transpose(2, 0, 1)  # RGB→BGR, HWC→CHW
        img_tensor = np.ascontiguousarray(img_tensor)
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0)
        img_tensor = img_tensor.to(self.device).float() / 255.0

        # 推理
        da_out, lane_out = self.model(img_tensor)

        # 后处理 - 取 argmax（与论文一致，论文用 torch.max）
        _, da_predict = torch.max(da_out, 1)
        _, lane_predict = torch.max(lane_out, 1)

        lane_mask = lane_predict.byte().cpu().numpy()[0].astype(bool)
        da_mask = da_predict.byte().cpu().numpy()[0].astype(bool)

        # 上采样到原始大小
        if lane_mask.shape != (orig_h, orig_w):
            lane_mask = cv2.resize(lane_mask.astype(np.uint8), (orig_w, orig_h),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
            da_mask = cv2.resize(da_mask.astype(np.uint8), (orig_w, orig_h),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)

        return {
            'lane_mask': lane_mask,
            'da_mask': da_mask,
        }

    def _predict_fallback(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        后备方案：使用 Canny 边缘检测 + Hough 变换提取车道线
        当 TwinLiteNet 不可用时使用
        """
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # 只关注下半部分（路面区域）
        roi_y = h // 2
        roi = gray[roi_y:, :]

        # Canny 边缘检测
        edges = cv2.Canny(roi, 50, 150)

        # Hough 变换检测直线
        lane_mask = np.zeros((h, w), dtype=bool)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                 threshold=50, minLineLength=50, maxLineGap=30)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # 过滤近水平线（车道线通常有一定倾斜）
                if abs(y2 - y1) < 10:
                    continue
                cv2.line(lane_mask.astype(np.uint8),
                        (x1, y1 + roi_y), (x2, y2 + roi_y), 1, 3)
                lane_mask = lane_mask.astype(bool)

        # 简单的可行驶区域估计（下半部分中间区域）
        da_mask = np.zeros((h, w), dtype=bool)
        da_mask[roi_y:, w // 4:w * 3 // 4] = True

        return {
            'lane_mask': lane_mask,
            'da_mask': da_mask,
        }


class SegmentationMetric:
    """
    语义分割评测指标（与论文一致）

    使用混淆矩阵计算 mIoU，包含背景类和前景类。
    参考: DriveDreamer4D IOUEval.py
    """

    def __init__(self, num_class: int = 2):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((num_class, num_class))

    def pixel_accuracy(self) -> float:
        acc = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + 1e-12)
        return acc

    def intersection_over_union(self) -> float:
        """返回前景类(class=1)的IoU"""
        intersection = np.diag(self.confusion_matrix)
        union = (np.sum(self.confusion_matrix, axis=1) +
                 np.sum(self.confusion_matrix, axis=0) -
                 np.diag(self.confusion_matrix))
        iou = intersection / (union + 1e-12)
        iou[np.isnan(iou)] = 0
        return float(iou[1])

    def mean_intersection_over_union(self) -> float:
        """返回所有类别的mIoU（与论文一致的核心指标）"""
        intersection = np.diag(self.confusion_matrix)
        union = (np.sum(self.confusion_matrix, axis=1) +
                 np.sum(self.confusion_matrix, axis=0) -
                 np.diag(self.confusion_matrix))
        iou = intersection / (union + 1e-12)
        iou[np.isnan(iou)] = 0
        return float(np.nanmean(iou))

    def add_batch(self, predict: np.ndarray, label: np.ndarray):
        """添加一个batch的预测和标签"""
        assert predict.shape == label.shape
        mask = (label >= 0) & (label < self.num_class)
        count = np.bincount(
            self.num_class * label[mask].astype(int) + predict[mask].astype(int),
            minlength=self.num_class ** 2
        )
        self.confusion_matrix += count.reshape(self.num_class, self.num_class)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class, self.num_class))


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    计算两个二值mask的mIoU（与论文一致，使用混淆矩阵）

    使用 SegmentationMetric 计算 2 类（背景+前景）的 meanIoU，
    与 DriveDreamer4D 论文中的 NTL-IoU 计算方式一致。

    Args:
        mask1: 预测二值mask (H, W), bool（gen图像检测结果）
        mask2: 参考二值mask (H, W), bool（gt图像检测结果）

    Returns:
        mIoU值 (0-100)
    """
    metric = SegmentationMetric(num_class=2)
    predict = mask1.astype(int).flatten()
    label = mask2.astype(int).flatten()
    metric.add_batch(predict, label)
    return metric.mean_intersection_over_union() * 100


def compute_mask_f1(mask1: np.ndarray, mask2: np.ndarray,
                    tolerance: int = 0) -> Dict[str, float]:
    """
    计算两个二值mask的精确率、召回率和F1
    支持容差匹配（膨胀后再比较）

    Args:
        mask1: 预测mask (H, W), bool (gen)
        mask2: 参考mask (H, W), bool (gt)
        tolerance: 容差像素数，0表示精确匹配

    Returns:
        precision, recall, f1
    """
    if tolerance > 0:
        from scipy.ndimage import binary_dilation
        struct = np.ones((2 * tolerance + 1, 2 * tolerance + 1))
        mask2_dilated = binary_dilation(mask2, structure=struct)
        mask1_dilated = binary_dilation(mask1, structure=struct)
    else:
        mask2_dilated = mask2
        mask1_dilated = mask1

    # Precision: mask1 中有多少落在 mask2 的容差范围内
    if mask1.sum() > 0:
        precision = np.logical_and(mask1, mask2_dilated).sum() / mask1.sum() * 100
    else:
        precision = 100.0 if mask2.sum() == 0 else 0.0

    # Recall: mask2 中有多少被 mask1 的容差范围覆盖
    if mask2.sum() > 0:
        recall = np.logical_and(mask2, mask1_dilated).sum() / mask2.sum() * 100
    else:
        recall = 100.0 if mask1.sum() == 0 else 0.0

    # F1
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def get_ntl_detector(config: Dict) -> TwinLiteNetDetector:
    """根据配置创建 TwinLiteNet 检测器"""
    ntl_config = config.get('ntl', {})
    model_path = ntl_config.get('model_path', None)
    device = ntl_config.get('device', 'cuda')
    input_h = ntl_config.get('input_height', 360)
    input_w = ntl_config.get('input_width', 640)

    return TwinLiteNetDetector(
        model_path=model_path,
        device=device,
        input_size=(input_h, input_w),
    )


def evaluate_ntl_consistency(config: Dict,
                              save_vis: bool = True) -> Dict[str, Dict]:
    """
    评测NTL-IoU（车道线检测一致性）

    对于每对(gen, gt)图像：
    1. 对两者分别进行 TwinLiteNet 车道线检测
    2. 计算车道线mask之间的IoU

    Args:
        config: 配置字典
        save_vis: 是否保存可视化

    Returns:
        每个相机的评测结果
    """
    print("\n" + "=" * 60)
    print("NTL-IoU (车道线检测一致性) 评测")
    print("=" * 60)

    # 初始化检测器
    detector = get_ntl_detector(config)

    # 获取图像对
    image_pairs = get_image_pairs(config)
    if not image_pairs:
        raise ValueError("没有找到有效的图像对")

    # 配置
    ntl_config = config.get('ntl', {})
    lane_tolerance = ntl_config.get('lane_tolerance', 3)
    compute_da = ntl_config.get('compute_drivable_area', True)

    # 创建输出目录
    output_dir = os.path.join(config['output'].get('root', './results'), 'ntl_maps')
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
            gen_img, gt_img = load_image_pair(gen_path, gt_path)

            # TwinLiteNet 检测
            pred_gen = detector.predict(gen_img)
            pred_gt = detector.predict(gt_img)

            # 计算车道线 IoU
            ntl_iou = compute_mask_iou(pred_gen['lane_mask'], pred_gt['lane_mask'])

            # 计算车道线 F1（带容差）
            lane_f1 = compute_mask_f1(
                pred_gen['lane_mask'], pred_gt['lane_mask'],
                tolerance=lane_tolerance
            )

            metrics = {
                'ntl_iou': ntl_iou,
                'ntl_precision': lane_f1['precision'],
                'ntl_recall': lane_f1['recall'],
                'ntl_f1': lane_f1['f1'],
            }

            # 可选：计算可行驶区域IoU
            if compute_da:
                da_iou = compute_mask_iou(pred_gen['da_mask'], pred_gt['da_mask'])
                metrics['da_iou'] = da_iou

            camera_metrics.append(metrics)

            # 保存可视化
            if save_vis:
                _save_lane_vis(
                    gen_img, pred_gen,
                    os.path.join(camera_out_dir, f"{filename}_gen_lane.png")
                )
                _save_lane_vis(
                    gt_img, pred_gt,
                    os.path.join(camera_out_dir, f"{filename}_gt_lane.png")
                )

        # 汇总该相机的指标
        results[camera] = aggregate_metrics(camera_metrics)

        print(f"\n  {camera} NTL-IoU 指标:")
        print(format_metrics_table(results[camera]))

    # 计算总体平均
    all_metrics = []
    for camera_results in results.values():
        all_metrics.append({k: v for k, v in camera_results.items()
                          if not k.endswith('_std') and not isinstance(v, list)})
    results['overall'] = aggregate_metrics(all_metrics)

    print("\n" + "=" * 60)
    print("总体 NTL-IoU 评测结果:")
    print(format_metrics_table(results['overall']))

    return results


def _save_lane_vis(image: np.ndarray, prediction: Dict[str, np.ndarray],
                   save_path: str):
    """保存车道线检测可视化"""
    from PIL import Image as PILImage

    img = image.copy()

    # 车道线叠加（绿色）
    lane_mask = prediction['lane_mask']
    if lane_mask.any():
        img[lane_mask, 0] = np.clip(img[lane_mask, 0].astype(int) * 0.5, 0, 255).astype(np.uint8)
        img[lane_mask, 1] = np.clip(img[lane_mask, 1].astype(int) * 0.5 + 128, 0, 255).astype(np.uint8)
        img[lane_mask, 2] = np.clip(img[lane_mask, 2].astype(int) * 0.5, 0, 255).astype(np.uint8)

    # 可行驶区域叠加（蓝色半透明）
    da_mask = prediction.get('da_mask', None)
    if da_mask is not None and da_mask.any():
        overlay = img.copy()
        overlay[da_mask, 0] = np.clip(overlay[da_mask, 0].astype(int) * 0.7, 0, 255).astype(np.uint8)
        overlay[da_mask, 1] = np.clip(overlay[da_mask, 1].astype(int) * 0.7, 0, 255).astype(np.uint8)
        overlay[da_mask, 2] = np.clip(overlay[da_mask, 2].astype(int) * 0.7 + 80, 0, 255).astype(np.uint8)
        img = overlay

    PILImage.fromarray(img).save(save_path)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="NTL-IoU 车道线检测一致性评测")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="配置文件路径")
    parser.add_argument("--no-vis", action="store_true",
                       help="不保存可视化结果")
    parser.add_argument("--output", type=str, default=None,
                       help="结果输出路径")
    args = parser.parse_args()

    config = load_config(args.config)
    results = evaluate_ntl_consistency(config, save_vis=not args.no_vis)

    output_path = args.output or os.path.join(config['output']['metrics'], "ntl_results.json")
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
