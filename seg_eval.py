"""
语义分割评测模块

使用在Cityscapes等自动驾驶数据集上预训练的分割模型，
对生成图像和真值图像进行语义分割，然后比较分割结果的一致性。

评测逻辑：
1. 对gen图像和gt图像分别进行语义分割
2. 比较seg(gen) vs seg(gt)的一致性
3. 如果分割一致性高，说明生成图像的语义内容保持良好

支持的模型：
- Mask2Former (Cityscapes预训练)
- SegFormer (Cityscapes预训练)
- OneFormer (多数据集预训练)
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import json

from utils import (
    load_config, get_image_pairs, load_image, ensure_dir,
    save_segmentation_visualization, get_cityscapes_palette, CITYSCAPES_CLASSES
)
from metrics import (
    compute_segmentation_metrics, compute_segmentation_metrics_multilevel,
    compute_segmentation_consistency,
    aggregate_metrics, format_metrics_table,
    CITYSCAPES_COARSE_CLASSES
)


class SemanticSegmentor:
    """语义分割器基类"""

    def __init__(self, device: str = "cuda", num_classes: int = 19):
        self.device = device
        self.num_classes = num_classes
        self.model = None

    def predict(self, image: np.ndarray) -> np.ndarray:
        """预测分割图"""
        raise NotImplementedError


class Mask2FormerSegmentor(SemanticSegmentor):
    """
    Mask2Former语义分割器
    使用Cityscapes预训练模型
    """

    def __init__(self, backbone: str = "swin-l", device: str = "cuda"):
        super().__init__(device, num_classes=19)
        self.backbone = backbone
        self._load_model()

    def _load_model(self):
        """加载Mask2Former模型"""
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        # Cityscapes预训练的Mask2Former模型
        model_mapping = {
            "swin-l": "facebook/mask2former-swin-large-cityscapes-semantic",
            "swin-b": "facebook/mask2former-swin-base-cityscapes-semantic",
            "swin-s": "facebook/mask2former-swin-small-cityscapes-semantic",
            "swin-t": "facebook/mask2former-swin-tiny-cityscapes-semantic",
        }

        model_name = model_mapping.get(self.backbone, model_mapping["swin-l"])
        print(f"加载 Mask2Former 模型: {model_name}")

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        print("成功加载 Mask2Former (Cityscapes)")

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        预测语义分割图

        Args:
            image: RGB图像 (H, W, 3), uint8

        Returns:
            segmentation: 分割图 (H, W), int64, 类别ID
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        # 后处理得到语义分割结果
        segmentation = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.shape[:2]]
        )[0]

        return segmentation.cpu().numpy().astype(np.int64)


class SegFormerSegmentor(SemanticSegmentor):
    """
    SegFormer语义分割器
    使用Cityscapes预训练模型
    """

    def __init__(self, model_size: str = "b5", device: str = "cuda"):
        super().__init__(device, num_classes=19)
        self.model_size = model_size
        self._load_model()

    def _load_model(self):
        """加载SegFormer模型"""
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        # Cityscapes预训练的SegFormer模型
        model_mapping = {
            "b0": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
            "b1": "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
            "b2": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
            "b3": "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
            "b4": "nvidia/segformer-b4-finetuned-cityscapes-1024-1024",
            "b5": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        }

        model_name = model_mapping.get(self.model_size, model_mapping["b5"])
        print(f"加载 SegFormer 模型: {model_name}")

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        print("成功加载 SegFormer (Cityscapes)")

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        """预测语义分割图"""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits

        # 上采样到原始大小
        logits = torch.nn.functional.interpolate(
            logits,
            size=image.shape[:2],
            mode="bilinear",
            align_corners=False
        )

        segmentation = logits.argmax(dim=1).squeeze().cpu().numpy()
        return segmentation.astype(np.int64)


class OneFormerSegmentor(SemanticSegmentor):
    """
    OneFormer语义分割器
    支持多种数据集
    """

    def __init__(self, dataset: str = "cityscapes", device: str = "cuda"):
        num_classes = 19 if dataset == "cityscapes" else 150
        super().__init__(device, num_classes=num_classes)
        self.dataset = dataset
        self._load_model()

    def _load_model(self):
        """加载OneFormer模型"""
        from transformers import AutoProcessor, OneFormerForUniversalSegmentation

        model_mapping = {
            "cityscapes": "shi-labs/oneformer_cityscapes_swin_large",
            "ade20k": "shi-labs/oneformer_ade20k_swin_large",
            "coco": "shi-labs/oneformer_coco_swin_large",
        }

        model_name = model_mapping.get(self.dataset, model_mapping["cityscapes"])
        print(f"加载 OneFormer 模型: {model_name}")

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        print(f"成功加载 OneFormer ({self.dataset})")

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        """预测语义分割图"""
        inputs = self.processor(images=image, task_inputs=["semantic"], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        segmentation = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.shape[:2]]
        )[0]

        return segmentation.cpu().numpy().astype(np.int64)


def get_segmentor(config: Dict) -> SemanticSegmentor:
    """根据配置获取分割器"""
    model_name = config['segmentation']['model']
    device = config['segmentation']['device']

    if model_name == "mask2former":
        backbone = config['segmentation'].get('backbone', 'swin-l')
        return Mask2FormerSegmentor(backbone=backbone, device=device)
    elif model_name == "segformer":
        model_size = config['segmentation'].get('model_size', 'b5')
        return SegFormerSegmentor(model_size=model_size, device=device)
    elif model_name == "oneformer":
        dataset = config['segmentation'].get('dataset', 'cityscapes')
        return OneFormerSegmentor(dataset=dataset, device=device)
    else:
        raise ValueError(f"不支持的分割模型: {model_name}")


def evaluate_segmentation_consistency(config: Dict,
                                       save_vis: bool = True) -> Dict[str, Dict]:
    """
    评测语义分割一致性

    对于每对(gen, gt)图像：
    1. 对两者分别进行语义分割
    2. 比较seg(gen)和seg(gt)的一致性

    Args:
        config: 配置字典
        save_vis: 是否保存可视化

    Returns:
        每个相机的评测结果
    """
    print("\n" + "=" * 60)
    print("语义分割一致性评测")
    print("=" * 60)

    # 初始化分割器
    segmentor = get_segmentor(config)

    # 获取图像对
    image_pairs = get_image_pairs(config)
    if not image_pairs:
        raise ValueError("没有找到有效的图像对")

    # 创建输出目录
    output_dir = config['output']['seg_maps']
    if save_vis:
        ensure_dir(output_dir)

    palette = get_cityscapes_palette()
    results = {}

    for camera, pairs in image_pairs.items():
        print(f"\n处理相机: {camera}")
        camera_metrics = []
        class_ious_gen_gt = []  # 存储每个样本的class IoU用于后续分析

        # 创建相机子目录
        if save_vis:
            camera_out_dir = os.path.join(output_dir, camera)
            ensure_dir(camera_out_dir)

        for gen_path, gt_path in tqdm(pairs, desc=f"  {camera}"):
            filename = os.path.basename(gen_path).replace('.png', '')

            # 加载图像
            gen_img = load_image(gen_path)
            gt_img = load_image(gt_path)

            # 语义分割
            seg_gen = segmentor.predict(gen_img)
            seg_gt = segmentor.predict(gt_img)

            # 计算一致性指标
            consistency = compute_segmentation_consistency(seg_gen, seg_gt)

            # 计算分割指标（将seg_gt作为"真值"），同时计算超类指标
            compute_coarse = config.get('segmentation', {}).get('class_merging', {}).get('enabled', True)
            seg_metrics = compute_segmentation_metrics_multilevel(
                seg_gen, seg_gt,
                num_classes=segmentor.num_classes,
                compute_coarse=compute_coarse
            )

            metrics = {**consistency, **seg_metrics}
            camera_metrics.append(metrics)

            # 保存可视化
            if save_vis:
                save_segmentation_visualization(
                    seg_gen,
                    os.path.join(camera_out_dir, f"{filename}_gen_seg.png"),
                    palette=palette
                )
                save_segmentation_visualization(
                    seg_gt,
                    os.path.join(camera_out_dir, f"{filename}_gt_seg.png"),
                    palette=palette
                )

        # 汇总该相机的指标
        results[camera] = aggregate_metrics(camera_metrics)

        # 打印每类IoU
        print(f"\n  {camera} 分割指标:")
        print(format_metrics_table(results[camera]))

        if 'class_iou' in results[camera]:
            print(f"\n  每类IoU (gen vs gt) - 19类细粒度:")
            class_iou = results[camera]['class_iou']
            for i, (cls_name, iou) in enumerate(zip(CITYSCAPES_CLASSES, class_iou)):
                if not np.isnan(iou):
                    print(f"    {cls_name:<15}: {iou:6.2f}%")

        if 'coarse_class_iou' in results[camera]:
            print(f"\n  每类IoU (gen vs gt) - 7类超类:")
            coarse_iou = results[camera]['coarse_class_iou']
            for cls_name, iou in zip(CITYSCAPES_COARSE_CLASSES, coarse_iou):
                if not np.isnan(iou):
                    print(f"    {cls_name:<15}: {iou:6.2f}%")

    # 计算总体平均
    all_metrics = []
    for camera_results in results.values():
        all_metrics.append({k: v for k, v in camera_results.items()
                          if not k.endswith('_std') and k not in ('class_iou', 'coarse_class_iou')})
    results['overall'] = aggregate_metrics(all_metrics)

    print("\n" + "=" * 60)
    print("总体分割评测结果:")
    print(format_metrics_table(results['overall']))

    return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="语义分割一致性评测")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="配置文件路径")
    parser.add_argument("--no-vis", action="store_true",
                       help="不保存可视化结果")
    parser.add_argument("--output", type=str, default=None,
                       help="结果输出路径")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 运行评测
    results = evaluate_segmentation_consistency(config, save_vis=not args.no_vis)

    # 保存结果
    output_path = args.output or os.path.join(config['output']['metrics'], "seg_results.json")
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
