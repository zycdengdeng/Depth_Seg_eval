"""
深度估计评测模块

使用Depth Anything V2对生成图像和真值图像进行深度估计，
然后比较两者的深度一致性。

评测逻辑：
1. 对gen图像和gt图像分别跑深度估计模型
2. 比较depth(gen) vs depth(gt)的相关性和一致性
3. 如果深度一致性高，说明生成图像的几何结构保持良好
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
    normalize_depth, align_depth_scale, save_depth_visualization
)
from metrics import (
    compute_depth_metrics, compute_depth_correlation,
    aggregate_metrics, format_metrics_table
)


class DepthEstimator:
    """深度估计器基类"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None

    def predict(self, image: np.ndarray) -> np.ndarray:
        """预测深度图"""
        raise NotImplementedError


class DepthAnythingV2Estimator(DepthEstimator):
    """
    Depth Anything V2 深度估计器
    https://github.com/DepthAnything/Depth-Anything-V2
    """

    def __init__(self, model_size: str = "large", device: str = "cuda"):
        super().__init__(device)
        self.model_size = model_size
        self._load_model()

    def _load_model(self):
        """加载Depth Anything V2模型"""
        try:
            # 尝试从transformers加载
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            # 正确的HuggingFace模型名称映射
            model_mapping = {
                "small": "depth-anything/Depth-Anything-V2-Small-hf",
                "base": "depth-anything/Depth-Anything-V2-Base-hf",
                "large": "depth-anything/Depth-Anything-V2-Large-hf",
                # 兼容旧配置
                "vits": "depth-anything/Depth-Anything-V2-Small-hf",
                "vitb": "depth-anything/Depth-Anything-V2-Base-hf",
                "vitl": "depth-anything/Depth-Anything-V2-Large-hf",
            }
            model_name = model_mapping.get(self.model_size.lower(), model_mapping["large"])
            print(f"加载 Depth Anything V2 模型: {model_name}")

            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.use_transformers = True
            print("成功加载 Depth Anything V2 (transformers)")

        except Exception as e:
            print(f"transformers加载失败: {e}")
            print("尝试从原始仓库加载...")
            self._load_from_original_repo()

    def _load_from_original_repo(self):
        """从原始仓库加载模型"""
        try:
            # 需要先克隆仓库: git clone https://github.com/DepthAnything/Depth-Anything-V2
            sys.path.append('./Depth-Anything-V2')
            from depth_anything_v2.dpt import DepthAnythingV2

            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }

            self.model = DepthAnythingV2(**model_configs[self.model_size])

            # 加载预训练权重
            ckpt_path = f'checkpoints/depth_anything_v2_{self.model_size}.pth'
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.use_transformers = False
            print("成功加载 Depth Anything V2 (原始仓库)")

        except Exception as e:
            raise RuntimeError(f"无法加载Depth Anything V2模型: {e}\n"
                             f"请确保已安装transformers或克隆了原始仓库")

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        预测深度图

        Args:
            image: RGB图像 (H, W, 3), uint8

        Returns:
            depth: 深度图 (H, W), float32
        """
        if self.use_transformers:
            # transformers版本
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            depth = outputs.predicted_depth

            # 插值到原始大小
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=image.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze().cpu().numpy()
        else:
            # 原始仓库版本
            depth = self.model.infer_image(image)

        return depth.astype(np.float32)


class MiDaSEstimator(DepthEstimator):
    """
    MiDaS深度估计器（备选方案）
    """

    def __init__(self, model_type: str = "DPT_Large", device: str = "cuda"):
        super().__init__(device)
        self.model_type = model_type
        self._load_model()

    def _load_model(self):
        """加载MiDaS模型"""
        self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.model.to(self.device)
        self.model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        print(f"成功加载 MiDaS 模型: {self.model_type}")

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        """预测深度图"""
        input_batch = self.transform(image).to(self.device)
        prediction = self.model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        return prediction.astype(np.float32)


def get_depth_estimator(config: Dict) -> DepthEstimator:
    """根据配置获取深度估计器"""
    model_name = config['depth']['model']
    device = config['depth']['device']

    if model_name == "depth_anything_v2":
        model_size = config['depth'].get('model_size', 'vitl')
        return DepthAnythingV2Estimator(model_size=model_size, device=device)
    elif model_name == "midas":
        return MiDaSEstimator(device=device)
    else:
        raise ValueError(f"不支持的深度模型: {model_name}")


def evaluate_depth_consistency(config: Dict,
                                save_vis: bool = True) -> Dict[str, Dict]:
    """
    评测深度一致性

    对于每对(gen, gt)图像：
    1. 对两者分别进行深度估计
    2. 比较depth(gen)和depth(gt)的一致性

    Args:
        config: 配置字典
        save_vis: 是否保存可视化

    Returns:
        每个相机的评测结果
    """
    # 初始化深度估计器
    print("\n" + "=" * 60)
    print("深度一致性评测")
    print("=" * 60)

    estimator = get_depth_estimator(config)

    # 获取图像对
    image_pairs = get_image_pairs(config)
    if not image_pairs:
        raise ValueError("没有找到有效的图像对")

    # 创建输出目录
    output_dir = config['output']['depth_maps']
    if save_vis:
        ensure_dir(output_dir)

    results = {}

    for camera, pairs in image_pairs.items():
        print(f"\n处理相机: {camera}")
        camera_metrics = []

        # 创建相机子目录
        if save_vis:
            camera_out_dir = os.path.join(output_dir, camera)
            ensure_dir(camera_out_dir)

        for gen_path, gt_path in tqdm(pairs, desc=f"  {camera}"):
            filename = os.path.basename(gen_path).replace('.png', '')

            # 加载图像
            gen_img = load_image(gen_path)
            gt_img = load_image(gt_path)

            # 深度估计
            depth_gen = estimator.predict(gen_img)
            depth_gt = estimator.predict(gt_img)

            # 对齐深度尺度（单目深度是相对深度）
            depth_gen_aligned = align_depth_scale(depth_gen, depth_gt, method="median")

            # 计算指标
            metrics = compute_depth_metrics(depth_gen_aligned, depth_gt)
            correlation = compute_depth_correlation(depth_gen_aligned, depth_gt)
            metrics.update(correlation)

            camera_metrics.append(metrics)

            # 保存可视化
            if save_vis:
                save_depth_visualization(
                    depth_gen,
                    os.path.join(camera_out_dir, f"{filename}_gen_depth.png")
                )
                save_depth_visualization(
                    depth_gt,
                    os.path.join(camera_out_dir, f"{filename}_gt_depth.png")
                )

        # 汇总该相机的指标
        results[camera] = aggregate_metrics(camera_metrics)
        print(f"\n  {camera} 深度指标:")
        print(format_metrics_table(results[camera]))

    # 计算总体平均
    all_metrics = []
    for camera_results in results.values():
        all_metrics.append({k: v for k, v in camera_results.items()
                          if not k.endswith('_std')})
    results['overall'] = aggregate_metrics(all_metrics)

    print("\n" + "=" * 60)
    print("总体深度评测结果:")
    print(format_metrics_table(results['overall']))

    return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="深度一致性评测")
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
    results = evaluate_depth_consistency(config, save_vis=not args.no_vis)

    # 保存结果
    output_path = args.output or os.path.join(config['output']['metrics'], "depth_results.json")
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
