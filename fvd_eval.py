"""
FVD (Fréchet Video Distance) 评测模块

使用I3D (Kinetics-400预训练) 提取视频特征，计算生成视频与真值视频的FVD。
这是视频生成论文的标配指标，衡量时序连贯性。

标准参数（对齐顶会）:
- 模型: I3D pretrained on Kinetics-400
- 帧数: 16帧/clip
- 分辨率: 224x224
- 特征层: logits (softmax前)

使用方法:
    python fvd_eval.py --config config.yaml
    python evaluate.py --task fvd --parallel --gpus 0,1,2,3,4,5,6
"""

import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

from utils import load_config, ensure_dir
from metrics import aggregate_metrics


def get_video_clips(config: Dict, clip_length: int = 16) -> Dict[str, List[Tuple[List[str], List[str]]]]:
    """
    将帧组织成视频片段

    根据文件名中的segment信息将帧分组，每clip_length帧组成一个clip。

    Args:
        config: 配置字典
        clip_length: 每个clip的帧数

    Returns:
        {camera: [(gen_frame_paths, gt_frame_paths), ...]}
        每个元素是一个clip，包含clip_length个帧路径
    """
    import glob

    root = config['data']['root']
    cameras = config['data']['cameras']
    gen_folder = config['data']['gen_folder']
    gt_folder = config['data']['gt_folder']

    all_clips = {}

    for camera in cameras:
        gen_dir = os.path.join(root, camera, gen_folder)
        gt_dir = os.path.join(root, camera, gt_folder)

        if not os.path.exists(gen_dir) or not os.path.exists(gt_dir):
            continue

        gen_files = sorted(glob.glob(os.path.join(gen_dir, "*.png")))

        if not gen_files:
            continue

        # 尝试从文件名中提取segment和frame信息
        # 常见格式: {segment}_{camera}_{frame_id}.png 或 {id}_{segment}_{camera}_{frame_id}.png
        segments = defaultdict(list)

        for gen_path in gen_files:
            filename = os.path.basename(gen_path).replace('.png', '')
            gt_path = os.path.join(gt_dir, os.path.basename(gen_path))

            if not os.path.exists(gt_path):
                continue

            # 尝试提取segment ID
            # 格式1: 031_seg01_front_wide_120fov_0001.png -> segment=031_seg01
            # 格式2: frame_0001.png -> segment=default
            parts = filename.split('_')

            # 尝试找segment标识
            seg_id = 'default'
            for i, part in enumerate(parts):
                if 'seg' in part.lower():
                    # 找到seg标识，用它之前的部分+seg部分作为segment ID
                    seg_id = '_'.join(parts[:i+1])
                    break

            segments[seg_id].append((gen_path, gt_path))

        # 将每个segment的帧按文件名排序，分成clips
        camera_clips = []
        for seg_id, pairs in segments.items():
            # 按文件名排序确保时间顺序
            pairs.sort(key=lambda x: os.path.basename(x[0]))

            # 切分成clip_length的片段
            for i in range(0, len(pairs) - clip_length + 1, clip_length):
                clip_pairs = pairs[i:i + clip_length]
                gen_paths = [p[0] for p in clip_pairs]
                gt_paths = [p[1] for p in clip_pairs]
                camera_clips.append((gen_paths, gt_paths))

        if camera_clips:
            all_clips[camera] = camera_clips
            print(f"相机 {camera}: {len(camera_clips)} 个视频clip "
                  f"(每clip {clip_length}帧, 来自{len(segments)}个segment)")

    return all_clips


def load_video_clip(frame_paths: List[str], resolution: int = 224) -> torch.Tensor:
    """
    加载一个视频clip

    Args:
        frame_paths: 帧文件路径列表
        resolution: 目标分辨率

    Returns:
        tensor: (T, C, H, W), float32, [0, 1]
    """
    frames = []
    for path in frame_paths:
        img = Image.open(path).convert('RGB')
        img = img.resize((resolution, resolution), Image.BILINEAR)
        frame = torch.from_numpy(np.array(img)).float() / 255.0
        frame = frame.permute(2, 0, 1)  # (C, H, W)
        frames.append(frame)

    return torch.stack(frames, dim=0)  # (T, C, H, W)


class I3DFeatureExtractor:
    """
    I3D特征提取器 (使用cd-fvd或自定义实现)

    优先使用cd-fvd库（CVPR 2024标准），
    回退到torchvision的video模型。
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.model = None
        self.method = None
        self._load_model()

    def _load_model(self):
        """加载I3D模型，优先cd-fvd，回退torchvision"""
        # 方法1: 使用cd-fvd (CVPR 2024标准)
        try:
            from cdfvd import FVD as CDFvd
            self.cdfvd = CDFvd(self.device)
            self.method = 'cdfvd'
            print("FVD模型: cd-fvd (I3D Kinetics-400, CVPR 2024标准)")
            return
        except ImportError:
            pass

        # 方法2: 使用torchvision的video模型
        try:
            from torchvision.models.video import r3d_18, R3D_18_Weights
            print("FVD模型: torchvision R3D-18 (Kinetics-400)")
            weights = R3D_18_Weights.KINETICS400_V1
            self.model = r3d_18(weights=weights)
            # 移除最后的fc层，使用avgpool后的特征
            self.model.fc = torch.nn.Identity()
            self.model.to(self.device)
            self.model.eval()
            self.method = 'r3d_18'
            return
        except Exception as e:
            print(f"torchvision R3D-18 加载失败: {e}")

        raise RuntimeError(
            "无法加载FVD模型。请安装: pip install cd-fvd 或确保torchvision版本支持video模型"
        )

    @torch.no_grad()
    def extract_features(self, videos: torch.Tensor) -> np.ndarray:
        """
        提取视频特征

        Args:
            videos: (N, T, C, H, W), float32, [0, 1]

        Returns:
            features: (N, D) numpy array
        """
        if self.method == 'cdfvd':
            # cd-fvd期望输入: (N, T, C, H, W), uint8 [0, 255]
            videos_uint8 = (videos * 255).to(torch.uint8)
            features = self.cdfvd.compute_feats(videos_uint8)
            if isinstance(features, torch.Tensor):
                return features.cpu().numpy()
            return np.array(features)

        elif self.method == 'r3d_18':
            # R3D-18期望输入: (N, C, T, H, W)
            videos = videos.permute(0, 2, 1, 3, 4)  # (N, T, C, H, W) -> (N, C, T, H, W)
            videos = videos.to(self.device)

            # 标准化 (Kinetics-400)
            mean = torch.tensor([0.43216, 0.394666, 0.37645]).reshape(1, 3, 1, 1, 1).to(self.device)
            std = torch.tensor([0.22803, 0.22145, 0.216989]).reshape(1, 3, 1, 1, 1).to(self.device)
            videos = (videos - mean) / std

            features_list = []
            batch_size = 4
            for i in range(0, len(videos), batch_size):
                batch = videos[i:i + batch_size]
                feats = self.model(batch)  # (batch, 512)
                features_list.append(feats.cpu().numpy())

            return np.concatenate(features_list, axis=0)


def compute_frechet_distance(mu1, sigma1, mu2, sigma2):
    """
    计算两个多元高斯分布的Fréchet距离

    FD = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    """
    from scipy import linalg

    diff = mu1 - mu2

    # 计算 sqrt(sigma1 @ sigma2)
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    # 处理数值不稳定
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fvd = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fvd)


def compute_fvd(gen_features: np.ndarray, gt_features: np.ndarray) -> float:
    """
    从特征计算FVD

    Args:
        gen_features: (N, D) 生成视频特征
        gt_features: (N, D) 真值视频特征

    Returns:
        FVD分数, 越低越好
    """
    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False)

    mu_gt = np.mean(gt_features, axis=0)
    sigma_gt = np.cov(gt_features, rowvar=False)

    return compute_frechet_distance(mu_gen, sigma_gen, mu_gt, sigma_gt)


def evaluate_fvd(config: Dict, save_vis: bool = False) -> Dict[str, Dict]:
    """
    评测FVD (Fréchet Video Distance)

    Args:
        config: 配置字典
        save_vis: 未使用

    Returns:
        每个相机的FVD结果
    """
    print("\n" + "=" * 60)
    print("FVD (Fréchet Video Distance) 评测")
    print("=" * 60)

    fvd_config = config.get('fvd', {})
    clip_length = fvd_config.get('clip_length', 16)
    resolution = fvd_config.get('resolution', 224)
    device = 'cuda:0'  # FVD在主进程cuda:0上计算

    # 获取视频clips
    print(f"\n参数: clip_length={clip_length}, resolution={resolution}")
    video_clips = get_video_clips(config, clip_length=clip_length)

    if not video_clips:
        raise ValueError("没有找到有效的视频clips")

    # 加载I3D特征提取器
    extractor = I3DFeatureExtractor(device=device)

    results = {}

    for camera, clips in video_clips.items():
        print(f"\n处理相机: {camera} ({len(clips)} clips)")

        gen_features_list = []
        gt_features_list = []

        for gen_paths, gt_paths in tqdm(clips, desc=f"  {camera}"):
            # 加载视频clips
            gen_clip = load_video_clip(gen_paths, resolution=resolution)
            gt_clip = load_video_clip(gt_paths, resolution=resolution)

            # 添加batch维度: (T,C,H,W) -> (1,T,C,H,W)
            gen_clip = gen_clip.unsqueeze(0)
            gt_clip = gt_clip.unsqueeze(0)

            # 提取特征
            gen_feat = extractor.extract_features(gen_clip)
            gt_feat = extractor.extract_features(gt_clip)

            gen_features_list.append(gen_feat)
            gt_features_list.append(gt_feat)

        gen_features = np.concatenate(gen_features_list, axis=0)
        gt_features = np.concatenate(gt_features_list, axis=0)

        # 计算FVD
        fvd_score = compute_fvd(gen_features, gt_features)

        results[camera] = {
            'fvd': fvd_score,
            'num_clips': len(clips),
            'clip_length': clip_length,
        }

        print(f"  {camera} FVD = {fvd_score:.2f} ({len(clips)} clips)")

    # 计算总体FVD (所有相机特征合并)
    all_gen = []
    all_gt = []
    all_fvds = []
    for camera, clips in video_clips.items():
        if camera in results:
            all_fvds.append(results[camera]['fvd'])

    if all_fvds:
        results['overall'] = {
            'fvd': np.mean(all_fvds),
            'fvd_std': np.std(all_fvds),
            'num_cameras': len(all_fvds),
        }

    print(f"\n总体FVD = {results['overall']['fvd']:.2f} ± {results['overall']['fvd_std']:.2f}")

    return results


def main():
    """主函数"""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="FVD评测")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    results = evaluate_fvd(config)

    output_path = args.output or os.path.join(
        config['output']['metrics'], "fvd_results.json"
    )
    ensure_dir(os.path.dirname(output_path))

    from evaluate import convert_to_serializable
    with open(output_path, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
