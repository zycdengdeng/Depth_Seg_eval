"""
工具函数
"""

import os
import glob
import yaml
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch


def load_config(config_path: str = "config.yaml") -> Dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_image_pairs(config: Dict) -> Dict[str, List[Tuple[str, str]]]:
    """
    获取所有gen和gt图像对
    返回: {camera_name: [(gen_path, gt_path), ...]}
    """
    root = config['data']['root']
    cameras = config['data']['cameras']
    gen_folder = config['data']['gen_folder']
    gt_folder = config['data']['gt_folder']

    pairs = {}
    for camera in cameras:
        gen_dir = os.path.join(root, camera, gen_folder)
        gt_dir = os.path.join(root, camera, gt_folder)

        if not os.path.exists(gen_dir) or not os.path.exists(gt_dir):
            print(f"警告: 相机 {camera} 的目录不完整")
            continue

        gen_files = sorted(glob.glob(os.path.join(gen_dir, "*.png")))
        camera_pairs = []

        for gen_path in gen_files:
            filename = os.path.basename(gen_path)
            gt_path = os.path.join(gt_dir, filename)

            if os.path.exists(gt_path):
                camera_pairs.append((gen_path, gt_path))
            else:
                print(f"警告: 找不到对应的gt文件: {gt_path}")

        if camera_pairs:
            pairs[camera] = camera_pairs
            print(f"相机 {camera}: 找到 {len(camera_pairs)} 对图像")

    return pairs


def load_image(path: str, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """加载图像并可选调整大小"""
    img = Image.open(path).convert('RGB')
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    return np.array(img)


def load_image_pair(gen_path: str, gt_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载 gen/gt 图像对，并将 gt resize 到与 gen 相同的分辨率。

    gen 图像可能是 720x720，gt 图像可能是 3840x2160 或 1920x1080，
    分辨率不一致会导致像素级比较指标 (PSNR/SSIM/LPIPS) 出错，
    也会导致感知模型 (深度/分割/SAM) 对不同分辨率输入产生不可比的结果。

    策略：以 gen 尺寸为基准，将 gt resize 到 gen 尺寸。
    """
    gen_img = Image.open(gen_path).convert('RGB')
    gt_img = Image.open(gt_path).convert('RGB')

    if gen_img.size != gt_img.size:
        gt_img = gt_img.resize(gen_img.size, Image.BILINEAR)

    return np.array(gen_img), np.array(gt_img)


def ensure_dir(path: str):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_depth(depth: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    归一化深度图用于比较
    method: minmax, median, percentile
    """
    if method == "minmax":
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-8:
            return (depth - d_min) / (d_max - d_min)
        return depth
    elif method == "median":
        median = np.median(depth)
        if median > 1e-8:
            return depth / median
        return depth
    elif method == "percentile":
        p5, p95 = np.percentile(depth, [5, 95])
        depth_clipped = np.clip(depth, p5, p95)
        if p95 - p5 > 1e-8:
            return (depth_clipped - p5) / (p95 - p5)
        return depth_clipped
    else:
        return depth


def align_depth_scale(pred: np.ndarray, gt: np.ndarray, method: str = "median") -> np.ndarray:
    """
    对齐预测深度和真值深度的尺度（仅scale）

    注意：Depth Anything V2 / MiDaS 输出的是仿射不变的相对视差，
    评测请用 metrics.evaluate_depth_pair（视差空间scale+shift对齐后转深度）。
    """
    mask = (gt > 0) & (pred > 0)
    if mask.sum() == 0:
        return pred

    if method == "median":
        scale = np.median(gt[mask]) / np.median(pred[mask])
    elif method == "mean":
        scale = np.mean(gt[mask]) / np.mean(pred[mask])
    elif method == "least_squares":
        # 最小二乘法对齐
        scale = np.sum(gt[mask] * pred[mask]) / np.sum(pred[mask] ** 2)
    else:
        scale = 1.0

    return pred * scale


def align_spatial(pred: np.ndarray, gt: np.ndarray,
                  max_shift: int = 20) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    全局空间对齐：用互相关找最优平移量，补偿时间偏移导致的像素漂移

    Args:
        pred: 预测深度图 (H, W)
        gt: 真值深度图 (H, W)
        max_shift: 最大搜索平移量（像素）

    Returns:
        aligned_pred: 对齐后的预测深度图
        (dy, dx): 最优平移量
    """
    from scipy.signal import fftconvolve

    # 归一化后做互相关
    pred_norm = (pred - pred.mean()) / (pred.std() + 1e-8)
    gt_norm = (gt - gt.mean()) / (gt.std() + 1e-8)

    # FFT互相关
    cross_corr = fftconvolve(gt_norm, pred_norm[::-1, ::-1], mode='full')

    h, w = pred.shape
    # 互相关中心对应零位移
    center_y, center_x = h - 1, w - 1

    # 只在max_shift范围内搜索
    y_start = max(0, center_y - max_shift)
    y_end = min(cross_corr.shape[0], center_y + max_shift + 1)
    x_start = max(0, center_x - max_shift)
    x_end = min(cross_corr.shape[1], center_x + max_shift + 1)

    search_region = cross_corr[y_start:y_end, x_start:x_end]
    peak = np.unravel_index(search_region.argmax(), search_region.shape)

    dy = peak[0] + y_start - center_y
    dx = peak[1] + x_start - center_x

    # 应用平移
    from scipy.ndimage import shift
    aligned_pred = shift(pred, [dy, dx], order=1, mode='reflect')

    return aligned_pred, (int(dy), int(dx))


def compute_tolerant_metrics(pred: np.ndarray, gt: np.ndarray,
                              window_size: int = 5,
                              mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    局部窗口容差：对于每个像素，在gt的邻域窗口内找使 |pred-g|/g 最小的值

    该最优值恰为 clip(pred, 邻域最小值, 邻域最大值)。
    无效gt像素不参与邻域统计（避免天空等无效区域被当成匹配对象）。

    Args:
        pred: 预测深度图 (H, W)
        gt: 真值深度图 (H, W)
        window_size: 搜索窗口大小（奇数）
        mask: gt有效区域（可选，默认 gt > 0）

    Returns:
        best_match_gt: 每个pred像素对应的最优gt匹配值（无效区域保持原gt）
    """
    from scipy.ndimage import minimum_filter, maximum_filter

    if mask is None:
        mask = gt > 0

    gt_local_min = minimum_filter(np.where(mask, gt, np.inf), size=window_size)
    gt_local_max = maximum_filter(np.where(mask, gt, -np.inf), size=window_size)

    best_match_gt = np.clip(pred, gt_local_min, gt_local_max)
    return np.where(mask, best_match_gt, gt)


def save_depth_visualization(depth: np.ndarray, path: str, cmap: str = "magma"):
    """保存深度图可视化"""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    plt.imshow(depth, cmap=cmap)
    plt.colorbar(label='Depth')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def save_segmentation_visualization(seg: np.ndarray, path: str, palette: Optional[np.ndarray] = None):
    """保存分割图可视化"""
    if palette is None:
        # Cityscapes调色板
        palette = get_cityscapes_palette()

    h, w = seg.shape
    color_seg = np.zeros((h, w, 3), dtype=np.uint8)

    for label_id in np.unique(seg):
        if label_id < len(palette):
            color_seg[seg == label_id] = palette[label_id]

    Image.fromarray(color_seg).save(path)


def get_cityscapes_palette() -> np.ndarray:
    """获取Cityscapes颜色调色板"""
    return np.array([
        [128, 64, 128],   # road
        [244, 35, 232],   # sidewalk
        [70, 70, 70],     # building
        [102, 102, 156],  # wall
        [190, 153, 153],  # fence
        [153, 153, 153],  # pole
        [250, 170, 30],   # traffic light
        [220, 220, 0],    # traffic sign
        [107, 142, 35],   # vegetation
        [152, 251, 152],  # terrain
        [70, 130, 180],   # sky
        [220, 20, 60],    # person
        [255, 0, 0],      # rider
        [0, 0, 142],      # car
        [0, 0, 70],       # truck
        [0, 60, 100],     # bus
        [0, 80, 100],     # train
        [0, 0, 230],      # motorcycle
        [119, 11, 32],    # bicycle
    ], dtype=np.uint8)


# Cityscapes类别名称
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle'
]
