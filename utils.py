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


def safe_import_transformers():
    """
    导入transformers，自动绕过huggingface_hub版本元数据损坏的问题。
    当huggingface_hub的版本号无法被检测到时(found=None)，
    先patch版本检查函数，再导入transformers。
    """
    try:
        import transformers
        return transformers
    except ValueError as e:
        if "Unable to compare versions" not in str(e):
            raise
        print(f"[safe_import_transformers] 检测到版本检查失败，正在patch...")
        # 必须在import transformers之前patch，因为transformers/__init__.py
        # 在模块级别就调用了版本检查
        import importlib
        import sys

        # 直接加载versions子模块（不触发transformers/__init__）
        import importlib.util
        spec = importlib.util.find_spec("transformers.utils.versions")
        versions_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(versions_mod)

        _orig = versions_mod.require_version
        def _patched(requirement, hint=None):
            try:
                return _orig(requirement, hint)
            except ValueError:
                pass

        versions_mod.require_version = _patched
        versions_mod.require_version_core = _patched
        sys.modules["transformers.utils.versions"] = versions_mod

        # 同样patch dependency_versions_check，防止它重新导入versions
        if "transformers" in sys.modules:
            del sys.modules["transformers"]
        if "transformers.dependency_versions_check" in sys.modules:
            del sys.modules["transformers.dependency_versions_check"]

        import transformers
        return transformers


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
    对齐预测深度和真值深度的尺度
    这对于单目深度估计很重要，因为预测的是相对深度
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
                              min_depth: float = 1e-3) -> np.ndarray:
    """
    局部窗口容差：对于每个像素，在窗口内找最小误差的匹配

    Args:
        pred: 预测深度图 (H, W)
        gt: 真值深度图 (H, W)
        window_size: 搜索窗口大小（奇数）
        min_depth: 最小有效深度

    Returns:
        best_match_gt: 每个pred像素对应的最优gt匹配值
    """
    from scipy.ndimage import minimum_filter, maximum_filter

    pad = window_size // 2
    h, w = pred.shape

    # 对于每个pred像素，找gt窗口内使得 |pred-gt|/gt 最小的值
    # 近似：用gt的local min和local max来bound
    best_match_gt = np.copy(gt)

    mask = (gt > min_depth) & (pred > min_depth)

    # 对gt做sliding window，找每个位置的最小误差匹配
    ratio = np.where(mask, pred / gt, 1.0)

    # 如果pred > gt，最好的匹配是gt窗口里最大的值
    # 如果pred < gt，最好的匹配是gt窗口里最小的值
    gt_local_min = minimum_filter(gt, size=window_size)
    gt_local_max = maximum_filter(gt, size=window_size)

    # 选择使ratio更接近1的匹配
    best_match_gt = np.where(
        pred > gt,
        np.minimum(gt_local_max, pred),  # pred偏大时，取gt邻域最大值
        np.maximum(gt_local_min, pred),  # pred偏小时，取gt邻域最小值
    )
    # 但不能超出gt local范围
    best_match_gt = np.clip(best_match_gt, gt_local_min, gt_local_max)

    return best_match_gt


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
