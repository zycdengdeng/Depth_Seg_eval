"""
评测指标计算
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# ============== 深度估计指标 ==============

def compute_depth_metrics(pred: np.ndarray, gt: np.ndarray,
                          mask: Optional[np.ndarray] = None,
                          min_depth: float = 1e-3,
                          max_depth: float = None) -> Dict[str, float]:
    """
    计算深度估计指标

    Args:
        pred: 预测深度图
        gt: 真值深度图
        mask: 有效区域mask（可选）
        min_depth: 最小有效深度阈值，过滤掉太小的值避免除法爆炸
        max_depth: 最大有效深度阈值（可选）

    Returns:
        包含各项指标的字典
    """
    if mask is None:
        # 过滤无效值和极端值
        mask = (gt > min_depth) & (pred > min_depth)
        if max_depth is not None:
            mask = mask & (gt < max_depth) & (pred < max_depth)

    pred_valid = pred[mask]
    gt_valid = gt[mask]

    if len(pred_valid) == 0:
        return {
            'abs_rel': float('nan'),
            'sq_rel': float('nan'),
            'rmse': float('nan'),
            'rmse_log': float('nan'),
            'delta_1': float('nan'),
            'delta_2': float('nan'),
            'delta_3': float('nan'),
        }

    # 所有指标使用相同的有效像素集合（标准做法，不做百分位数过滤）
    # 绝对相对误差 (Absolute Relative Error)
    abs_rel = np.mean(np.abs(pred_valid - gt_valid) / gt_valid)

    # 平方相对误差 (Squared Relative Error)
    sq_rel = np.mean(((pred_valid - gt_valid) ** 2) / gt_valid)

    # 均方根误差 (RMSE)
    rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))

    # 对数均方根误差 (RMSE log)
    rmse_log = np.sqrt(np.mean((np.log(pred_valid) - np.log(gt_valid)) ** 2))

    # 阈值准确率 (δ < threshold)
    thresh = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    delta_1 = np.mean(thresh < 1.25) * 100
    delta_2 = np.mean(thresh < 1.25 ** 2) * 100
    delta_3 = np.mean(thresh < 1.25 ** 3) * 100

    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'delta_1': delta_1,
        'delta_2': delta_2,
        'delta_3': delta_3,
    }


def compute_depth_correlation(pred: np.ndarray, gt: np.ndarray,
                              mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    计算深度图之间的相关性指标
    用于比较gen和gt图像经过同一深度模型后的深度一致性
    """
    if mask is None:
        mask = (gt > 0) & (pred > 0)

    pred_valid = pred[mask].flatten()
    gt_valid = gt[mask].flatten()

    if len(pred_valid) < 2:
        return {'pearson': float('nan'), 'spearman': float('nan')}

    # Pearson相关系数
    pearson = np.corrcoef(pred_valid, gt_valid)[0, 1]

    # Spearman等级相关系数
    from scipy.stats import spearmanr
    spearman, _ = spearmanr(pred_valid, gt_valid)

    return {
        'pearson': pearson,
        'spearman': spearman,
    }


def compute_depth_ssim(pred: np.ndarray, gt: np.ndarray,
                        window_size: int = 11) -> Dict[str, float]:
    """
    计算深度图的SSIM（结构相似性指数）
    SSIM天然对小空间偏移鲁棒，因为它基于局部窗口统计

    Args:
        pred: 预测深度图 (H, W)
        gt: 真值深度图 (H, W)
        window_size: SSIM窗口大小

    Returns:
        ssim: 结构相似性指数 (0-1, 越高越好)
    """
    from scipy.ndimage import uniform_filter

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # 归一化到0-1
    pred_min, pred_max = pred.min(), pred.max()
    gt_min, gt_max = gt.min(), gt.max()
    if pred_max - pred_min > 1e-8:
        pred_n = (pred - pred_min) / (pred_max - pred_min)
    else:
        pred_n = pred
    if gt_max - gt_min > 1e-8:
        gt_n = (gt - gt_min) / (gt_max - gt_min)
    else:
        gt_n = gt

    mu_pred = uniform_filter(pred_n, size=window_size)
    mu_gt = uniform_filter(gt_n, size=window_size)

    sigma_pred_sq = uniform_filter(pred_n ** 2, size=window_size) - mu_pred ** 2
    sigma_gt_sq = uniform_filter(gt_n ** 2, size=window_size) - mu_gt ** 2
    sigma_pred_gt = uniform_filter(pred_n * gt_n, size=window_size) - mu_pred * mu_gt

    # 修正负方差
    sigma_pred_sq = np.maximum(sigma_pred_sq, 0)
    sigma_gt_sq = np.maximum(sigma_gt_sq, 0)

    ssim_map = ((2 * mu_pred * mu_gt + C1) * (2 * sigma_pred_gt + C2)) / \
               ((mu_pred ** 2 + mu_gt ** 2 + C1) * (sigma_pred_sq + sigma_gt_sq + C2))

    ssim_val = np.mean(ssim_map)

    return {
        'ssim': ssim_val,
    }


# ============== Cityscapes 超类映射 ==============

# 19个细粒度类 -> 7个超类（官方Cityscapes分类层级）
CITYSCAPES_COARSE_MAP = {
    0: 0,   # road -> flat
    1: 0,   # sidewalk -> flat
    2: 1,   # building -> construction
    3: 1,   # wall -> construction
    4: 1,   # fence -> construction
    5: 2,   # pole -> object
    6: 2,   # traffic_light -> object
    7: 2,   # traffic_sign -> object
    8: 3,   # vegetation -> nature
    9: 3,   # terrain -> nature
    10: 4,  # sky -> sky
    11: 5,  # person -> human
    12: 5,  # rider -> human
    13: 6,  # car -> vehicle
    14: 6,  # truck -> vehicle
    15: 6,  # bus -> vehicle
    16: 6,  # train -> vehicle
    17: 6,  # motorcycle -> vehicle
    18: 6,  # bicycle -> vehicle
}

CITYSCAPES_COARSE_CLASSES = [
    'flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle'
]

NUM_COARSE_CLASSES = 7


def remap_to_coarse(seg: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """
    将19类细粒度分割图重映射为7类超类分割图

    Args:
        seg: 分割图 (H, W)，类别ID 0-18
        ignore_index: 无效像素值

    Returns:
        coarse_seg: 超类分割图 (H, W)，类别ID 0-6
    """
    lut = np.full(256, ignore_index, dtype=np.int64)
    for fine_id, coarse_id in CITYSCAPES_COARSE_MAP.items():
        lut[fine_id] = coarse_id

    safe_seg = np.clip(seg, 0, 255).astype(np.int64)
    return lut[safe_seg]


# 地面类别 trainId（road=0, sidewalk=1），对应 coarse class flat=0
FLAT_CLASS_IDS = {0, 1}


def _mask_flat_classes(seg: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """将地面类（road, sidewalk）设为 ignore，不参与评测"""
    seg = seg.copy()
    for cid in FLAT_CLASS_IDS:
        seg[seg == cid] = ignore_index
    return seg


def compute_segmentation_metrics_multilevel(
    pred: np.ndarray, gt: np.ndarray,
    num_classes: int = 19,
    ignore_index: int = 255,
    compute_coarse: bool = True,
    exclude_flat: bool = True,
) -> Dict[str, float]:
    """
    同时计算细粒度(19类)和超类(7类)的分割指标

    Args:
        pred: 预测分割图 (H, W)
        gt: 真值分割图 (H, W)
        num_classes: 细粒度类别数
        ignore_index: 忽略的标签值
        compute_coarse: 是否计算超类指标
        exclude_flat: 是否排除地面类（road/sidewalk）

    Returns:
        包含细粒度和超类指标的字典
    """
    if exclude_flat:
        pred = _mask_flat_classes(pred, ignore_index)
        gt = _mask_flat_classes(gt, ignore_index)

    fine_metrics = compute_segmentation_metrics(pred, gt, num_classes, ignore_index)

    if not compute_coarse:
        return fine_metrics

    pred_coarse = remap_to_coarse(pred, ignore_index=ignore_index)
    gt_coarse = remap_to_coarse(gt, ignore_index=ignore_index)
    coarse_metrics = compute_segmentation_metrics(
        pred_coarse, gt_coarse,
        num_classes=NUM_COARSE_CLASSES,
        ignore_index=ignore_index
    )

    result = dict(fine_metrics)
    for key, value in coarse_metrics.items():
        result[f'coarse_{key}'] = value

    return result


# ============== 语义分割指标 ==============

def compute_segmentation_metrics(pred: np.ndarray, gt: np.ndarray,
                                  num_classes: int = 19,
                                  ignore_index: int = 255) -> Dict[str, float]:
    """
    计算语义分割指标

    Args:
        pred: 预测分割图 (H, W)
        gt: 真值分割图 (H, W)
        num_classes: 类别数
        ignore_index: 忽略的标签值

    Returns:
        包含各项指标的字典
    """
    # 有效mask
    valid_mask = gt != ignore_index
    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]

    if len(pred_valid) == 0:
        return {
            'miou': float('nan'),
            'pixel_acc': float('nan'),
            'fwiou': float('nan'),
            'class_iou': [float('nan')] * num_classes,
        }

    # 计算混淆矩阵
    conf_matrix = compute_confusion_matrix(pred_valid, gt_valid, num_classes)

    # 像素准确率
    pixel_acc = np.diag(conf_matrix).sum() / conf_matrix.sum() * 100

    # 每类IoU
    intersection = np.diag(conf_matrix)
    union = conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - intersection
    iou_per_class = intersection / (union + 1e-10)

    # 只计算出现过的类别的mIoU
    valid_classes = union > 0
    miou = np.mean(iou_per_class[valid_classes]) * 100

    # 频率加权IoU
    freq = conf_matrix.sum(axis=1) / conf_matrix.sum()
    fwiou = np.sum(freq[valid_classes] * iou_per_class[valid_classes]) * 100

    return {
        'miou': miou,
        'pixel_acc': pixel_acc,
        'fwiou': fwiou,
        'class_iou': (iou_per_class * 100).tolist(),
    }


def compute_confusion_matrix(pred: np.ndarray, gt: np.ndarray,
                              num_classes: int) -> np.ndarray:
    """计算混淆矩阵"""
    mask = (gt >= 0) & (gt < num_classes)
    conf_matrix = np.bincount(
        num_classes * gt[mask].astype(int) + pred[mask].astype(int),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return conf_matrix


def compute_segmentation_consistency(pred_gen: np.ndarray, pred_gt: np.ndarray,
                                      ignore_index: int = 255,
                                      exclude_flat: bool = True) -> Dict[str, float]:
    """
    计算分割一致性指标
    比较gen图像和gt图像经过同一分割模型后的分割结果一致性

    Args:
        pred_gen: 对生成图像的分割预测
        pred_gt: 对真值图像的分割预测
        ignore_index: 忽略的标签值
        exclude_flat: 是否排除地面类（road/sidewalk）

    Returns:
        一致性指标
    """
    if exclude_flat:
        pred_gen = _mask_flat_classes(pred_gen, ignore_index)
        pred_gt = _mask_flat_classes(pred_gt, ignore_index)

    # 有效区域（两者都不是ignore的区域）
    valid_mask = (pred_gen != ignore_index) & (pred_gt != ignore_index)

    if valid_mask.sum() == 0:
        return {
            'consistency': float('nan'),
            'disagreement_rate': float('nan'),
        }

    pred_gen_valid = pred_gen[valid_mask]
    pred_gt_valid = pred_gt[valid_mask]

    # 一致性：两个预测相同的比例
    consistency = np.mean(pred_gen_valid == pred_gt_valid) * 100

    # 不一致率
    disagreement_rate = 100 - consistency

    return {
        'consistency': consistency,
        'disagreement_rate': disagreement_rate,
    }


# ============== 汇总指标 ==============

def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    汇总多个样本的指标

    Args:
        metrics_list: 每个样本的指标字典列表

    Returns:
        平均指标
    """
    if not metrics_list:
        return {}

    aggregated = {}
    # 收集所有样本中出现过的key（per_class指标可能不是每帧都有）
    keys = set()
    for m in metrics_list:
        keys.update(m.keys())

    for key in keys:
        raw_values = [m[key] for m in metrics_list if key in m]

        if not raw_values:
            continue

        # 检查是否是列表类型（如class_iou）
        if isinstance(raw_values[0], (list, np.ndarray)):
            # 对于class_iou等列表类型，直接计算nanmean
            aggregated[key] = np.nanmean(raw_values, axis=0).tolist()
        else:
            # 标量类型，过滤nan值
            values = [v for v in raw_values if not np.isnan(v)]
            if values:
                aggregated[key] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
            else:
                aggregated[key] = float('nan')

    return aggregated


def format_metrics_table(metrics: Dict[str, float],
                          metric_names: Optional[List[str]] = None) -> str:
    """格式化指标为表格字符串"""
    if metric_names is None:
        metric_names = list(metrics.keys())

    lines = []
    lines.append("-" * 50)
    lines.append(f"{'Metric':<20} {'Value':>15}")
    lines.append("-" * 50)

    for name in metric_names:
        if name in metrics and not name.endswith('_std') and name not in ('class_iou', 'coarse_class_iou'):
            value = metrics[name]
            std_key = f'{name}_std'
            if std_key in metrics:
                lines.append(f"{name:<20} {value:>12.4f} ± {metrics[std_key]:.4f}")
            else:
                if isinstance(value, float):
                    lines.append(f"{name:<20} {value:>15.4f}")
                else:
                    lines.append(f"{name:<20} {str(value):>15}")

    lines.append("-" * 50)
    return "\n".join(lines)
