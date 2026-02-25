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

    # 使用百分位数过滤异常值 (去掉最极端的1%)
    ratio = pred_valid / gt_valid
    p1, p99 = np.percentile(ratio, [1, 99])
    inlier_mask = (ratio >= p1) & (ratio <= p99)
    pred_filtered = pred_valid[inlier_mask]
    gt_filtered = gt_valid[inlier_mask]

    # 绝对相对误差 (Absolute Relative Error) - 使用过滤后的数据
    abs_rel = np.mean(np.abs(pred_filtered - gt_filtered) / gt_filtered)

    # 平方相对误差 (Squared Relative Error) - 使用过滤后的数据
    sq_rel = np.mean(((pred_filtered - gt_filtered) ** 2) / gt_filtered)

    # 均方根误差 (RMSE) - 使用过滤后的数据
    rmse = np.sqrt(np.mean((pred_filtered - gt_filtered) ** 2))

    # 对数均方根误差 (RMSE log) - 使用过滤后的数据
    rmse_log = np.sqrt(np.mean((np.log(pred_filtered) - np.log(gt_filtered)) ** 2))

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
                                      ignore_index: int = 255) -> Dict[str, float]:
    """
    计算分割一致性指标
    比较gen图像和gt图像经过同一分割模型后的分割结果一致性

    Args:
        pred_gen: 对生成图像的分割预测
        pred_gt: 对真值图像的分割预测
        ignore_index: 忽略的标签值

    Returns:
        一致性指标
    """
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
    keys = metrics_list[0].keys()

    for key in keys:
        raw_values = [m[key] for m in metrics_list]

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
        if name in metrics and not name.endswith('_std') and name != 'class_iou':
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
