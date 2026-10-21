"""
评测指标计算
"""

import warnings
import numpy as np
from typing import Dict, List, Optional, Tuple


# ============== 深度估计指标 ==============

def compute_depth_metrics(pred: np.ndarray, gt: np.ndarray,
                          mask: Optional[np.ndarray] = None,
                          min_depth: float = 1e-3,
                          max_depth: float = None) -> Dict[str, float]:
    """
    计算深度估计指标（Eigen et al. 标准协议）

    输入必须是**深度**（不是视差），且pred已与gt对齐尺度。
    有效区域只由gt决定；pred不参与筛选（否则pred的坏值会被剔除而不计入误差）。
    不做任何截尾/离群点过滤。

    Args:
        pred: 预测深度图
        gt: 真值深度图
        mask: 有效区域mask（可选，默认 gt > min_depth）
        min_depth: 最小有效深度，pred会被clamp到该值以避免log/除零
        max_depth: 最大有效深度阈值（可选）

    Returns:
        包含各项指标的字典
    """
    if mask is None:
        mask = np.isfinite(gt) & (gt > min_depth)
        if max_depth is not None:
            mask = mask & (gt < max_depth)

    gt_valid = gt[mask]
    pred_valid = np.clip(pred[mask], min_depth, max_depth)

    if len(gt_valid) == 0:
        return {
            'abs_rel': float('nan'),
            'sq_rel': float('nan'),
            'rmse': float('nan'),
            'rmse_log': float('nan'),
            'delta_1': float('nan'),
            'delta_2': float('nan'),
            'delta_3': float('nan'),
        }

    diff = pred_valid - gt_valid

    # 绝对相对误差 (Absolute Relative Error)
    abs_rel = np.mean(np.abs(diff) / gt_valid)

    # 平方相对误差 (Squared Relative Error)
    sq_rel = np.mean(diff ** 2 / gt_valid)

    # 均方根误差 (RMSE)
    rmse = np.sqrt(np.mean(diff ** 2))

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


def align_disparity_affine(disp_pred: np.ndarray, disp_gt: np.ndarray,
                           mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    视差空间最小二乘 scale + shift 对齐（相对视差是仿射不变的，只对齐scale不够）

    Args:
        disp_pred: 预测视差 (H, W)
        disp_gt: 参照视差 (H, W)
        mask: 参与拟合的像素（可选，默认全部有限值）

    Returns:
        对齐后的预测视差 (H, W), float64
    """
    if mask is None:
        mask = np.isfinite(disp_pred) & np.isfinite(disp_gt)
    x = disp_pred[mask].astype(np.float64)
    y = disp_gt[mask].astype(np.float64)
    A = np.stack([x, np.ones_like(x)], axis=1)
    scale, shift = np.linalg.lstsq(A, y, rcond=None)[0]
    return scale * disp_pred.astype(np.float64) + shift


def prepare_relative_depth_pair(disp_pred: np.ndarray, disp_gt: np.ndarray,
                                max_depth_ratio: float = 100.0
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    把相对视差（Depth Anything V2 / MiDaS 的输出，仿射不变）转成可比较的深度对

    步骤（MiDaS / Depth Anything 论文的评测协议）：
    1. gt有效区域：视差 > 近处视差(99分位) / max_depth_ratio，
       即只评测深度在 [最近处, 最近处 × max_depth_ratio] 内的像素，排除天空等无穷远区域
    2. 在视差空间用最小二乘拟合 scale + shift，把pred对齐到gt
    3. pred视差clamp到有效下限后取倒数得到深度
    4. 两者再同除以gt深度中位数，使rmse/sq_rel以"场景中位深度"为单位，跨帧可比

    Args:
        disp_pred: 预测视差 (H, W)
        disp_gt: 参照视差 (H, W)
        max_depth_ratio: 有效深度范围（最远/最近）

    Returns:
        depth_pred: 对齐后的预测深度 (H, W)
        depth_gt: 参照深度 (H, W)，无效区域为0
        valid: 有效区域mask (H, W)
    """
    valid = np.isfinite(disp_gt) & np.isfinite(disp_pred) & (disp_gt > 0)
    if valid.sum() < 2:
        zeros = np.zeros_like(disp_gt, dtype=np.float64)
        return zeros, zeros, np.zeros_like(valid)

    min_disp = np.percentile(disp_gt[valid], 99) / max_depth_ratio
    valid &= disp_gt > min_disp
    if valid.sum() < 2:
        zeros = np.zeros_like(disp_gt, dtype=np.float64)
        return zeros, zeros, valid

    disp_pred_aligned = np.maximum(align_disparity_affine(disp_pred, disp_gt, valid), min_disp)
    depth_pred = 1.0 / disp_pred_aligned
    depth_gt = np.where(valid, 1.0 / np.maximum(disp_gt, min_disp), 0.0)

    norm = np.median(depth_gt[valid])
    return depth_pred / norm, depth_gt / norm, valid


def evaluate_depth_pair(disp_pred: np.ndarray, disp_gt: np.ndarray,
                        max_depth_ratio: float = 100.0,
                        with_tolerant: bool = True) -> Dict[str, float]:
    """
    对一对相对视差图计算全部深度指标

    Args:
        disp_pred: gen图像的视差 (H, W)
        disp_gt: gt图像的视差 (H, W)
        max_depth_ratio: 见 prepare_relative_depth_pair
        with_tolerant: 是否额外计算空间对齐/局部容差指标

    Returns:
        指标字典
    """
    depth_pred, depth_gt, valid = prepare_relative_depth_pair(
        disp_pred, disp_gt, max_depth_ratio=max_depth_ratio
    )

    metrics = compute_depth_metrics(depth_pred, depth_gt, mask=valid)
    # 相关性和SSIM对仿射变换不变，直接在原始视差上算
    metrics.update(compute_depth_correlation(disp_pred, disp_gt))
    metrics.update(compute_depth_ssim(disp_pred, disp_gt))

    if not with_tolerant:
        return metrics

    from utils import align_spatial, compute_tolerant_metrics

    # 1. 全局空间对齐：先用互相关平移视差，再重新做仿射对齐和深度转换
    disp_pred_shifted, (dy, dx) = align_spatial(disp_pred, disp_gt, max_shift=20)
    depth_pred_shifted, depth_gt_s, valid_s = prepare_relative_depth_pair(
        disp_pred_shifted, disp_gt, max_depth_ratio=max_depth_ratio
    )
    aligned_metrics = compute_depth_metrics(depth_pred_shifted, depth_gt_s, mask=valid_s)
    metrics['aligned_abs_rel'] = aligned_metrics['abs_rel']
    metrics['aligned_delta_1'] = aligned_metrics['delta_1']
    metrics['aligned_rmse'] = aligned_metrics['rmse']
    metrics['shift_dx'] = float(dx)
    metrics['shift_dy'] = float(dy)

    # 2. 局部窗口容差：每个像素在5x5窗口内找最优匹配
    best_match_gt = compute_tolerant_metrics(depth_pred, depth_gt, window_size=5, mask=valid)
    tolerant_metrics = compute_depth_metrics(depth_pred, best_match_gt, mask=valid)
    metrics['tolerant_abs_rel'] = tolerant_metrics['abs_rel']
    metrics['tolerant_delta_1'] = tolerant_metrics['delta_1']
    metrics['tolerant_rmse'] = tolerant_metrics['rmse']

    return metrics


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
    同时计算细粒度(19类)和超类(7类, Cityscapes category)的分割指标

    返回值中带有 conf_matrix / coarse_conf_matrix，
    aggregate_metrics 会把它们累加，得到数据集级 mIoU（Cityscapes官方算法）。

    Args:
        pred: 预测分割图 (H, W)
        gt: 真值分割图 (H, W)
        num_classes: 细粒度类别数
        ignore_index: 忽略的标签值
        compute_coarse: 是否计算超类指标（仅Cityscapes 19类时有效）
        exclude_flat: 是否排除地面类（road/sidewalk）

    Returns:
        包含细粒度和超类指标的字典
    """
    if exclude_flat:
        pred = _mask_flat_classes(pred, ignore_index)
        gt = _mask_flat_classes(gt, ignore_index)

    fine_metrics = compute_segmentation_metrics(pred, gt, num_classes, ignore_index)

    if not compute_coarse or num_classes != len(CITYSCAPES_COARSE_MAP):
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

def segmentation_metrics_from_confusion(conf_matrix: np.ndarray) -> Dict[str, float]:
    """
    由混淆矩阵计算分割指标

    在pred和gt中都没出现的类（union为0）IoU记为NaN，不参与mIoU。

    Args:
        conf_matrix: (C, C+1) 混淆矩阵，行=gt，列=pred；
            最后一列是pred落在类别表之外（如被排除的地面类）的像素，只算作gt类的FN

    Returns:
        miou / pixel_acc / fwiou / class_iou
    """
    conf_matrix = np.asarray(conf_matrix, dtype=np.float64)
    num_classes = conf_matrix.shape[0]
    total = conf_matrix.sum()
    pred_per_class = conf_matrix[:, :num_classes].sum(axis=0)

    if total == 0:
        return {
            'miou': float('nan'),
            'pixel_acc': float('nan'),
            'fwiou': float('nan'),
            'class_iou': [float('nan')] * num_classes,
        }

    # 像素准确率
    intersection = np.diag(conf_matrix[:, :num_classes])
    pixel_acc = intersection.sum() / total * 100

    # 每类IoU，未出现的类为NaN
    union = conf_matrix.sum(axis=1) + pred_per_class - intersection
    valid_classes = union > 0
    iou_per_class = np.full(num_classes, np.nan)
    iou_per_class[valid_classes] = intersection[valid_classes] / union[valid_classes]

    miou = np.mean(iou_per_class[valid_classes]) * 100

    # 频率加权IoU
    freq = conf_matrix.sum(axis=1) / total
    fwiou = np.sum(freq[valid_classes] * iou_per_class[valid_classes]) * 100

    return {
        'miou': miou,
        'pixel_acc': pixel_acc,
        'fwiou': fwiou,
        'class_iou': (iou_per_class * 100).tolist(),
    }


def compute_segmentation_metrics(pred: np.ndarray, gt: np.ndarray,
                                  num_classes: int = 19,
                                  ignore_index: int = 255) -> Dict[str, float]:
    """
    计算单张图的语义分割指标

    Args:
        pred: 预测分割图 (H, W)
        gt: 真值分割图 (H, W)
        num_classes: 类别数
        ignore_index: 忽略的标签值

    Returns:
        包含各项指标的字典，以及该图的混淆矩阵 conf_matrix（用于数据集级累加）
    """
    valid_mask = gt != ignore_index
    conf_matrix = compute_confusion_matrix(pred[valid_mask], gt[valid_mask], num_classes)

    metrics = segmentation_metrics_from_confusion(conf_matrix)
    metrics['conf_matrix'] = conf_matrix
    return metrics


def compute_confusion_matrix(pred: np.ndarray, gt: np.ndarray,
                              num_classes: int) -> np.ndarray:
    """
    计算混淆矩阵 (C, C+1)，行=gt，列=pred

    有效像素只由gt决定；pred不在类别表内的像素（如被排除的地面类、ignore）
    记入最后一列，即算作该gt类的漏检，而不是被丢弃。
    """
    gt = gt.astype(np.int64)
    pred = pred.astype(np.int64)
    mask = (gt >= 0) & (gt < num_classes)
    gt = gt[mask]
    pred = pred[mask]
    pred = np.where((pred >= 0) & (pred < num_classes), pred, num_classes)
    conf_matrix = np.bincount(
        (num_classes + 1) * gt + pred,
        minlength=num_classes * (num_classes + 1)
    ).reshape(num_classes, num_classes + 1)
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

    # 有效区域只由参照图(gt)决定；gen预测成被排除类的像素算不一致
    valid_mask = pred_gt != ignore_index

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

    - 标量：均值 + 标准差（忽略nan）
    - 列表（如class_iou）：逐元素nanmean
    - *conf_matrix：累加，并据此重算数据集级 miou / pixel_acc / fwiou / class_iou
      （Cityscapes官方算法）；逐图mIoU的均值保留为 miou_img 作参考

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

        if key.endswith('conf_matrix'):
            aggregated[key] = np.sum([np.asarray(v, dtype=np.int64) for v in raw_values],
                                     axis=0).tolist()
        elif isinstance(raw_values[0], (list, np.ndarray)):
            # 对于class_iou等列表类型，逐元素nanmean（全为nan的类保持nan）
            with np.errstate(all='ignore'), warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                aggregated[key] = np.nanmean(np.asarray(raw_values, dtype=np.float64),
                                             axis=0).tolist()
        else:
            # 标量类型，过滤nan值
            values = [v for v in raw_values if not np.isnan(v)]
            if values:
                aggregated[key] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
            else:
                aggregated[key] = float('nan')

    _apply_dataset_level_segmentation(aggregated)
    return aggregated


def _apply_dataset_level_segmentation(aggregated: Dict) -> None:
    """用累加后的混淆矩阵覆盖分割指标（原地修改）"""
    for prefix in ('', 'coarse_'):
        cm_key = f'{prefix}conf_matrix'
        if cm_key not in aggregated:
            continue

        # 逐图mIoU均值保留作参考
        if f'{prefix}miou' in aggregated:
            aggregated[f'{prefix}miou_img'] = aggregated[f'{prefix}miou']
            if f'{prefix}miou_std' in aggregated:
                aggregated[f'{prefix}miou_img_std'] = aggregated[f'{prefix}miou_std']

        for name in ('miou', 'pixel_acc', 'fwiou'):
            aggregated.pop(f'{prefix}{name}_std', None)

        for name, value in segmentation_metrics_from_confusion(aggregated[cm_key]).items():
            aggregated[f'{prefix}{name}'] = value


def aggregate_overall(results: Dict[str, Dict]) -> Dict[str, float]:
    """
    汇总各相机结果为overall

    标量取各相机均值；分割的混淆矩阵跨相机累加，得到全数据集mIoU。

    Args:
        results: {camera: 该相机的aggregate_metrics结果}

    Returns:
        overall指标
    """
    all_metrics = []
    for camera, camera_results in results.items():
        if camera == 'overall' or not camera_results:
            continue
        all_metrics.append({
            k: v for k, v in camera_results.items()
            if not k.endswith('_std')
            and (k.endswith('conf_matrix') or not isinstance(v, (list, np.ndarray)))
        })
    return aggregate_metrics(all_metrics)


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
        if name in metrics and not name.endswith('_std') and not isinstance(metrics[name], (list, np.ndarray)):
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
