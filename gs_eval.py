#!/usr/bin/env python3
"""
GS方法评测脚本

对多个3D Gaussian Splatting方法的渲染结果进行统一评测。
支持5种GS方法，自动匹配GT图像，运行全部评测指标。

使用方法：
    python gs_eval.py --task all
    python gs_eval.py --task image_metrics --methods DDGS DropGaussian
    python gs_eval.py --task nta --clips 003_car0325_road0327_t3
    python gs_eval.py --task depth --cameras FL FN

目录结构：
    方法1-3 (DDGS/DropGaussian/Co-Adaptation):
        {root}/{clip}/{timestamp}/vehicle_renders/{camera}/render.png
    方法4 (SparseGS):
        {root}/scene{NNN}_{dist}/vehicle_renders/{camera}/render.png
    方法5 (S3Gaussian):
        {root}/scene{NNN}_{dist}/{camera}.png
    GT:
        /mnt/car_road_data_TianJin/{clip}/car/images/{camera}/addc_*_{vehicle_ts}.jpg
"""

import os
import sys
import glob
import json
import argparse
import re
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# ============== 时间戳映射表 ==============
# 路侧毫秒时间戳，从标注表格获取
# {clip_name: {"near": ts, "middle": ts, "far": ts}}

CLIP_TIMESTAMPS = {
    "003_car0325_road0327_t3":  {"near": "1742877822545", "middle": "1742877826207", "far": "1742877830287"},
    "004_car0325_road0327_t4":  {"near": "1742878212980", "middle": "1742878217631", "far": "1742878220772"},
    "009_car0325_road0327_t10": {"near": "1742880304953", "middle": "1742880308674", "far": "1742880311421"},
    "015_car0325_road0327_t18": {"near": "1742884382100", "middle": "1742884384678", "far": "1742884386580"},
    "020_car0325_road0327_t25": {"near": "1742887514589", "middle": "1742887517308", "far": "1742887519629"},
    "031_car0402_road0402_t9":  {"near": "1743572672542", "middle": "1743572675860", "far": "1743572679422"},
    "035_car0402_road0402_t13": {"near": "1743574755822", "middle": "1743574759909", "far": "1743574763720"},
    "039_car0402_road0402_t17": {"near": "1743576182579", "middle": "1743576186959", "far": "1743576190790"},
    "050_car0402_road0402_t28": {"near": "1743582087079", "middle": "1743582090530", "far": "1743582093290"},
    "055_car0402_road0402_t33": {"near": "1743583783950", "middle": "1743583787671", "far": "1743583791002"},
    "056_car0402_road0402_t34": {"near": "1743584168546", "middle": "1743584171142", "far": "1743584173593"},
    "059_car0402_road0402_t37": {"near": "1743584970206", "middle": "1743584972552", "far": "1743584975161"},
    "063_car0402_road0402_t41": {"near": "1743587038293", "middle": "1743587040355", "far": "1743587042705"},
    "076_car0402_road0402_t54": {"near": "1743646656417", "middle": "1743646658638", "far": "1743646661123"},
    "082_car0402_road0402_t64": {"near": "1743650233549", "middle": "1743650237382", "far": "1743650240522"},
    "085_car0402_road0402_t67": {"near": "1743651916518", "middle": "1743651920099", "far": "1743651923810"},
    "086_car0402_road0402_t68": {"near": "1743652311985", "middle": "1743652315384", "far": "1743652319536"},
    "088_car0402_road0402_t70": {"near": "1743652835642", "middle": "1743652838934", "far": "1743652843091"},
}

# 从clip名提取场景编号（如 "003_car0325_road0327_t3" → "003"）
def _clip_to_scene_num(clip: str) -> str:
    return clip.split("_")[0]

# 建立时间戳→(clip, distance)的反向索引
_TS_TO_CLIP_DIST = {}
for _clip, _dists in CLIP_TIMESTAMPS.items():
    for _dist, _ts in _dists.items():
        _TS_TO_CLIP_DIST[_ts] = (_clip, _dist)


# ============== 相机列表 ==============
CAMERAS = ["FL", "FN", "FR", "FW", "RL", "RN", "RR"]
DISTANCES = ["near", "middle", "far"]


# ============== GS方法定义 ==============

METHODS = {
    "DDGS": {
        "root": "/mnt/myn/project/DDGS/output",
        "type": "timestamp",
    },
    "DropGaussian": {
        "root": "/mnt/myn/project/DropGaussian_release/output",
        "type": "timestamp",
    },
    "Co-Adaptation": {
        "root": "/mnt/myn/project/Co-Adaptation-of-3DGS/output",
        "type": "timestamp",
    },
    "AD-GS": {
        "root": "/mnt/myn/project/AD-GS/output",
        "type": "timestamp",
    },
    "SparseGS": {
        "root": "/mnt/zyc_wzh/SparseGS/output/car_road",
        "type": "scene_dist",
    },
    "S3Gaussian": {
        "root": "/mnt/zyc_wzh/S3Gaussian/work_dirs/roadside_colmap/renders",
        "type": "scene_flat",
    },
}

# GT数据根目录
GT_ROOT = "/mnt/car_road_data_TianJin"


# ============== 路径构建 ==============

def get_gen_path(method_name: str, clip: str, distance: str, camera: str) -> str:
    """构建GS方法渲染图像的路径"""
    method = METHODS[method_name]
    root = method["root"]
    mtype = method["type"]

    if mtype == "timestamp":
        ts = CLIP_TIMESTAMPS[clip][distance]
        return os.path.join(root, clip, ts, "vehicle_renders", camera, "render.png")

    elif mtype == "scene_dist":
        scene_num = _clip_to_scene_num(clip)
        scene_dir = f"scene{scene_num}_{distance}"
        return os.path.join(root, scene_dir, f"vehicle_render_{distance}",
                            "vehicle_renders", camera, "render.png")

    elif mtype == "scene_flat":
        scene_num = _clip_to_scene_num(clip)
        scene_dir = f"scene{scene_num}_{distance}"
        return os.path.join(root, scene_dir, f"{camera}.png")

    else:
        raise ValueError(f"Unknown method type: {mtype}")


def _parse_vehicle_timestamp(filename: str) -> Optional[float]:
    """从车端图像文件名中提取时间戳（秒）

    文件名格式: addc_2025-03-25-12-30-26_0_1742877030.849978.jpg
    提取最后的 1742877030.849978
    """
    match = re.search(r'_(\d+\.\d+)\.jpg$', filename)
    if match:
        return float(match.group(1))
    return None


# GT文件名缓存：{(clip, camera): [(vehicle_ts_float, filepath), ...]}
_gt_file_cache: Dict[Tuple[str, str], List[Tuple[float, str]]] = {}


def _build_gt_cache(clip: str, camera: str) -> List[Tuple[float, str]]:
    """构建GT目录的时间戳索引（排序后缓存）"""
    key = (clip, camera)
    if key in _gt_file_cache:
        return _gt_file_cache[key]

    gt_dir = os.path.join(GT_ROOT, clip, "car", "images", camera)
    entries = []

    if os.path.isdir(gt_dir):
        for fname in os.listdir(gt_dir):
            if not fname.endswith('.jpg'):
                continue
            ts = _parse_vehicle_timestamp(fname)
            if ts is not None:
                entries.append((ts, os.path.join(gt_dir, fname)))
        entries.sort(key=lambda x: x[0])

    _gt_file_cache[key] = entries
    return entries


def find_gt_path(clip: str, distance: str, camera: str,
                 max_diff_ms: float = 100.0) -> Optional[str]:
    """找到与指定路侧时间戳最近的GT车端图像

    Args:
        clip: clip名称
        distance: near/middle/far
        camera: 相机名称 (FL/FN/FR等)
        max_diff_ms: 最大允许时间差（毫秒）

    Returns:
        GT图像路径，找不到则返回None
    """
    road_ts_ms = int(CLIP_TIMESTAMPS[clip][distance])
    target_ts_sec = road_ts_ms / 1000.0  # 转为车端秒时间戳

    entries = _build_gt_cache(clip, camera)
    if not entries:
        return None

    # 二分查找最近的时间戳
    import bisect
    timestamps = [e[0] for e in entries]
    idx = bisect.bisect_left(timestamps, target_ts_sec)

    best_idx = None
    best_diff = float('inf')

    for candidate in [idx - 1, idx]:
        if 0 <= candidate < len(entries):
            diff = abs(entries[candidate][0] - target_ts_sec)
            if diff < best_diff:
                best_diff = diff
                best_idx = candidate

    if best_idx is not None and best_diff * 1000 <= max_diff_ms:
        return entries[best_idx][1]

    return None


# ============== 图像对收集 ==============

def collect_image_pairs(methods: List[str] = None,
                        clips: List[str] = None,
                        cameras: List[str] = None,
                        distances: List[str] = None,
                        ) -> Dict[str, List[Dict]]:
    """收集所有 (gen, gt) 图像对

    Returns:
        {method_name: [
            {"gen_path": ..., "gt_path": ..., "clip": ..., "distance": ..., "camera": ...},
            ...
        ]}
    """
    if methods is None:
        methods = list(METHODS.keys())
    if clips is None:
        clips = list(CLIP_TIMESTAMPS.keys())
    if cameras is None:
        cameras = CAMERAS
    if distances is None:
        distances = DISTANCES

    result = {}
    for method in methods:
        pairs = []
        missing_gen = 0
        missing_gt = 0

        for clip in clips:
            if clip not in CLIP_TIMESTAMPS:
                print(f"  警告: clip {clip} 不在时间戳表中，跳过")
                continue

            for dist in distances:
                for cam in cameras:
                    gen_path = get_gen_path(method, clip, dist, cam)
                    if not os.path.exists(gen_path):
                        missing_gen += 1
                        continue

                    gt_path = find_gt_path(clip, dist, cam)
                    if gt_path is None:
                        missing_gt += 1
                        continue

                    pairs.append({
                        "gen_path": gen_path,
                        "gt_path": gt_path,
                        "clip": clip,
                        "distance": dist,
                        "camera": cam,
                    })

        result[method] = pairs
        total = len(clips) * len(distances) * len(cameras)
        print(f"  {method}: {len(pairs)}/{total} 对 "
              f"(gen缺失:{missing_gen}, gt缺失:{missing_gt})")

    return result


# ============== 评测执行 ==============

def load_image_pair(gen_path: str, gt_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载图像对，将GT resize到gen尺寸"""
    from PIL import Image
    gen_img = Image.open(gen_path).convert('RGB')
    gt_img = Image.open(gt_path).convert('RGB')
    if gen_img.size != gt_img.size:
        gt_img = gt_img.resize(gen_img.size, Image.BILINEAR)
    return np.array(gen_img), np.array(gt_img)


def run_image_metrics(pairs: List[Dict], device: str = "cuda") -> List[Dict]:
    """运行图像质量指标 (PSNR/SSIM/LPIPS)"""
    from image_metrics_eval import ImageMetricsEvaluator
    from tqdm import tqdm

    evaluator = ImageMetricsEvaluator(device=device)
    results = []

    for pair in tqdm(pairs, desc="    图像质量"):
        gen_img, gt_img = load_image_pair(pair["gen_path"], pair["gt_path"])
        metrics = evaluator.evaluate_pair(gen_img, gt_img)
        metrics.update({
            "clip": pair["clip"],
            "distance": pair["distance"],
            "camera": pair["camera"],
        })
        results.append(metrics)

    return results


def run_depth_eval(pairs: List[Dict], device: str = "cuda") -> List[Dict]:
    """运行深度一致性评测"""
    from depth_eval import get_depth_estimator
    from metrics import compute_depth_metrics, compute_depth_correlation
    from utils import align_depth_scale
    from tqdm import tqdm

    depth_config = {
        'depth': {'model': 'depth_anything_v2', 'model_size': 'large', 'device': device}
    }
    depth_model = get_depth_estimator(depth_config)
    results = []

    for pair in tqdm(pairs, desc="    深度一致性"):
        gen_img, gt_img = load_image_pair(pair["gen_path"], pair["gt_path"])
        depth_gen = depth_model.predict(gen_img)
        depth_gt = depth_model.predict(gt_img)

        # 对齐尺度
        depth_gen_aligned = align_depth_scale(depth_gen, depth_gt, method="median")

        metrics = compute_depth_metrics(depth_gen_aligned, depth_gt)
        corr = compute_depth_correlation(depth_gen_aligned, depth_gt)
        metrics.update(corr)
        metrics.update({
            "clip": pair["clip"],
            "distance": pair["distance"],
            "camera": pair["camera"],
        })
        results.append(metrics)

    return results


def run_seg_eval(pairs: List[Dict], device: str = "cuda") -> List[Dict]:
    """运行语义分割一致性评测"""
    from seg_eval import get_segmentor
    from metrics import compute_segmentation_metrics_multilevel, compute_segmentation_consistency
    from tqdm import tqdm

    seg_config = {
        'segmentation': {
            'model': 'mask2former', 'backbone': 'swin-l',
            'dataset': 'cityscapes', 'device': device,
            'class_merging': {'enabled': True},
        }
    }
    seg_model = get_segmentor(seg_config)
    results = []

    for pair in tqdm(pairs, desc="    语义分割"):
        gen_img, gt_img = load_image_pair(pair["gen_path"], pair["gt_path"])
        seg_gen = seg_model.predict(gen_img)
        seg_gt = seg_model.predict(gt_img)

        metrics = compute_segmentation_metrics_multilevel(seg_gen, seg_gt, compute_coarse=True)
        consistency = compute_segmentation_consistency(seg_gen, seg_gt)
        metrics.update(consistency)
        metrics.update({
            "clip": pair["clip"],
            "distance": pair["distance"],
            "camera": pair["camera"],
        })
        results.append(metrics)

    return results


def run_nta_eval(pairs: List[Dict], device: str = "cuda") -> List[Dict]:
    """运行NTA-IoU评测（交通参与者检测一致性）"""
    from nta_eval import YOLO11Detector, match_detections, TRAFFIC_AGENT_IDS
    from tqdm import tqdm

    detector = YOLO11Detector(model_size="l", device=device,
                              conf_threshold=0.5, detect_size=(480, 320))
    results = []

    for pair in tqdm(pairs, desc="    NTA-IoU"):
        gen_img, gt_img = load_image_pair(pair["gen_path"], pair["gt_path"])
        dets_gen = detector.detect(gen_img, filter_classes=TRAFFIC_AGENT_IDS)
        dets_gt = detector.detect(gt_img, filter_classes=TRAFFIC_AGENT_IDS)

        metrics = match_detections(dets_gen, dets_gt, distance_threshold=50.0)
        metrics.update({
            "clip": pair["clip"],
            "distance": pair["distance"],
            "camera": pair["camera"],
        })
        results.append(metrics)

    return results


def run_ntl_eval(pairs: List[Dict], device: str = "cuda") -> List[Dict]:
    """运行NTL-IoU评测（车道线检测一致性）"""
    from ntl_eval import TwinLiteNetDetector, compute_mask_iou, compute_mask_f1
    from tqdm import tqdm

    detector = TwinLiteNetDetector(device=device, input_size=(360, 640))
    results = []

    for pair in tqdm(pairs, desc="    NTL-IoU"):
        gen_img, gt_img = load_image_pair(pair["gen_path"], pair["gt_path"])
        pred_gen = detector.predict(gen_img)
        pred_gt = detector.predict(gt_img)

        ntl_iou = compute_mask_iou(pred_gen['lane_mask'], pred_gt['lane_mask'])
        lane_f1 = compute_mask_f1(pred_gen['lane_mask'], pred_gt['lane_mask'], tolerance=3)
        da_iou = compute_mask_iou(pred_gen['da_mask'], pred_gt['da_mask'])

        metrics = {
            'ntl_iou': ntl_iou,
            'ntl_f1': lane_f1['f1'],
            'ntl_precision': lane_f1['precision'],
            'ntl_recall': lane_f1['recall'],
            'da_iou': da_iou,
            "clip": pair["clip"],
            "distance": pair["distance"],
            "camera": pair["camera"],
        }
        results.append(metrics)

    return results


def run_sam_eval(pairs: List[Dict], device: str = "cuda") -> List[Dict]:
    """运行SAM结构一致性评测（边缘F1）"""
    from sam_eval import SAMSegmentorFast, compute_edge_consistency
    from tqdm import tqdm

    sam = SAMSegmentorFast(model_size="large", device=device)
    results = []

    for pair in tqdm(pairs, desc="    SAM边缘"):
        gen_img, gt_img = load_image_pair(pair["gen_path"], pair["gt_path"])
        edge_gen = sam.get_edge_map(gen_img)
        edge_gt = sam.get_edge_map(gt_img)

        metrics = compute_edge_consistency(edge_gen, edge_gt, tolerance=3)
        metrics.update({
            "clip": pair["clip"],
            "distance": pair["distance"],
            "camera": pair["camera"],
        })
        results.append(metrics)

    return results


def run_fid_eval(pairs: List[Dict], device: str = "cuda") -> List[Dict]:
    """计算FID（需要把图像按方法收集到临时目录）

    FID是分布级指标，对整个图像集合计算一个值。
    这里返回单条记录，aggregate时直接取值即可。
    """
    from image_metrics_eval import compute_fid_for_camera
    import tempfile
    import shutil
    from PIL import Image

    # 将gen和gt图像软链接到临时目录
    tmp_dir = tempfile.mkdtemp(prefix="gs_fid_")
    gen_dir = os.path.join(tmp_dir, "gen")
    gt_dir = os.path.join(tmp_dir, "gt")
    os.makedirs(gen_dir)
    os.makedirs(gt_dir)

    try:
        for i, pair in enumerate(pairs):
            # FID需要同格式图像，统一转png
            gen_img, gt_img = load_image_pair(pair["gen_path"], pair["gt_path"])
            Image.fromarray(gen_img).save(os.path.join(gen_dir, f"{i:06d}.png"))
            Image.fromarray(gt_img).save(os.path.join(gt_dir, f"{i:06d}.png"))

        fid_score = compute_fid_for_camera(gen_dir, gt_dir, device=device)
        return [{"fid": fid_score, "clip": "all", "distance": "all", "camera": "all"}]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ============== 结果汇总 ==============

def aggregate_results(raw_results: List[Dict],
                      group_keys: List[str] = None) -> Dict:
    """按多维度汇总结果

    Args:
        raw_results: 每帧的指标列表
        group_keys: 要汇总的维度组合

    Returns:
        多层嵌套字典
    """
    from metrics import aggregate_metrics

    if not raw_results:
        return {}

    # 找出指标key（非元信息key）
    meta_keys = {"clip", "distance", "camera", "gen_path", "gt_path"}
    metric_keys = [k for k in raw_results[0].keys() if k not in meta_keys]

    def _extract_metrics(items):
        return [{k: item[k] for k in metric_keys if k in item} for item in items]

    result = {}

    # 总体汇总
    result["overall"] = aggregate_metrics(_extract_metrics(raw_results))

    # 按相机汇总
    by_camera = defaultdict(list)
    for item in raw_results:
        by_camera[item["camera"]].append(item)
    result["by_camera"] = {
        cam: aggregate_metrics(_extract_metrics(items))
        for cam, items in sorted(by_camera.items())
    }

    # 按距离汇总
    by_distance = defaultdict(list)
    for item in raw_results:
        by_distance[item["distance"]].append(item)
    result["by_distance"] = {
        dist: aggregate_metrics(_extract_metrics(items))
        for dist, items in sorted(by_distance.items())
    }

    # 按clip汇总
    by_clip = defaultdict(list)
    for item in raw_results:
        by_clip[item["clip"]].append(item)
    result["by_clip"] = {
        clip: aggregate_metrics(_extract_metrics(items))
        for clip, items in sorted(by_clip.items())
    }

    return result


def convert_to_serializable(obj):
    """将numpy类型转换为Python原生类型"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def print_summary_table(all_results: Dict[str, Dict[str, Dict]],
                        task: str):
    """打印横向对比表格（每个方法一行）"""
    methods = list(all_results.keys())
    if not methods:
        return

    # 获取指标名列表
    sample = all_results[methods[0]]
    if "overall" not in sample:
        return
    metric_keys = [k for k in sample["overall"].keys()
                   if not k.endswith('_std') and k != 'class_iou' and k != 'coarse_class_iou']

    # 打印表头
    header = f"{'Method':<20}"
    for mk in metric_keys:
        header += f" {mk:>12}"
    print("\n" + "=" * len(header))
    print(f"  {task} 横向对比")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for method in methods:
        row = f"{method:<20}"
        overall = all_results[method].get("overall", {})
        for mk in metric_keys:
            val = overall.get(mk, float('nan'))
            if isinstance(val, float):
                row += f" {val:>12.4f}"
            else:
                row += f" {str(val):>12}"
        print(row)
    print("=" * len(header))


# ============== 主函数 ==============

TASK_RUNNERS = {
    "image_metrics": run_image_metrics,
    "depth": run_depth_eval,
    "seg": run_seg_eval,
    "sam": run_sam_eval,
    "nta": run_nta_eval,
    "ntl": run_ntl_eval,
    "fid": run_fid_eval,
}

def main():
    parser = argparse.ArgumentParser(
        description="GS方法统一评测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", type=str, default="all",
                        choices=["all", "image_metrics", "depth", "seg", "sam", "nta", "ntl", "fid"],
                        help="评测任务")
    parser.add_argument("--methods", nargs="+", default=None,
                        choices=list(METHODS.keys()),
                        help="要评测的方法（默认全部）")
    parser.add_argument("--clips", nargs="+", default=None,
                        help="要评测的clip（默认全部18个）")
    parser.add_argument("--cameras", nargs="+", default=None,
                        choices=CAMERAS,
                        help="要评测的相机（默认全部7个）")
    parser.add_argument("--distances", nargs="+", default=None,
                        choices=DISTANCES,
                        help="要评测的距离（默认 near/middle/far 全部）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备")
    parser.add_argument("--output-dir", type=str, default="./results/gs_eval",
                        help="结果输出目录")
    parser.add_argument("--gt-root", type=str, default=None,
                        help="GT数据根目录（默认: /mnt/car_road_data_TianJin）")
    parser.add_argument("--max-time-diff-ms", type=float, default=100.0,
                        help="GT匹配最大时间差（毫秒）")

    args = parser.parse_args()

    # 允许覆盖GT根目录
    global GT_ROOT
    if args.gt_root is not None:
        GT_ROOT = args.gt_root

    print("=" * 70)
    print("GS方法统一评测")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 确定要跑的任务
    if args.task == "all":
        tasks = list(TASK_RUNNERS.keys())
    else:
        tasks = [args.task]

    # 收集图像对
    print("\n收集图像对...")
    all_pairs = collect_image_pairs(
        methods=args.methods,
        clips=args.clips,
        cameras=args.cameras,
        distances=args.distances,
    )

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 对每个任务运行评测
    full_results = {}

    for task in tasks:
        print(f"\n{'=' * 70}")
        print(f"运行评测: {task}")
        print(f"{'=' * 70}")

        runner = TASK_RUNNERS[task]
        task_results = {}

        for method, pairs in all_pairs.items():
            if not pairs:
                print(f"\n  {method}: 无有效图像对，跳过")
                continue

            print(f"\n  评测方法: {method} ({len(pairs)} 对)")
            try:
                raw = runner(pairs, device=args.device)
                task_results[method] = aggregate_results(raw)
            except Exception as e:
                print(f"    错误: {e}")
                import traceback
                traceback.print_exc()
                continue

        if task_results:
            # 打印对比表格
            print_summary_table(task_results, task)

            # 保存结果
            output_path = os.path.join(args.output_dir, f"{task}_results.json")
            with open(output_path, 'w') as f:
                json.dump(convert_to_serializable(task_results), f, indent=2)
            print(f"\n  结果已保存: {output_path}")

        full_results[task] = task_results

    # 保存完整结果
    full_output = os.path.join(args.output_dir, "full_results.json")
    with open(full_output, 'w') as f:
        json.dump(convert_to_serializable({
            "timestamp": datetime.now().isoformat(),
            "tasks": full_results,
        }), f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"评测完成！完整结果: {full_output}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
