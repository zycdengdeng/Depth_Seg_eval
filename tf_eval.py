#!/usr/bin/env python3
"""
Transfer Zone Synchrone 评测脚本

对 transfer_zone 方法生成的视频进行评测：
1. 从 MP4 视频中抽取帧
2. 根据时间戳范围将每帧映射到路侧毫秒时间戳
3. 在 GT 车端数据中找最近邻图像
4. 运行全部评测指标

使用方法：
    python tf_eval.py --task all
    python tf_eval.py --task image_metrics --clips 031
    python tf_eval.py --task nta --cameras FL FN

数据目录：
    Gen: /mnt/zihanw/tf2.5_verion2_test_evaluation/{clip}_seg01/{camera}_generated.mp4
    GT:  /mnt/car_road_data_TianJin/{clip_full}/car/images/{camera_short}/addc_*_{ts}.jpg
"""

import os
import sys
import glob
import json
import re
import argparse
import bisect
import tempfile
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# ============== 配置 ==============

# 生成视频根目录
GEN_ROOT = "/mnt/zihanw/tf2.5_verion2_test_evaluation"

# GT 数据根目录
GT_ROOT = "/mnt/car_road_data_TianJin"

# 每个视频的帧数
FRAMES_PER_VIDEO = 29

# clip 时间戳范围（路侧毫秒）
CLIP_TS_RANGES = {
    "031": {"start": 1743572673646, "end": 1743572677516, "seg": "seg01"},
    "033": {"start": 1743573841662, "end": 1743573845466, "seg": "seg01"},
    "053": {"start": 1743583129937, "end": 1743583133770, "seg": "seg01"},
    "056": {"start": 1743584169360, "end": 1743584173177, "seg": "seg01"},
    "076": {"start": 1743646657111, "end": 1743646660984, "seg": "seg01"},
    "077": {"start": 1743647057373, "end": 1743647061207, "seg": "seg01"},
    "088": {"start": 1743652836881, "end": 1743652840713, "seg": "seg01"},
    "089": {"start": 1743653090176, "end": 1743653094003, "seg": "seg01"},
}

# 相机名映射：长名（MP4文件名）→ 短名（GT目录名）
CAMERA_LONG_TO_SHORT = {
    "cross_left_120fov":  "FL",
    "cross_right_120fov": "FR",
    "front_tele_30fov":   "FN",
    "front_wide_120fov":  "FW",
    "rear_left_70fov":    "RL",
    "rear_right_70fov":   "RR",
    "rear_tele_30fov":    "RN",
}

CAMERA_SHORT_TO_LONG = {v: k for k, v in CAMERA_LONG_TO_SHORT.items()}

# 可用相机列表（短名）
CAMERAS = list(CAMERA_LONG_TO_SHORT.values())


# ============== clip 名称解析 ==============

# 缓存: clip_num → full_clip_name
_clip_full_name_cache: Dict[str, str] = {}


def find_clip_full_name(clip_num: str) -> Optional[str]:
    """在 GT 目录中查找 clip 编号对应的完整名称

    例如: "031" → "031_car0402_road0402_t9"
    """
    if clip_num in _clip_full_name_cache:
        return _clip_full_name_cache[clip_num]

    pattern = os.path.join(GT_ROOT, f"{clip_num}_*")
    matches = glob.glob(pattern)
    if matches:
        full_name = os.path.basename(matches[0])
        _clip_full_name_cache[clip_num] = full_name
        return full_name

    print(f"  警告: 在 {GT_ROOT} 中找不到 clip {clip_num} 的目录")
    return None


# ============== GT 匹配（复用 gs_eval 逻辑）==============

def _parse_vehicle_timestamp(filename: str) -> Optional[float]:
    """从车端图像文件名提取时间戳（秒）"""
    match = re.search(r'_(\d+\.\d+)\.jpg$', filename)
    if match:
        return float(match.group(1))
    return None


_gt_file_cache: Dict[Tuple[str, str], List[Tuple[float, str]]] = {}


def _build_gt_cache(clip_full: str, camera_short: str) -> List[Tuple[float, str]]:
    """构建 GT 目录的时间戳索引"""
    key = (clip_full, camera_short)
    if key in _gt_file_cache:
        return _gt_file_cache[key]

    gt_dir = os.path.join(GT_ROOT, clip_full, "car", "images", camera_short)
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


def find_gt_path(clip_full: str, camera_short: str, road_ts_ms: int,
                 max_diff_ms: float = 100.0) -> Optional[str]:
    """找到与指定路侧时间戳最近的 GT 车端图像"""
    target_ts_sec = road_ts_ms / 1000.0

    entries = _build_gt_cache(clip_full, camera_short)
    if not entries:
        return None

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


# ============== MP4 帧提取 ==============

def extract_frames(video_path: str) -> List[np.ndarray]:
    """从 MP4 中提取所有帧为 RGB numpy 数组"""
    import cv2

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def compute_frame_timestamps(clip_num: str, num_frames: int) -> List[int]:
    """计算每帧对应的路侧毫秒时间戳（线性插值）"""
    info = CLIP_TS_RANGES[clip_num]
    ts_start = info["start"]
    ts_end = info["end"]

    if num_frames == 1:
        return [ts_start]

    timestamps = []
    for i in range(num_frames):
        ts = ts_start + i * (ts_end - ts_start) / (num_frames - 1)
        timestamps.append(int(round(ts)))
    return timestamps


# ============== 图像对收集 ==============

def collect_image_pairs(clips: List[str] = None,
                        cameras: List[str] = None,
                        ) -> List[Dict]:
    """收集所有 (gen_frame, gt_path) 图像对

    Returns:
        [{"gen_frame": np.ndarray, "gt_path": str, "clip": str,
          "camera": str, "frame_idx": int, "timestamp": int}, ...]
    """
    if clips is None:
        clips = list(CLIP_TS_RANGES.keys())
    if cameras is None:
        cameras = CAMERAS

    pairs = []
    missing_gt = 0
    missing_video = 0
    total = 0

    for clip_num in clips:
        if clip_num not in CLIP_TS_RANGES:
            print(f"  警告: clip {clip_num} 不在时间戳表中，跳过")
            continue

        clip_full = find_clip_full_name(clip_num)
        if clip_full is None:
            continue

        seg = CLIP_TS_RANGES[clip_num]["seg"]
        clip_dir = f"{clip_num}_{seg}"

        for cam_short in cameras:
            cam_long = CAMERA_SHORT_TO_LONG.get(cam_short)
            if cam_long is None:
                continue

            video_path = os.path.join(GEN_ROOT, clip_dir,
                                      f"{cam_long}_generated.mp4")
            total += FRAMES_PER_VIDEO

            if not os.path.exists(video_path):
                missing_video += FRAMES_PER_VIDEO
                continue

            # 提取帧
            frames = extract_frames(video_path)
            if not frames:
                missing_video += FRAMES_PER_VIDEO
                continue

            # 计算时间戳
            frame_timestamps = compute_frame_timestamps(clip_num, len(frames))

            for i, (frame, ts) in enumerate(zip(frames, frame_timestamps)):
                gt_path = find_gt_path(clip_full, cam_short, ts)
                if gt_path is None:
                    missing_gt += 1
                    continue

                pairs.append({
                    "gen_frame": frame,
                    "gt_path": gt_path,
                    "clip": clip_num,
                    "camera": cam_short,
                    "frame_idx": i,
                    "timestamp": ts,
                })

    print(f"  收集完成: {len(pairs)}/{total} 对 "
          f"(视频缺失:{missing_video}, GT缺失:{missing_gt})")
    return pairs


# ============== 图像加载 ==============

def load_pair(pair: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """加载 gen 帧和 GT 图像，对GT去畸变后resize到gen尺寸"""
    from undistort import load_gt_undistorted

    gen_img = pair["gen_frame"]
    gen_h, gen_w = gen_img.shape[:2]

    gt_arr = load_gt_undistorted(
        pair["gt_path"], pair["camera"],
        target_size=(gen_w, gen_h)
    )

    return gen_img, gt_arr


# ============== 评测任务 ==============

def run_image_metrics(pairs: List[Dict], device: str = "cuda") -> List[Dict]:
    """图像质量 (PSNR/SSIM/LPIPS)"""
    from image_metrics_eval import ImageMetricsEvaluator
    from tqdm import tqdm

    evaluator = ImageMetricsEvaluator(device=device)
    results = []
    for pair in tqdm(pairs, desc="    图像质量"):
        gen_img, gt_img = load_pair(pair)
        metrics = evaluator.evaluate_pair(gen_img, gt_img)
        metrics.update({"clip": pair["clip"], "camera": pair["camera"],
                        "frame_idx": pair["frame_idx"]})
        results.append(metrics)
    return results


def run_depth_eval(pairs: List[Dict], device: str = "cuda") -> List[Dict]:
    """深度一致性"""
    from depth_eval import get_depth_estimator
    from metrics import compute_depth_metrics, compute_depth_correlation
    from utils import align_depth_scale
    from tqdm import tqdm

    depth_config = {'depth': {'model': 'depth_anything_v2', 'model_size': 'large', 'device': device}}
    model = get_depth_estimator(depth_config)
    results = []
    for pair in tqdm(pairs, desc="    深度一致性"):
        gen_img, gt_img = load_pair(pair)
        d_gen = model.predict(gen_img)
        d_gt = model.predict(gt_img)
        d_gen = align_depth_scale(d_gen, d_gt, method="median")
        metrics = compute_depth_metrics(d_gen, d_gt)
        metrics.update(compute_depth_correlation(d_gen, d_gt))
        metrics.update({"clip": pair["clip"], "camera": pair["camera"],
                        "frame_idx": pair["frame_idx"]})
        results.append(metrics)
    return results


def run_seg_eval(pairs: List[Dict], device: str = "cuda") -> List[Dict]:
    """语义分割一致性"""
    from seg_eval import get_segmentor
    from metrics import compute_segmentation_metrics_multilevel, compute_segmentation_consistency
    from tqdm import tqdm

    seg_config = {'segmentation': {
        'model': 'mask2former', 'backbone': 'swin-l',
        'dataset': 'cityscapes', 'device': device,
        'class_merging': {'enabled': True},
    }}
    model = get_segmentor(seg_config)
    results = []
    for pair in tqdm(pairs, desc="    语义分割"):
        gen_img, gt_img = load_pair(pair)
        s_gen = model.predict(gen_img)
        s_gt = model.predict(gt_img)
        metrics = compute_segmentation_metrics_multilevel(s_gen, s_gt, compute_coarse=True)
        metrics.update(compute_segmentation_consistency(s_gen, s_gt))
        metrics.update({"clip": pair["clip"], "camera": pair["camera"],
                        "frame_idx": pair["frame_idx"]})
        results.append(metrics)
    return results


def run_sam_eval(pairs: List[Dict], device: str = "cuda") -> List[Dict]:
    """SAM 边缘一致性"""
    from sam_eval import SAMSegmentorFast, compute_edge_consistency
    from tqdm import tqdm

    sam = SAMSegmentorFast(model_size="large", device=device)
    results = []
    for pair in tqdm(pairs, desc="    SAM边缘"):
        gen_img, gt_img = load_pair(pair)
        edge_gen = sam.get_edge_map(gen_img)
        edge_gt = sam.get_edge_map(gt_img)
        h, w = edge_gen.shape[:2]
        tol = max(3, int(round((h**2 + w**2)**0.5 * 0.005)))
        metrics = compute_edge_consistency(edge_gen, edge_gt, tolerance=tol)
        metrics.update({"clip": pair["clip"], "camera": pair["camera"],
                        "frame_idx": pair["frame_idx"]})
        results.append(metrics)
    return results


def run_nta_eval(pairs: List[Dict], device: str = "cuda") -> List[Dict]:
    """NTA-IoU（交通参与者检测一致性）"""
    from nta_eval import YOLO11Detector, match_detections, TRAFFIC_AGENT_IDS
    from tqdm import tqdm

    detector = YOLO11Detector(model_size="l", device=device,
                              conf_threshold=0.5, detect_size=(480, 320))
    results = []
    for pair in tqdm(pairs, desc="    NTA-IoU"):
        gen_img, gt_img = load_pair(pair)
        dets_gen = detector.detect(gen_img, filter_classes=TRAFFIC_AGENT_IDS)
        dets_gt = detector.detect(gt_img, filter_classes=TRAFFIC_AGENT_IDS)
        metrics = match_detections(dets_gen, dets_gt, distance_threshold=50.0)
        metrics.update({"clip": pair["clip"], "camera": pair["camera"],
                        "frame_idx": pair["frame_idx"]})
        results.append(metrics)
    return results


def run_ntl_eval(pairs: List[Dict], device: str = "cuda") -> List[Dict]:
    """NTL-IoU（车道线检测一致性）"""
    from ntl_eval import TwinLiteNetDetector, compute_mask_iou, compute_mask_f1
    from tqdm import tqdm

    detector = TwinLiteNetDetector(device=device, input_size=(360, 640))
    results = []
    for pair in tqdm(pairs, desc="    NTL-IoU"):
        gen_img, gt_img = load_pair(pair)
        pred_gen = detector.predict(gen_img)
        pred_gt = detector.predict(gt_img)
        ntl_iou = compute_mask_iou(pred_gen['lane_mask'], pred_gt['lane_mask'])
        lane_f1 = compute_mask_f1(pred_gen['lane_mask'], pred_gt['lane_mask'], tolerance=3)
        da_iou = compute_mask_iou(pred_gen['da_mask'], pred_gt['da_mask'])
        metrics = {
            'ntl_iou': ntl_iou, 'ntl_f1': lane_f1['f1'],
            'ntl_precision': lane_f1['precision'], 'ntl_recall': lane_f1['recall'],
            'da_iou': da_iou,
            "clip": pair["clip"], "camera": pair["camera"],
            "frame_idx": pair["frame_idx"],
        }
        results.append(metrics)
    return results


def run_fid_eval(pairs: List[Dict], device: str = "cuda") -> List[Dict]:
    """FID（分布级指标）"""
    from image_metrics_eval import compute_fid_for_camera
    from PIL import Image
    import shutil

    tmp_dir = tempfile.mkdtemp(prefix="tf_fid_")
    gen_dir = os.path.join(tmp_dir, "gen")
    gt_dir = os.path.join(tmp_dir, "gt")
    os.makedirs(gen_dir)
    os.makedirs(gt_dir)

    try:
        for i, pair in enumerate(pairs):
            gen_img, gt_img = load_pair(pair)
            Image.fromarray(gen_img).save(os.path.join(gen_dir, f"{i:06d}.png"))
            Image.fromarray(gt_img).save(os.path.join(gt_dir, f"{i:06d}.png"))
        fid_score = compute_fid_for_camera(gen_dir, gt_dir, device=device)
        return [{"fid": fid_score, "clip": "all", "camera": "all", "frame_idx": -1}]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ============== 结果汇总 ==============

def aggregate_results(raw_results: List[Dict]) -> Dict:
    """按多维度汇总"""
    from metrics import aggregate_metrics

    if not raw_results:
        return {}

    meta_keys = {"clip", "camera", "frame_idx", "timestamp", "gen_frame", "gt_path"}
    metric_keys = [k for k in raw_results[0].keys() if k not in meta_keys]

    def _extract(items):
        return [{k: item[k] for k in metric_keys if k in item} for item in items]

    result = {"overall": aggregate_metrics(_extract(raw_results))}

    # 按相机
    by_cam = defaultdict(list)
    for item in raw_results:
        by_cam[item["camera"]].append(item)
    result["by_camera"] = {
        cam: aggregate_metrics(_extract(items))
        for cam, items in sorted(by_cam.items())
    }

    # 按 clip
    by_clip = defaultdict(list)
    for item in raw_results:
        by_clip[item["clip"]].append(item)
    result["by_clip"] = {
        clip: aggregate_metrics(_extract(items))
        for clip, items in sorted(by_clip.items())
    }

    return result


def convert_to_serializable(obj):
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


def print_summary(results: Dict, task: str):
    """打印汇总表格"""
    overall = results.get("overall", {})
    metric_keys = [k for k in overall.keys()
                   if not k.endswith('_std') and k not in ('class_iou', 'coarse_class_iou')]
    if not metric_keys:
        return

    header = f"{'维度':<20}"
    for mk in metric_keys:
        header += f" {mk:>12}"
    print("\n" + "=" * len(header))
    print(f"  {task} 结果")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # Overall
    row = f"{'overall':<20}"
    for mk in metric_keys:
        val = overall.get(mk, float('nan'))
        row += f" {val:>12.4f}" if isinstance(val, float) else f" {str(val):>12}"
    print(row)

    # 按相机
    for cam, cam_res in sorted(results.get("by_camera", {}).items()):
        row = f"{cam:<20}"
        for mk in metric_keys:
            val = cam_res.get(mk, float('nan'))
            row += f" {val:>12.4f}" if isinstance(val, float) else f" {str(val):>12}"
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
        description="Transfer Zone Synchrone 评测",
    )
    parser.add_argument("--task", type=str, default="all",
                        choices=["all", "image_metrics", "depth", "seg", "sam",
                                 "nta", "ntl", "fid"],
                        help="评测任务")
    parser.add_argument("--clips", nargs="+", default=None,
                        choices=list(CLIP_TS_RANGES.keys()),
                        help="要评测的 clip（默认全部 8 个）")
    parser.add_argument("--cameras", nargs="+", default=None,
                        choices=CAMERAS,
                        help="要评测的相机（默认全部 7 个）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备")
    parser.add_argument("--output-dir", type=str,
                        default="./results/tf_eval",
                        help="结果输出目录")
    parser.add_argument("--gen-root", type=str, default=None,
                        help="生成视频根目录")
    parser.add_argument("--gt-root", type=str, default=None,
                        help="GT 数据根目录")

    args = parser.parse_args()

    global GEN_ROOT, GT_ROOT
    if args.gen_root is not None:
        GEN_ROOT = args.gen_root
    if args.gt_root is not None:
        GT_ROOT = args.gt_root

    print("=" * 70)
    print("Transfer Zone Synchrone 评测")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据: {GEN_ROOT}")
    print("=" * 70)

    tasks = list(TASK_RUNNERS.keys()) if args.task == "all" else [args.task]

    # 收集图像对（提取帧 + 匹配 GT）
    print("\n收集图像对（提取MP4帧 + 匹配GT）...")
    pairs = collect_image_pairs(clips=args.clips, cameras=args.cameras)

    if not pairs:
        print("没有找到有效的图像对！")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    full_results = {}

    for task in tasks:
        print(f"\n{'=' * 70}")
        print(f"运行评测: {task}")
        print(f"{'=' * 70}")

        runner = TASK_RUNNERS[task]
        try:
            raw = runner(pairs, device=args.device)
            result = aggregate_results(raw)
            print_summary(result, task)

            output_path = os.path.join(args.output_dir, f"{task}_results.json")
            with open(output_path, 'w') as f:
                json.dump(convert_to_serializable(result), f, indent=2)
            print(f"  结果已保存: {output_path}")

            full_results[task] = result
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()

    # 保存完整结果
    full_output = os.path.join(args.output_dir, "full_results.json")
    with open(full_output, 'w') as f:
        json.dump(convert_to_serializable({
            "timestamp": datetime.now().isoformat(),
            "source": GEN_ROOT,
            "tasks": full_results,
        }), f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"评测完成！完整结果: {full_output}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
