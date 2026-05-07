#!/usr/bin/env python3
"""
帧间连续性评测 — Temporal SSIM

对 TF/Ours 的 MP4 视频和对应 GT 帧序列计算相邻帧之间的 SSIM。

使用：
    python temporal_ssim.py
    python temporal_ssim.py --clips 031 076 --cameras FL FW
    python temporal_ssim.py --skip-gt   # 只跑 Ours，跳过 GT
"""

import os
import re
import glob
import json
import bisect
import argparse
import cv2
import numpy as np
from datetime import datetime
from skimage.metrics import structural_similarity
from collections import defaultdict
from typing import List, Optional, Tuple

GEN_ROOT = "/mnt/zihanw/tf2.5_verion2_test_evaluation"
GT_ROOT = "/mnt/car_road_data_TianJin"

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

CAMERA_SHORT_TO_LONG = {
    "FL": "cross_left_120fov",
    "FR": "cross_right_120fov",
    "FN": "front_tele_30fov",
    "FW": "front_wide_120fov",
    "RL": "rear_left_70fov",
    "RR": "rear_right_70fov",
    "RN": "rear_tele_30fov",
}

CAMERAS = list(CAMERA_SHORT_TO_LONG.keys())


# ============== GT 帧序列 ==============

_clip_full_name_cache = {}
_gt_file_cache = {}


def find_clip_full_name(clip_num: str) -> Optional[str]:
    if clip_num in _clip_full_name_cache:
        return _clip_full_name_cache[clip_num]
    matches = glob.glob(os.path.join(GT_ROOT, f"{clip_num}_*"))
    if matches:
        full_name = os.path.basename(matches[0])
        _clip_full_name_cache[clip_num] = full_name
        return full_name
    return None


def _build_gt_cache(clip_full: str, camera: str) -> List[Tuple[float, str]]:
    key = (clip_full, camera)
    if key in _gt_file_cache:
        return _gt_file_cache[key]
    gt_dir = os.path.join(GT_ROOT, clip_full, "car", "images", camera)
    entries = []
    if os.path.isdir(gt_dir):
        for fname in os.listdir(gt_dir):
            if not fname.endswith('.jpg'):
                continue
            match = re.search(r'_(\d+\.\d+)\.jpg$', fname)
            if match:
                ts = float(match.group(1))
                entries.append((ts, os.path.join(gt_dir, fname)))
        entries.sort(key=lambda x: x[0])
    _gt_file_cache[key] = entries
    return entries


def compute_frame_timestamps(clip_num: str, num_frames: int) -> List[int]:
    info = CLIP_TS_RANGES[clip_num]
    ts_start, ts_end = info["start"], info["end"]
    if num_frames == 1:
        return [ts_start]
    return [int(round(ts_start + i * (ts_end - ts_start) / (num_frames - 1)))
            for i in range(num_frames)]


def collect_gt_frame_sequence(clip_num: str, camera: str,
                              num_frames: int = 29) -> List[np.ndarray]:
    """收集与视频帧对应的 GT 帧序列（去畸变 + resize 到 1280x720）"""
    from undistort import load_gt_undistorted

    clip_full = find_clip_full_name(clip_num)
    if clip_full is None:
        return []

    timestamps = compute_frame_timestamps(clip_num, num_frames)
    entries = _build_gt_cache(clip_full, camera)
    if not entries:
        return []

    ts_list = [e[0] for e in entries]
    frames = []
    for road_ts_ms in timestamps:
        target = road_ts_ms / 1000.0
        idx = bisect.bisect_left(ts_list, target)
        best_idx, best_diff = None, float('inf')
        for c in [idx - 1, idx]:
            if 0 <= c < len(entries):
                diff = abs(entries[c][0] - target)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = c
        if best_idx is not None and best_diff * 1000 <= 100.0:
            gt_rgb = load_gt_undistorted(entries[best_idx][1], camera,
                                         target_size=(1280, 720))
            frames.append(gt_rgb)
        else:
            return []  # 有帧缺失则放弃整个序列
    return frames


# ============== 通用计算 ==============

def extract_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def compute_temporal_ssim(frames):
    ssim_list = []
    for i in range(len(frames) - 1):
        s = structural_similarity(frames[i], frames[i + 1],
                                  channel_axis=2, data_range=255)
        ssim_list.append(s)
    return ssim_list


def summarize_results(all_results, label):
    if not all_results:
        return {}

    all_means = [r["temporal_ssim_mean"] for r in all_results]
    overall_mean = sum(all_means) / len(all_means)
    overall_min = min(r["temporal_ssim_min"] for r in all_results)

    by_camera = defaultdict(list)
    by_clip = defaultdict(list)
    for r in all_results:
        by_camera[r["camera"]].append(r["temporal_ssim_mean"])
        by_clip[r["clip"]].append(r["temporal_ssim_mean"])

    print(f"\n{'=' * 60}")
    print(f"[{label}] 汇总")
    print(f"{'=' * 60}")
    print(f"Overall tSSIM: {overall_mean:.4f} (worst pair: {overall_min:.4f})")

    print(f"\n{'Camera':<8} {'tSSIM':>8}")
    print("-" * 18)
    for cam in CAMERAS:
        if cam in by_camera:
            m = sum(by_camera[cam]) / len(by_camera[cam])
            print(f"{cam:<8} {m:>8.4f}")

    print(f"\n{'Clip':<8} {'tSSIM':>8}")
    print("-" * 18)
    for clip in sorted(by_clip.keys()):
        m = sum(by_clip[clip]) / len(by_clip[clip])
        print(f"{clip:<8} {m:>8.4f}")

    return {
        "overall_mean": round(overall_mean, 4),
        "overall_worst_pair": round(overall_min, 4),
        "by_camera": {cam: round(sum(v)/len(v), 4)
                      for cam, v in by_camera.items()},
        "by_clip": {clip: round(sum(v)/len(v), 4)
                    for clip, v in by_clip.items()},
    }


def run_source(source_label, clips, cameras, frame_loader):
    """对一个数据源跑 tSSIM"""
    all_results = []
    print(f"\n[{source_label}]")
    for clip_num in clips:
        for cam in cameras:
            frames = frame_loader(clip_num, cam)
            if not frames or len(frames) < 2:
                continue
            ssim_list = compute_temporal_ssim(frames)
            mean_ssim = sum(ssim_list) / len(ssim_list)
            min_ssim = min(ssim_list)
            std_ssim = (sum((s - mean_ssim)**2 for s in ssim_list) / len(ssim_list)) ** 0.5

            result = {
                "clip": clip_num,
                "camera": cam,
                "source": source_label,
                "num_frames": len(frames),
                "num_pairs": len(ssim_list),
                "temporal_ssim_mean": round(mean_ssim, 4),
                "temporal_ssim_min": round(min_ssim, 4),
                "temporal_ssim_std": round(std_ssim, 4),
                "per_pair_ssim": [round(s, 4) for s in ssim_list],
            }
            all_results.append(result)
            print(f"  [{clip_num}/{cam}] {len(frames)} frames  "
                  f"tSSIM={mean_ssim:.4f} (min={min_ssim:.4f})")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="帧间连续性评测 (Temporal SSIM)")
    parser.add_argument("--clips", nargs="+", default=None)
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument("--gen-root", type=str, default=None)
    parser.add_argument("--skip-gt", action="store_true", help="跳过 GT")
    parser.add_argument("--output", type=str, default="./results/temporal_ssim.json")
    args = parser.parse_args()

    gen_root = args.gen_root or GEN_ROOT
    clips = args.clips or list(CLIP_TS_RANGES.keys())
    cameras = args.cameras or CAMERAS

    print("=" * 60)
    print("Temporal SSIM — 帧间连续性评测")
    print(f"Clips: {clips}")
    print(f"Cameras: {cameras}")
    print("=" * 60)

    # Ours
    def load_ours(clip_num, cam):
        seg = CLIP_TS_RANGES[clip_num]["seg"]
        video_path = os.path.join(gen_root, f"{clip_num}_{seg}",
                                  f"{CAMERA_SHORT_TO_LONG[cam]}_generated.mp4")
        if not os.path.exists(video_path):
            return []
        return extract_frames(video_path)

    ours_results = run_source("Ours", clips, cameras, load_ours)

    # GT
    gt_results = []
    if not args.skip_gt:
        def load_gt(clip_num, cam):
            return collect_gt_frame_sequence(clip_num, cam, num_frames=29)
        gt_results = run_source("GT", clips, cameras, load_gt)

    # 汇总
    ours_summary = summarize_results(ours_results, "Ours")
    gt_summary = summarize_results(gt_results, "GT") if gt_results else {}

    if ours_summary and gt_summary:
        print(f"\n{'=' * 60}")
        print("对比")
        print(f"{'=' * 60}")
        print(f"  Ours tSSIM: {ours_summary['overall_mean']:.4f}")
        print(f"  GT   tSSIM: {gt_summary['overall_mean']:.4f}")
        diff = ours_summary['overall_mean'] - gt_summary['overall_mean']
        print(f"  差值:        {diff:+.4f} ({'Ours 更平滑' if diff > 0 else 'GT 更平滑'})")

    # 保存
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "ours_summary": ours_summary,
        "gt_summary": gt_summary,
        "ours_raw": ours_results,
        "gt_raw": gt_results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存: {args.output}")


if __name__ == "__main__":
    main()
