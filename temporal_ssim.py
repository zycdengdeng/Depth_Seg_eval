#!/usr/bin/env python3
"""
帧间连续性评测 — Temporal SSIM（多进程并行）

对 TF/Ours 的 MP4 视频和对应 GT 帧序列计算相邻帧之间的 SSIM。

使用：
    python temporal_ssim.py
    python temporal_ssim.py --clips 031 076 --cameras FL FW
    python temporal_ssim.py --skip-gt
    python temporal_ssim.py --workers 64
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
from multiprocessing import Pool
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


# ============== 帧加载（顶层函数，可被 pickle） ==============

def _parse_vehicle_timestamp(filename):
    match = re.search(r'_(\d+\.\d+)\.jpg$', filename)
    return float(match.group(1)) if match else None


def _load_gt_entries(clip_full, camera):
    gt_dir = os.path.join(GT_ROOT, clip_full, "car", "images", camera)
    entries = []
    if os.path.isdir(gt_dir):
        for fname in os.listdir(gt_dir):
            if not fname.endswith('.jpg'):
                continue
            ts = _parse_vehicle_timestamp(fname)
            if ts is not None:
                entries.append((ts, os.path.join(gt_dir, fname)))
        entries.sort(key=lambda x: x[0])
    return entries


def _find_clip_full_name(clip_num):
    matches = glob.glob(os.path.join(GT_ROOT, f"{clip_num}_*"))
    return os.path.basename(matches[0]) if matches else None


def _compute_frame_timestamps(clip_num, num_frames):
    info = CLIP_TS_RANGES[clip_num]
    s, e = info["start"], info["end"]
    if num_frames == 1:
        return [s]
    return [int(round(s + i * (e - s) / (num_frames - 1))) for i in range(num_frames)]


def _compute_tssim(frames):
    ssim_list = []
    for i in range(len(frames) - 1):
        s = structural_similarity(frames[i], frames[i + 1],
                                  channel_axis=2, data_range=255)
        ssim_list.append(s)
    return ssim_list


PASS_THRESHOLD = 0.5


def _make_result(clip_num, cam, source, frames, ssim_list):
    mean_s = sum(ssim_list) / len(ssim_list)
    min_s = min(ssim_list)
    std_s = (sum((s - mean_s)**2 for s in ssim_list) / len(ssim_list)) ** 0.5
    pass_count = sum(1 for s in ssim_list if s >= PASS_THRESHOLD)
    pass_rate = pass_count / len(ssim_list) * 100
    return {
        "clip": clip_num, "camera": cam, "source": source,
        "num_frames": len(frames), "num_pairs": len(ssim_list),
        "temporal_ssim_mean": round(mean_s, 4),
        "temporal_ssim_min": round(min_s, 4),
        "temporal_ssim_std": round(std_s, 4),
        "pass_rate": round(pass_rate, 1),
        "per_pair_ssim": [round(s, 4) for s in ssim_list],
    }


def process_ours(args):
    """处理一个 Ours clip/camera（顶层函数）"""
    clip_num, cam, gen_root = args
    seg = CLIP_TS_RANGES[clip_num]["seg"]
    video_path = os.path.join(gen_root, f"{clip_num}_{seg}",
                              f"{CAMERA_SHORT_TO_LONG[cam]}_generated.mp4")
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) < 2:
        return None
    ssim_list = _compute_tssim(frames)
    return _make_result(clip_num, cam, "Ours", frames, ssim_list)


def process_gt(args):
    """处理一个 GT clip/camera（顶层函数）"""
    from undistort import load_gt_undistorted

    clip_num, cam = args
    clip_full = _find_clip_full_name(clip_num)
    if clip_full is None:
        return None

    entries = _load_gt_entries(clip_full, cam)
    if not entries:
        return None

    timestamps = _compute_frame_timestamps(clip_num, 29)
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
            gt_rgb = load_gt_undistorted(entries[best_idx][1], cam,
                                         target_size=(1280, 720))
            frames.append(gt_rgb)
        else:
            return None

    if len(frames) < 2:
        return None
    ssim_list = _compute_tssim(frames)
    return _make_result(clip_num, cam, "GT", frames, ssim_list)


# ============== 汇总与主函数 ==============

def summarize_results(all_results, label):
    if not all_results:
        return {}
    all_means = [r["temporal_ssim_mean"] for r in all_results]
    all_pass = [r["pass_rate"] for r in all_results]
    overall_mean = sum(all_means) / len(all_means)
    overall_min = min(r["temporal_ssim_min"] for r in all_results)
    overall_pass = sum(all_pass) / len(all_pass)

    by_camera_ssim = defaultdict(list)
    by_camera_pass = defaultdict(list)
    by_clip_ssim = defaultdict(list)
    by_clip_pass = defaultdict(list)
    for r in all_results:
        by_camera_ssim[r["camera"]].append(r["temporal_ssim_mean"])
        by_camera_pass[r["camera"]].append(r["pass_rate"])
        by_clip_ssim[r["clip"]].append(r["temporal_ssim_mean"])
        by_clip_pass[r["clip"]].append(r["pass_rate"])

    print(f"\n{'=' * 60}")
    print(f"[{label}] 汇总 (阈值={PASS_THRESHOLD})")
    print(f"{'=' * 60}")
    print(f"Overall tSSIM: {overall_mean:.4f}  帧连续性通过率: {overall_pass:.1f}%")
    print(f"\n{'Camera':<8} {'tSSIM':>8} {'通过率':>8}")
    print("-" * 28)
    for cam in CAMERAS:
        if cam in by_camera_ssim:
            m = sum(by_camera_ssim[cam]) / len(by_camera_ssim[cam])
            p = sum(by_camera_pass[cam]) / len(by_camera_pass[cam])
            print(f"{cam:<8} {m:>8.4f} {p:>7.1f}%")
    print(f"\n{'Clip':<8} {'tSSIM':>8} {'通过率':>8}")
    print("-" * 28)
    for clip in sorted(by_clip_ssim.keys()):
        m = sum(by_clip_ssim[clip]) / len(by_clip_ssim[clip])
        p = sum(by_clip_pass[clip]) / len(by_clip_pass[clip])
        print(f"{clip:<8} {m:>8.4f} {p:>7.1f}%")

    return {
        "overall_mean": round(overall_mean, 4),
        "overall_worst_pair": round(overall_min, 4),
        "overall_pass_rate": round(overall_pass, 1),
        "pass_threshold": PASS_THRESHOLD,
        "by_camera": {c: {"tssim": round(sum(v)/len(v), 4),
                          "pass_rate": round(sum(by_camera_pass[c])/len(by_camera_pass[c]), 1)}
                      for c, v in by_camera_ssim.items()},
        "by_clip": {c: {"tssim": round(sum(v)/len(v), 4),
                         "pass_rate": round(sum(by_clip_pass[c])/len(by_clip_pass[c]), 1)}
                    for c, v in by_clip_ssim.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="帧间连续性评测 (Temporal SSIM)")
    parser.add_argument("--clips", nargs="+", default=None)
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument("--gen-root", type=str, default=None)
    parser.add_argument("--skip-gt", action="store_true", help="跳过 GT")
    parser.add_argument("--workers", type=int, default=32, help="并行进程数")
    parser.add_argument("--output", type=str, default="./results/temporal_ssim.json")
    args = parser.parse_args()

    gen_root = args.gen_root or GEN_ROOT
    clips = args.clips or list(CLIP_TS_RANGES.keys())
    cameras = args.cameras or CAMERAS
    W = args.workers

    print("=" * 60)
    print(f"Temporal SSIM — 帧间连续性评测 ({W} workers)")
    print(f"Clips: {clips}")
    print(f"Cameras: {cameras}")
    print("=" * 60)

    # Ours（多进程）
    ours_tasks = [(clip, cam, gen_root) for clip in clips for cam in cameras]
    print(f"\n[Ours] {len(ours_tasks)} 个任务")
    ours_results = []
    with Pool(W) as pool:
        for r in pool.imap_unordered(process_ours, ours_tasks):
            if r:
                ours_results.append(r)
                print(f"  [{r['clip']}/{r['camera']}] tSSIM={r['temporal_ssim_mean']:.4f} 通过率={r['pass_rate']:.0f}%")

    # GT（多进程）
    gt_results = []
    if not args.skip_gt:
        gt_tasks = [(clip, cam) for clip in clips for cam in cameras]
        print(f"\n[GT] {len(gt_tasks)} 个任务")
        with Pool(W) as pool:
            for r in pool.imap_unordered(process_gt, gt_tasks):
                if r:
                    gt_results.append(r)
                    print(f"  [{r['clip']}/{r['camera']}] tSSIM={r['temporal_ssim_mean']:.4f} 通过率={r['pass_rate']:.0f}%")

    # 汇总
    ours_summary = summarize_results(ours_results, "Ours")
    gt_summary = summarize_results(gt_results, "GT") if gt_results else {}

    if ours_summary and gt_summary:
        print(f"\n{'=' * 60}")
        print(f"对比 (阈值={PASS_THRESHOLD})")
        print(f"{'=' * 60}")
        print(f"         {'tSSIM':>8}  {'通过率':>8}")
        print(f"  Ours   {ours_summary['overall_mean']:>8.4f}  {ours_summary['overall_pass_rate']:>7.1f}%")
        print(f"  GT     {gt_summary['overall_mean']:>8.4f}  {gt_summary['overall_pass_rate']:>7.1f}%")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "ours_summary": ours_summary,
            "gt_summary": gt_summary,
            "ours_raw": ours_results,
            "gt_raw": gt_results,
        }, f, indent=2)
    print(f"\n结果已保存: {args.output}")


if __name__ == "__main__":
    main()
