#!/usr/bin/env python3
"""
帧间连续性评测 — Temporal SSIM

对 TF/Ours 的 MP4 视频计算相邻帧之间的 SSIM，衡量帧间连续性。

使用：
    python temporal_ssim.py --device cuda:0
    python temporal_ssim.py --clips 031 076 --cameras FL FW
"""

import os
import json
import argparse
import cv2
import numpy as np
from datetime import datetime
from skimage.metrics import structural_similarity
from collections import defaultdict

GEN_ROOT = "/mnt/zihanw/tf2.5_verion2_test_evaluation"

CLIP_TS_RANGES = {
    "031": {"seg": "seg01"},
    "033": {"seg": "seg01"},
    "053": {"seg": "seg01"},
    "056": {"seg": "seg01"},
    "076": {"seg": "seg01"},
    "077": {"seg": "seg01"},
    "088": {"seg": "seg01"},
    "089": {"seg": "seg01"},
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
    """计算相邻帧之间的 SSIM 序列"""
    ssim_list = []
    for i in range(len(frames) - 1):
        s = structural_similarity(frames[i], frames[i + 1],
                                  channel_axis=2, data_range=255)
        ssim_list.append(s)
    return ssim_list


def main():
    parser = argparse.ArgumentParser(description="帧间连续性评测 (Temporal SSIM)")
    parser.add_argument("--clips", nargs="+", default=None,
                        help="clip 编号（默认全部 8 个）")
    parser.add_argument("--cameras", nargs="+", default=None,
                        help="相机（默认全部 7 个）")
    parser.add_argument("--gen-root", type=str, default=None)
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

    all_results = []

    for clip_num in clips:
        seg = CLIP_TS_RANGES[clip_num]["seg"]
        clip_dir = f"{clip_num}_{seg}"

        for cam in cameras:
            cam_long = CAMERA_SHORT_TO_LONG.get(cam)
            if not cam_long:
                continue
            video_path = os.path.join(gen_root, clip_dir, f"{cam_long}_generated.mp4")
            if not os.path.exists(video_path):
                print(f"  [{clip_num}/{cam}] 视频不存在，跳过")
                continue

            frames = extract_frames(video_path)
            if len(frames) < 2:
                continue

            ssim_list = compute_temporal_ssim(frames)
            mean_ssim = sum(ssim_list) / len(ssim_list)
            min_ssim = min(ssim_list)
            std_ssim = (sum((s - mean_ssim) ** 2 for s in ssim_list) / len(ssim_list)) ** 0.5

            result = {
                "clip": clip_num,
                "camera": cam,
                "num_frames": len(frames),
                "num_pairs": len(ssim_list),
                "temporal_ssim_mean": round(mean_ssim, 4),
                "temporal_ssim_min": round(min_ssim, 4),
                "temporal_ssim_std": round(std_ssim, 4),
                "per_pair_ssim": [round(s, 4) for s in ssim_list],
            }
            all_results.append(result)
            print(f"  [{clip_num}/{cam}] {len(frames)} frames  "
                  f"tSSIM={mean_ssim:.4f} (min={min_ssim:.4f}, std={std_ssim:.4f})")

    if not all_results:
        print("没有结果！")
        return

    # 汇总
    all_means = [r["temporal_ssim_mean"] for r in all_results]
    all_mins = [r["temporal_ssim_min"] for r in all_results]
    overall_mean = sum(all_means) / len(all_means)
    overall_min = min(all_mins)

    # 按相机汇总
    by_camera = defaultdict(list)
    for r in all_results:
        by_camera[r["camera"]].append(r["temporal_ssim_mean"])

    # 按clip汇总
    by_clip = defaultdict(list)
    for r in all_results:
        by_clip[r["clip"]].append(r["temporal_ssim_mean"])

    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    print(f"Overall tSSIM: {overall_mean:.4f} (worst single pair: {overall_min:.4f})")

    print(f"\n{'Camera':<8} {'tSSIM':>8}")
    print("-" * 18)
    for cam in CAMERAS:
        if cam in by_camera:
            m = sum(by_camera[cam]) / len(by_camera[cam])
            print(f"{cam:<8} {m:>8.4f}")

    print(f"\n{'Clip':<8} {'tSSIM':>8}")
    print("-" * 18)
    for clip in clips:
        if clip in by_clip:
            m = sum(by_clip[clip]) / len(by_clip[clip])
            print(f"{clip:<8} {m:>8.4f}")

    summary = {
        "overall_mean": round(overall_mean, 4),
        "overall_worst_pair": round(overall_min, 4),
        "by_camera": {cam: round(sum(v)/len(v), 4)
                      for cam, v in by_camera.items()},
        "by_clip": {clip: round(sum(v)/len(v), 4)
                    for clip, v in by_clip.items()},
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "raw": all_results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存: {args.output}")


if __name__ == "__main__":
    main()
