#!/usr/bin/env python3
"""
公平 FID 评测 — 统一 clips 和样本量

只使用 GS 和 Ours 重叠的 4 个 clips (031/056/076/088)，
Ours 取首帧/中间帧/末帧对应 GS 的 near/middle/far，
保证两边样本量完全一致 (4 clips × 3 × 7 cameras = 84 张)。

使用：
    python fid_fair.py --device cuda:0
"""

import os
import sys
import shutil
import tempfile
import argparse
import cv2
import numpy as np
from PIL import Image

# 重叠的 clips
SHARED_CLIPS = ["031", "056", "076", "088"]

# GS 路径配置（复用 gs_eval 的定义）
from gs_eval import METHODS, CLIP_TIMESTAMPS, CAMERAS, DISTANCES, get_gen_path, find_gt_path
from undistort import load_gt_undistorted

# TF 路径配置
TF_GEN_ROOT = "/mnt/zihanw/tf2.5_verion2_test_evaluation"
TF_CLIP_SEGS = {"031": "seg01", "056": "seg01", "076": "seg01", "088": "seg01"}
CAMERA_SHORT_TO_LONG = {
    "FL": "cross_left_120fov", "FR": "cross_right_120fov",
    "FN": "front_tele_30fov", "FW": "front_wide_120fov",
    "RL": "rear_left_70fov", "RR": "rear_right_70fov",
    "RN": "rear_tele_30fov",
}


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def collect_gs_images(method_name):
    """收集 GS 方法在 4 个共享 clips 上的图像"""
    images = []
    for clip_num in SHARED_CLIPS:
        clip_full = [c for c in CLIP_TIMESTAMPS if c.startswith(f"{clip_num}_")][0]
        for dist in DISTANCES:
            for cam in CAMERAS:
                path = get_gen_path(method_name, clip_full, dist, cam)
                if os.path.exists(path):
                    img = np.array(Image.open(path).convert('RGB'))
                    images.append(img)
    return images


def collect_gs_gt_images():
    """收集 GS 对应的 GT 图像（去畸变）"""
    images = []
    for clip_num in SHARED_CLIPS:
        clip_full = [c for c in CLIP_TIMESTAMPS if c.startswith(f"{clip_num}_")][0]
        for dist in DISTANCES:
            for cam in CAMERAS:
                gt_path = find_gt_path(clip_full, dist, cam)
                if gt_path:
                    img = load_gt_undistorted(gt_path, cam, target_size=(1280, 720))
                    images.append(img)
    return images


def collect_tf_images():
    """收集 Ours 在 4 个共享 clips 上的图像（首/中/末帧）"""
    images = []
    for clip_num in SHARED_CLIPS:
        seg = TF_CLIP_SEGS[clip_num]
        clip_dir = f"{clip_num}_{seg}"
        for cam in CAMERAS:
            cam_long = CAMERA_SHORT_TO_LONG[cam]
            video_path = os.path.join(TF_GEN_ROOT, clip_dir, f"{cam_long}_generated.mp4")
            if not os.path.exists(video_path):
                continue
            frames = extract_frames(video_path)
            if len(frames) < 3:
                continue
            mid = len(frames) // 2
            for idx in [0, mid, len(frames) - 1]:
                images.append(frames[idx])
    return images


def save_images_to_dir(images, directory):
    os.makedirs(directory, exist_ok=True)
    for i, img in enumerate(images):
        Image.fromarray(img).save(os.path.join(directory, f"{i:06d}.png"))


def compute_fid(dir1, dir2, device):
    from image_metrics_eval import compute_fid_for_camera
    return compute_fid_for_camera(dir1, dir2, device=device)


def main():
    parser = argparse.ArgumentParser(description="公平 FID 评测")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("=" * 60)
    print("公平 FID 评测")
    print(f"共享 clips: {SHARED_CLIPS}")
    print(f"每方法样本量: 4 clips × 3 distances × 7 cameras = 84")
    print("=" * 60)

    # 收集 GT（只需要一份）
    print("\n收集 GT 图像...")
    gt_images = collect_gs_gt_images()
    print(f"  GT: {len(gt_images)} 张")

    tmp_dir = tempfile.mkdtemp(prefix="fid_fair_")
    gt_dir = os.path.join(tmp_dir, "gt")
    save_images_to_dir(gt_images, gt_dir)

    results = {}

    try:
        # Ours
        print("\n收集 Ours 图像 (首/中/末帧)...")
        tf_images = collect_tf_images()
        print(f"  Ours: {len(tf_images)} 张")
        tf_dir = os.path.join(tmp_dir, "tf_ours")
        save_images_to_dir(tf_images, tf_dir)
        fid_score = compute_fid(tf_dir, gt_dir, args.device)
        results["TF_Ours"] = fid_score
        print(f"  FID = {fid_score:.1f}")

        # GS 方法
        for method_name in METHODS:
            print(f"\n收集 {method_name} 图像...")
            gs_images = collect_gs_images(method_name)
            print(f"  {method_name}: {len(gs_images)} 张")
            if not gs_images:
                continue
            gs_dir = os.path.join(tmp_dir, method_name)
            save_images_to_dir(gs_images, gs_dir)
            fid_score = compute_fid(gs_dir, gt_dir, args.device)
            results[method_name] = fid_score
            print(f"  FID = {fid_score:.1f}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # 汇总
    print(f"\n{'=' * 60}")
    print("FID 对比 (公平模式, 4 shared clips)")
    print(f"{'=' * 60}")
    print(f"{'Method':<20} {'FID':>8} {'N_images':>10}")
    print("-" * 40)
    for method, fid in sorted(results.items(), key=lambda x: x[1]):
        print(f"{method:<20} {fid:>8.1f}")
    print(f"\nGT 样本数: {len(gt_images)}")


if __name__ == "__main__":
    main()
