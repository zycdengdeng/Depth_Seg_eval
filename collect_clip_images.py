#!/usr/bin/env python3
"""
收集指定 clip 的所有 GS 方法渲染结果，复制到统一目录方便查看对比。

使用：
    python collect_clip_images.py --clip 053 --output ./visualizations/clip053

输出结构：
    {output}/{method}/{distance}/{camera}.png
"""

import os
import shutil
import glob
import argparse


GS_METHODS = {
    "DDGS":          "/mnt/myn/project/DDGS/output",
    "DropGaussian":  "/mnt/myn/project/DropGaussian_release/output",
    "Co-Adaptation": "/mnt/myn/project/Co-Adaptation-of-3DGS/output",
    "AD-GS":         "/mnt/myn/project/AD-GS/output",
    "SparseGS":      "/mnt/zyc_wzh/SparseGS/output/car_road",
    "S3Gaussian":    "/mnt/zyc_wzh/S3Gaussian/work_dirs/roadside_colmap/renders",
}

CAMERAS = ["FL", "FN", "FR", "FW", "RL", "RN", "RR"]
DISTANCES = ["near", "middle", "far"]


def collect_timestamp_method(method, root, clip_num, output_dir):
    """方法类型1: {root}/{clip_full}/{timestamp}/vehicle_renders/{cam}/render.png"""
    # 找 clip 完整目录名（可能是 053_carXXX_roadXXX_tN）
    clip_dirs = glob.glob(os.path.join(root, f"{clip_num}_*"))
    if not clip_dirs:
        print(f"  [{method}] 未找到 clip {clip_num} 目录")
        return 0

    clip_dir = clip_dirs[0]
    # timestamp 子目录按字典序 = 时间先后
    ts_dirs = sorted([d for d in os.listdir(clip_dir)
                      if os.path.isdir(os.path.join(clip_dir, d)) and d.isdigit()])
    if len(ts_dirs) != 3:
        print(f"  [{method}] 警告: 时间戳目录数={len(ts_dirs)}, 期望3")

    # 前3个按顺序当作 near/middle/far
    count = 0
    for dist, ts in zip(DISTANCES, ts_dirs):
        for cam in CAMERAS:
            src = os.path.join(clip_dir, ts, "vehicle_renders", cam, "render.png")
            if os.path.exists(src):
                dst_dir = os.path.join(output_dir, method, dist)
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, f"{cam}.png")
                shutil.copy2(src, dst)
                count += 1
    return count


def collect_sparse_method(method, root, clip_num, output_dir):
    """方法类型2 (SparseGS): scene{NNN}_{dist}/vehicle_render_{dist}/vehicle_renders/{cam}/render.png"""
    count = 0
    for dist in DISTANCES:
        scene_dir = os.path.join(root, f"scene{clip_num}_{dist}")
        if not os.path.isdir(scene_dir):
            continue
        for cam in CAMERAS:
            src = os.path.join(scene_dir, f"vehicle_render_{dist}",
                               "vehicle_renders", cam, "render.png")
            if os.path.exists(src):
                dst_dir = os.path.join(output_dir, method, dist)
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, f"{cam}.png")
                shutil.copy2(src, dst)
                count += 1
    return count


def collect_s3gaussian(method, root, clip_num, output_dir):
    """方法类型3 (S3Gaussian): scene{NNN}_{dist}/{cam}.png"""
    count = 0
    for dist in DISTANCES:
        scene_dir = os.path.join(root, f"scene{clip_num}_{dist}")
        if not os.path.isdir(scene_dir):
            continue
        for cam in CAMERAS:
            src = os.path.join(scene_dir, f"{cam}.png")
            if os.path.exists(src):
                dst_dir = os.path.join(output_dir, method, dist)
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, f"{cam}.png")
                shutil.copy2(src, dst)
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=str, required=True,
                        help="clip 编号，如 053")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录（默认 ./visualizations/clip{num}）")
    args = parser.parse_args()

    output_dir = args.output or f"./visualizations/clip{args.clip}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"收集 clip {args.clip} 的所有 GS 方法渲染结果")
    print(f"输出目录: {output_dir}\n")

    total = 0
    for method, root in GS_METHODS.items():
        if method in ["DDGS", "DropGaussian", "Co-Adaptation", "AD-GS"]:
            n = collect_timestamp_method(method, root, args.clip, output_dir)
        elif method == "SparseGS":
            n = collect_sparse_method(method, root, args.clip, output_dir)
        elif method == "S3Gaussian":
            n = collect_s3gaussian(method, root, args.clip, output_dir)
        else:
            continue
        print(f"  {method}: 复制了 {n} 张图")
        total += n

    print(f"\n总计: {total} 张图已复制到 {output_dir}")
    print(f"\n目录结构: {output_dir}/{{method}}/{{near|middle|far}}/{{camera}}.png")


if __name__ == "__main__":
    main()
