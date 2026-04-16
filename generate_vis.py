#!/usr/bin/env python3
"""
生成展示用的深度图和分割图

对指定 clip 的 GT + 所有 GS 方法 + TF(Ours) 生成：
- 原图（GT去畸变、GS/TF渲染）
- Depth Anything V2 深度图
- Mask2Former 语义分割图

使用方法：
    python generate_vis.py --clip 031 --device cuda:0
    python generate_vis.py --clip 031 --distances near --cameras FL FW

输出结构：
    {output}/
      gt/{dist}/{cam}_rgb.png
      gt/{dist}/{cam}_depth.png
      gt/{dist}/{cam}_seg.png
      DDGS/{dist}/{cam}_rgb.png
      DDGS/{dist}/{cam}_depth.png
      DDGS/{dist}/{cam}_seg.png
      ...
      TF_Ours/{dist}/{cam}_rgb.png
      ...
"""

import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional

# GS 方法路径复用
from gs_eval import (
    METHODS, CLIP_TIMESTAMPS, CAMERAS, DISTANCES,
    get_gen_path, find_gt_path,
)
from undistort import load_gt_undistorted

# TF 方法
from tf_eval import (
    CLIP_TS_RANGES, CAMERA_SHORT_TO_LONG, GEN_ROOT as TF_GEN_ROOT,
    extract_frames, compute_frame_timestamps, find_clip_full_name,
)


def load_models(device: str):
    """加载深度和分割模型"""
    print("加载模型...")

    # Depth Anything V2
    from depth_eval import get_depth_estimator
    depth_config = {'depth': {'model': 'depth_anything_v2', 'model_size': 'large', 'device': device}}
    depth_model = get_depth_estimator(depth_config)

    # Mask2Former
    from seg_eval import get_segmentor
    seg_config = {'segmentation': {
        'model': 'mask2former', 'backbone': 'swin-l',
        'dataset': 'cityscapes', 'device': device,
        'class_merging': {'enabled': True},
    }}
    seg_model = get_segmentor(seg_config)

    print("模型加载完成\n")
    return depth_model, seg_model


def save_depth_colormap(depth: np.ndarray, path: str):
    """保存深度图为彩色图"""
    # 归一化到 0-255
    d_min, d_max = np.percentile(depth, [2, 98])
    depth_norm = np.clip((depth - d_min) / (d_max - d_min + 1e-8), 0, 1)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    # magma colormap
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
    cv2.imwrite(path, depth_color)


def save_seg_colormap(seg: np.ndarray, path: str):
    """保存分割图为 Cityscapes 调色板彩色图"""
    from utils import get_cityscapes_palette
    palette = get_cityscapes_palette()

    h, w = seg.shape
    color_seg = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id in np.unique(seg):
        if label_id < len(palette):
            color_seg[seg == label_id] = palette[label_id]

    cv2.imwrite(path, cv2.cvtColor(color_seg, cv2.COLOR_RGB2BGR))


def process_image(rgb: np.ndarray, depth_model, seg_model,
                  output_dir: str, prefix: str):
    """对一张 RGB 图像跑深度+分割，保存结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存原图
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f"{prefix}_rgb.png"), rgb_bgr)

    # 深度图
    depth = depth_model.predict(rgb)
    save_depth_colormap(depth, os.path.join(output_dir, f"{prefix}_depth.png"))

    # 分割图
    seg = seg_model.predict(rgb)
    save_seg_colormap(seg, os.path.join(output_dir, f"{prefix}_seg.png"))


def process_gs_methods(clip: str, distances: List[str], cameras: List[str],
                       depth_model, seg_model, output_dir: str):
    """处理所有 GS 方法"""
    for method in METHODS:
        print(f"\n[{method}]")
        count = 0
        for dist in distances:
            for cam in cameras:
                gen_path = get_gen_path(method, clip, dist, cam)
                if not os.path.exists(gen_path):
                    continue
                rgb = np.array(Image.open(gen_path).convert('RGB'))
                method_dir = os.path.join(output_dir, method, dist)
                process_image(rgb, depth_model, seg_model, method_dir, cam)
                count += 1
        print(f"  处理了 {count} 张")


def process_gt(clip: str, distances: List[str], cameras: List[str],
               depth_model, seg_model, output_dir: str):
    """处理 GT 图像（去畸变）"""
    print("\n[GT]")
    count = 0
    for dist in distances:
        for cam in cameras:
            gt_path = find_gt_path(clip, dist, cam)
            if gt_path is None:
                continue
            # 去畸变 + resize 到 1280x720
            rgb = load_gt_undistorted(gt_path, cam, target_size=(1280, 720))
            gt_dir = os.path.join(output_dir, "gt", dist)
            process_image(rgb, depth_model, seg_model, gt_dir, cam)
            count += 1
    print(f"  处理了 {count} 张")


def process_tf_ours(clip_num: str, cameras: List[str],
                    depth_model, seg_model, output_dir: str):
    """处理 TF (Ours) 方法 — 从 MP4 抽全部 29 帧"""
    if clip_num not in CLIP_TS_RANGES:
        print(f"\n[TF_Ours] clip {clip_num} 不在 tf_eval 的时间戳表中，跳过")
        return

    print("\n[TF_Ours]")
    count = 0
    seg = CLIP_TS_RANGES[clip_num]["seg"]
    clip_dir = f"{clip_num}_{seg}"

    for cam in cameras:
        cam_long = CAMERA_SHORT_TO_LONG.get(cam)
        if cam_long is None:
            continue
        video_path = os.path.join(TF_GEN_ROOT, clip_dir,
                                  f"{cam_long}_generated.mp4")
        if not os.path.exists(video_path):
            print(f"  {cam}: 视频不存在 {video_path}")
            continue

        frames = extract_frames(video_path)
        if not frames:
            continue

        # 保存所有帧（按帧号命名）
        for frame_idx, rgb in enumerate(frames):
            tf_dir = os.path.join(output_dir, "TF_Ours", cam)
            process_image(rgb, depth_model, seg_model, tf_dir,
                          f"frame{frame_idx:03d}")
            count += 1

    print(f"  处理了 {count} 张（{len(cameras)} 相机 x {len(frames)} 帧）")


def main():
    parser = argparse.ArgumentParser(description="生成展示用的深度图和分割图")
    parser.add_argument("--clip", type=str, required=True,
                        help="clip编号或完整名（如 031 或 031_car0402_road0402_t9）")
    parser.add_argument("--distances", nargs="+", default=None,
                        choices=DISTANCES,
                        help="距离（默认全部）")
    parser.add_argument("--cameras", nargs="+", default=None,
                        choices=CAMERAS,
                        help="相机（默认全部）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录")
    parser.add_argument("--skip-gs", action="store_true",
                        help="跳过GS方法")
    parser.add_argument("--skip-gt", action="store_true",
                        help="跳过GT")
    parser.add_argument("--skip-tf", action="store_true",
                        help="跳过TF(Ours)")
    args = parser.parse_args()

    distances = args.distances or DISTANCES
    cameras = args.cameras or CAMERAS

    # 解析 clip 名
    matching = [c for c in CLIP_TIMESTAMPS
                if c == args.clip or c.startswith(f"{args.clip}_")]
    if not matching:
        print(f"错误: clip '{args.clip}' 不在 CLIP_TIMESTAMPS 中")
        return
    clip_full = matching[0]
    clip_num = clip_full.split("_")[0]

    output_dir = args.output or f"./visualizations/clip{clip_num}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Clip: {clip_full}")
    print(f"距离: {distances}")
    print(f"相机: {cameras}")
    print(f"输出: {output_dir}\n")

    # 加载模型
    depth_model, seg_model = load_models(args.device)

    # 处理 GT
    if not args.skip_gt:
        process_gt(clip_full, distances, cameras,
                   depth_model, seg_model, output_dir)

    # 处理 GS 方法
    if not args.skip_gs:
        process_gs_methods(clip_full, distances, cameras,
                           depth_model, seg_model, output_dir)

    # 处理 TF (Ours) — 全部帧，不分 distance
    if not args.skip_tf:
        process_tf_ours(clip_num, cameras,
                        depth_model, seg_model, output_dir)

    print(f"\n完成！所有结果保存在 {output_dir}")


if __name__ == "__main__":
    main()
