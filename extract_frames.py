#!/usr/bin/env python3
"""
从MP4视频中提取帧，转换为evaluate.py所需的目录结构。

MP4数据结构:
    {mp4_root}/{segment}/{camera}_generated.mp4
    {mp4_root}/{segment}/{camera}_gt.mp4

输出目录结构:
    {output_root}/{camera}/gen/{segment}_{frame:04d}.png
    {output_root}/{camera}/gt/{segment}_{frame:04d}.png
"""

import os
import sys
import argparse
import cv2
from pathlib import Path
from tqdm import tqdm

CAMERAS = [
    "cross_left_120fov",
    "cross_right_120fov",
    "front_tele_30fov",
    "front_wide_120fov",
    "rear_left_70fov",
    "rear_right_70fov",
    "rear_tele_30fov",
]

EXCLUDE_FOLDERS = {"Depth_Seg_eval_Result", "guidance", "visualize_controls.py"}


def get_segments(mp4_root):
    """获取所有数据段文件夹"""
    segments = []
    for name in sorted(os.listdir(mp4_root)):
        full_path = os.path.join(mp4_root, name)
        if os.path.isdir(full_path) and name not in EXCLUDE_FOLDERS:
            segments.append(name)
    return segments


def extract_video_frames(video_path, output_dir, prefix, frame_step=1):
    """从单个视频中提取帧并保存为PNG"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  警告: 无法打开 {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            filename = f"{prefix}_{frame_idx:04d}.png"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description="MP4抽帧工具")
    parser.add_argument("--mp4-root", type=str,
                        default="/mnt/zyc_wzh/inference_SLAM_output",
                        help="MP4数据根目录")
    parser.add_argument("--output", type=str,
                        default="/mnt/zyc_wzh/inference_SLAM_output/_eval_frames",
                        help="输出帧目录")
    parser.add_argument("--frame-step", type=int, default=1,
                        help="帧采样间隔 (1=全部, 2=隔一帧)")
    parser.add_argument("--cameras", type=str, default=None,
                        help="指定相机，逗号分隔（默认全部）")
    args = parser.parse_args()

    mp4_root = args.mp4_root
    output_root = args.output
    frame_step = args.frame_step

    cameras = CAMERAS
    if args.cameras:
        cameras = [c.strip() for c in args.cameras.split(',')]

    segments = get_segments(mp4_root)
    print(f"MP4数据目录: {mp4_root}")
    print(f"输出目录: {output_root}")
    print(f"数据段数量: {len(segments)}")
    print(f"相机数量: {len(cameras)}")
    print(f"帧采样间隔: {frame_step}")
    print()

    # 创建输出目录
    for camera in cameras:
        os.makedirs(os.path.join(output_root, camera, "gen"), exist_ok=True)
        os.makedirs(os.path.join(output_root, camera, "gt"), exist_ok=True)

    total_frames = 0

    for segment in tqdm(segments, desc="处理数据段"):
        segment_dir = os.path.join(mp4_root, segment)

        for camera in cameras:
            gen_video = os.path.join(segment_dir, f"{camera}_generated.mp4")
            gt_video = os.path.join(segment_dir, f"{camera}_gt.mp4")

            if not os.path.exists(gen_video):
                print(f"  跳过: {gen_video} 不存在")
                continue
            if not os.path.exists(gt_video):
                print(f"  跳过: {gt_video} 不存在")
                continue

            gen_out = os.path.join(output_root, camera, "gen")
            gt_out = os.path.join(output_root, camera, "gt")

            n_gen = extract_video_frames(gen_video, gen_out, segment, frame_step)
            n_gt = extract_video_frames(gt_video, gt_out, segment, frame_step)

            total_frames += n_gen + n_gt

    print(f"\n抽帧完成! 共提取 {total_frames} 帧")
    print(f"输出目录: {output_root}")
    print(f"\n现在可以运行:")
    print(f"  CUDA_VISIBLE_DEVICES=6,7 python evaluate.py --config config.yaml --task all --parallel --gpus 0,1")


if __name__ == "__main__":
    main()
