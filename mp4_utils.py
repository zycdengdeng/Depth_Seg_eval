"""
MP4视频处理工具

提供从MP4视频读取帧的功能，用于视频数据集的评测。
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Iterator, Optional
import yaml


def load_mp4_config(config_path: str = "config_mp4.yaml") -> Dict:
    """加载MP4评测配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_segment_folders(config: Dict) -> List[str]:
    """
    获取所有数据段文件夹

    Args:
        config: 配置字典

    Returns:
        段文件夹路径列表
    """
    root = config['data']['root']
    exclude = set(config['data'].get('exclude_folders', []))

    segments = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(path) and name not in exclude:
            segments.append(name)

    return segments


def get_video_paths(config: Dict) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    获取所有视频对的路径

    Args:
        config: 配置字典

    Returns:
        {camera: [(gen_video_path, gt_video_path, segment_name), ...]}
    """
    root = config['data']['root']
    cameras = config['data']['cameras']
    gen_suffix = config['data'].get('gen_suffix', '_generated')
    gt_suffix = config['data'].get('gt_suffix', '_gt')

    segments = get_segment_folders(config)

    video_pairs = {cam: [] for cam in cameras}

    for segment in segments:
        seg_path = os.path.join(root, segment)

        for camera in cameras:
            gen_video = os.path.join(seg_path, f"{camera}{gen_suffix}.mp4")
            gt_video = os.path.join(seg_path, f"{camera}{gt_suffix}.mp4")

            if os.path.exists(gen_video) and os.path.exists(gt_video):
                video_pairs[camera].append((gen_video, gt_video, segment))

    # 过滤掉没有视频的相机
    video_pairs = {k: v for k, v in video_pairs.items() if v}

    return video_pairs


def read_video_frames(video_path: str, frame_step: int = 1) -> List[np.ndarray]:
    """
    读取视频的所有帧

    Args:
        video_path: 视频文件路径
        frame_step: 帧采样步长 (1=全部帧)

    Returns:
        帧列表 [(H, W, 3), ...], RGB格式, uint8
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        frame_idx += 1

    cap.release()
    return frames


def iter_video_frame_pairs(gen_path: str, gt_path: str,
                           frame_step: int = 1) -> Iterator[Tuple[np.ndarray, np.ndarray, int]]:
    """
    迭代器：逐帧读取生成视频和真值视频的帧对

    Args:
        gen_path: 生成视频路径
        gt_path: 真值视频路径
        frame_step: 帧采样步长

    Yields:
        (gen_frame, gt_frame, frame_idx)
    """
    cap_gen = cv2.VideoCapture(gen_path)
    cap_gt = cv2.VideoCapture(gt_path)

    if not cap_gen.isOpened():
        raise ValueError(f"无法打开生成视频: {gen_path}")
    if not cap_gt.isOpened():
        raise ValueError(f"无法打开真值视频: {gt_path}")

    frame_idx = 0
    output_idx = 0

    while True:
        ret_gen, frame_gen = cap_gen.read()
        ret_gt, frame_gt = cap_gt.read()

        if not ret_gen or not ret_gt:
            break

        if frame_idx % frame_step == 0:
            # BGR -> RGB
            frame_gen_rgb = cv2.cvtColor(frame_gen, cv2.COLOR_BGR2RGB)
            frame_gt_rgb = cv2.cvtColor(frame_gt, cv2.COLOR_BGR2RGB)
            yield frame_gen_rgb, frame_gt_rgb, output_idx
            output_idx += 1

        frame_idx += 1

    cap_gen.release()
    cap_gt.release()


def get_video_info(video_path: str) -> Dict:
    """
    获取视频信息

    Args:
        video_path: 视频文件路径

    Returns:
        {frame_count, fps, width, height, duration}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    info = {
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info['duration'] = info['frame_count'] / info['fps'] if info['fps'] > 0 else 0

    cap.release()
    return info


def get_frame_pairs_from_videos(config: Dict) -> Dict[str, List[Tuple[str, str, str, int]]]:
    """
    获取所有帧对信息（不实际读取帧，只返回索引信息）

    Args:
        config: 配置字典

    Returns:
        {camera: [(gen_video, gt_video, segment, frame_count), ...]}
    """
    video_pairs = get_video_paths(config)
    frame_step = config.get('processing', {}).get('frame_step', 1)

    frame_info = {}

    for camera, pairs in video_pairs.items():
        frame_info[camera] = []
        for gen_path, gt_path, segment in pairs:
            info = get_video_info(gen_path)
            effective_frames = (info['frame_count'] + frame_step - 1) // frame_step
            frame_info[camera].append((gen_path, gt_path, segment, effective_frames))

    return frame_info


def count_total_frames(config: Dict) -> Dict[str, int]:
    """
    统计每个相机的总帧数

    Args:
        config: 配置字典

    Returns:
        {camera: total_frame_count}
    """
    frame_info = get_frame_pairs_from_videos(config)

    counts = {}
    for camera, info_list in frame_info.items():
        counts[camera] = sum(item[3] for item in info_list)

    return counts


def ensure_dir(path: str):
    """确保目录存在"""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ============== 测试代码 ==============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MP4工具测试")
    parser.add_argument("--config", type=str, default="config_mp4.yaml")
    args = parser.parse_args()

    config = load_mp4_config(args.config)

    print("=" * 60)
    print("MP4数据集信息")
    print("=" * 60)

    segments = get_segment_folders(config)
    print(f"\n数据段数量: {len(segments)}")
    print(f"数据段列表: {segments[:5]}...")

    video_pairs = get_video_paths(config)
    print(f"\n相机数量: {len(video_pairs)}")
    for camera, pairs in video_pairs.items():
        print(f"  {camera}: {len(pairs)} 个视频对")

    frame_counts = count_total_frames(config)
    print(f"\n帧数统计:")
    total = 0
    for camera, count in frame_counts.items():
        print(f"  {camera}: {count} 帧")
        total += count
    print(f"  总计: {total} 帧")

    # 测试读取一个视频的信息
    if video_pairs:
        first_camera = list(video_pairs.keys())[0]
        first_pair = video_pairs[first_camera][0]
        print(f"\n示例视频信息 ({first_pair[2]}/{first_camera}):")
        info = get_video_info(first_pair[0])
        for k, v in info.items():
            print(f"  {k}: {v}")
