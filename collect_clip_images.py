#!/usr/bin/env python3
"""
收集指定 clip 的所有 GS 方法渲染结果（+ 对应GT），复制到统一目录方便查看对比。

直接复用 gs_eval.py 的路径定义和 GT 匹配逻辑，保证路径100%正确。

使用：
    python collect_clip_images.py --clip 076
    python collect_clip_images.py --clip 076 --output ./vis/clip076

输出结构：
    {output}/gt/{near|middle|far}/{camera}.jpg
    {output}/{method}/{near|middle|far}/{camera}.png
"""

import os
import shutil
import argparse

from gs_eval import (
    METHODS, CLIP_TIMESTAMPS, CAMERAS, DISTANCES,
    get_gen_path, find_gt_path,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=str, required=True,
                        help="clip 编号，如 076")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录（默认 ./visualizations/clip{num}）")
    args = parser.parse_args()

    # 在 CLIP_TIMESTAMPS 中找完整 clip 名（输入可以是数字前缀如"076"或完整名）
    matching_clips = [c for c in CLIP_TIMESTAMPS
                      if c == args.clip or c.startswith(f"{args.clip}_")]
    if not matching_clips:
        print(f"错误: clip '{args.clip}' 不在 gs_eval 的 CLIP_TIMESTAMPS 中")
        print(f"可用的 clip: {list(CLIP_TIMESTAMPS.keys())}")
        return
    clip_full = matching_clips[0]
    print(f"匹配 clip: {clip_full}")

    output_dir = args.output or f"./visualizations/clip{args.clip}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}\n")

    # 复制 GT 图像
    print("[GT]")
    gt_count = 0
    gt_missing = 0
    for dist in DISTANCES:
        for cam in CAMERAS:
            gt_path = find_gt_path(clip_full, dist, cam)
            if gt_path is None:
                gt_missing += 1
                continue
            dst_dir = os.path.join(output_dir, "gt", dist)
            os.makedirs(dst_dir, exist_ok=True)
            # 保留原始扩展名（GT 是 .jpg）
            ext = os.path.splitext(gt_path)[1]
            dst = os.path.join(dst_dir, f"{cam}{ext}")
            shutil.copy2(gt_path, dst)
            gt_count += 1
    print(f"  复制了 {gt_count} 张 (缺失 {gt_missing})")

    # 复制各方法的渲染
    total = gt_count
    for method in METHODS:
        print(f"\n[{method}]")
        count = 0
        missing = 0
        for dist in DISTANCES:
            for cam in CAMERAS:
                src = get_gen_path(method, clip_full, dist, cam)
                if not os.path.exists(src):
                    missing += 1
                    continue
                dst_dir = os.path.join(output_dir, method, dist)
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, f"{cam}.png")
                shutil.copy2(src, dst)
                count += 1
        print(f"  复制了 {count} 张 (缺失 {missing})")
        total += count

    print(f"\n总计: {total} 张图")
    print(f"目录结构: {output_dir}/{{gt|method}}/{{near|middle|far}}/{{camera}}.{{png|jpg}}")


if __name__ == "__main__":
    main()
