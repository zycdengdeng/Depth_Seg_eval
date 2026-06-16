#!/usr/bin/env python3
"""
下游感知定性可视化：gt | baseline | gsnet 三联对比图。

对每个采样帧，每种模态各出一张 1x3 拼接图（列 = GT / Baseline / GS-Net）：
    <out>/<group>/rgb/<frame>.png     原图对照
    <out>/<group>/depth/<frame>.png   深度图（baseline/gsnet 已对齐到 gt 尺度，同色阶）
    <out>/<group>/seg/<frame>.png     语义分割彩色图（Cityscapes 调色板）
    <out>/<group>/sam/<frame>.png     SAM 边缘图

数据来源（图像帧模式，与评测同一套）：
    baseline_root/<group>/{gen,gt}/<frame>.png
    gsnet_root/<group>/{gen,gt}/<frame>.png
gt 取自 baseline_root（两次 run 的 gt 完全相同）。

用法：
    python downstream/visualize_compare.py \
        --baseline-root /mnt/zihanw/downstream_cse/baseline/_eval_frames \
        --gsnet-root    /mnt/zihanw/downstream_cse/gsnet/_eval_frames \
        --out           /mnt/zihanw/downstream_cse/compare_vis \
        --tasks rgb depth seg sam \
        --every 10

可选：
    --groups 110 210    只可视化部分序列/clip（默认全部）
    --every N           每隔 N 帧出一张（默认 10；=1 则每帧都出）
    --max-per-group K   每个 group 最多出 K 帧
    --config FILE       从评测 config 读模型设置（depth_size/backbone/sam_size），默认与下游 config 一致
    --gpu N             指定 GPU（默认自动选空闲卡）
"""

import os
import sys
import glob
import argparse
import numpy as np
from typing import List, Optional

# 保证能 import 库里的模块（本文件在 downstream/ 下）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_image, ensure_dir, get_cityscapes_palette, align_depth_scale


COL_TITLES = ["GT", "Baseline", "GS-Net"]


def list_groups(baseline_root: str, gsnet_root: str,
                groups: Optional[List[str]]) -> List[str]:
    if groups:
        return groups
    found = []
    for name in sorted(os.listdir(baseline_root)):
        if os.path.isdir(os.path.join(baseline_root, name, "gt")) and \
           os.path.isdir(os.path.join(gsnet_root, name, "gen")):
            found.append(name)
    return found


def sample_frames(gt_dir: str, every: int, max_per_group: Optional[int],
                  frames: Optional[List[str]] = None) -> List[str]:
    files = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    files = [os.path.basename(f) for f in files]
    if frames:
        # 指定帧：允许带或不带 .png 后缀
        wanted = {f if f.endswith(".png") else f + ".png" for f in frames}
        picked = [f for f in files if f in wanted]
        return picked
    picked = files[::max(1, every)]
    if max_per_group:
        picked = picked[:max_per_group]
    return picked


def save_triptych(images, out_path, cmap=None, vrange=None, titles=COL_TITLES):
    """把三张图横排成一张，列标题 GT/Baseline/GS-Net。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        if cmap is not None:
            vmin = vrange[0] if vrange else None
            vmax = vrange[1] if vrange else None
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=15)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def colorize_seg(seg: np.ndarray, palette: np.ndarray) -> np.ndarray:
    h, w = seg.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id in np.unique(seg):
        if 0 <= label_id < len(palette):
            color[seg == label_id] = palette[label_id]
    return color


def main():
    parser = argparse.ArgumentParser(
        description="gt|baseline|gsnet 三联可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--baseline-root", required=True,
                        help="baseline 的 _eval_frames 目录（gt 也从这里取）")
    parser.add_argument("--gsnet-root", required=True,
                        help="gsnet 的 _eval_frames 目录")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--tasks", nargs="+", default=["rgb", "depth", "seg", "sam"],
                        choices=["rgb", "depth", "seg", "sam"],
                        help="要出哪些模态的三联图")
    parser.add_argument("--groups", nargs="+", default=None,
                        help="只可视化部分 group（默认全部）")
    parser.add_argument("--every", type=int, default=10, help="每隔 N 帧出一张")
    parser.add_argument("--frames", nargs="+", default=None,
                        help="只可视化指定帧（如 00012 00034，可带或不带.png）；指定后忽略 --every")
    parser.add_argument("--max-per-group", type=int, default=None,
                        help="每个 group 最多出多少帧")
    parser.add_argument("--pairs", nargs="+", default=None,
                        help="精确指定 场景:帧 对（如 510:00044 410:00002）；"
                             "用于把分散在不同场景的最佳帧一次出图。指定后忽略 --groups/--frames")
    parser.add_argument("--flat", action="store_true",
                        help="所有图平铺到同一个输出文件夹，命名 <group>_<frame>_<modality>.png")
    parser.add_argument("--config", type=str, default=None,
                        help="评测 config，用于读模型设置（可选）")
    parser.add_argument("--gpu", type=int, default=None,
                        help="指定 GPU（默认自动选空闲卡）")
    args = parser.parse_args()

    # 模型设置（默认与下游 config 一致）
    depth_size, seg_backbone, sam_size = "large", "swin-l", "large"
    if args.config and os.path.exists(args.config):
        import yaml
        cfg = yaml.safe_load(open(args.config))
        depth_size = cfg.get("depth", {}).get("model_size", depth_size)
        seg_backbone = cfg.get("segmentation", {}).get("backbone", seg_backbone)
        sam_size = cfg.get("sam", {}).get("model_size", sam_size)

    # 选 GPU
    if args.gpu is not None:
        gpu_id = args.gpu
    else:
        from gpu_utils import select_free_gpus
        gpu_id = select_free_gpus(min_free_mb=15000, max_gpus=1)[0]
    device = f"cuda:{gpu_id}"
    print(f"使用 {device}")

    # 按需加载模型
    depth_model = seg_model = sam_model = palette = None
    if "depth" in args.tasks:
        from depth_eval import DepthAnythingV2Estimator
        depth_model = DepthAnythingV2Estimator(model_size=depth_size, device=device)
    if "seg" in args.tasks:
        from seg_eval import Mask2FormerSegmentor
        seg_model = Mask2FormerSegmentor(backbone=seg_backbone, device=device)
        palette = get_cityscapes_palette()
    if "sam" in args.tasks:
        from sam_eval import SAMSegmentorFast
        sam_model = SAMSegmentorFast(model_size=sam_size, device=device)

    # 构造 {group: [frame.png, ...]} 待处理表
    if args.pairs:
        work = {}
        for p in args.pairs:
            if ":" not in p:
                print(f"[skip] --pairs 格式应为 场景:帧，收到: {p}")
                continue
            grp, fr = p.split(":", 1)
            fr = fr if fr.endswith(".png") else fr + ".png"
            work.setdefault(grp, []).append(fr)
        print(f"精确指定 {sum(len(v) for v in work.values())} 帧，跨 {len(work)} 个场景")
    else:
        groups = list_groups(args.baseline_root, args.gsnet_root, args.groups)
        work = {}
        for grp in groups:
            gt_dir = os.path.join(args.baseline_root, grp, "gt")
            work[grp] = sample_frames(gt_dir, args.every, args.max_per_group, args.frames)
        print(f"将可视化 {len(groups)} 个 group: {groups}")

    def out_path(grp, stem, modality):
        """flat=同一文件夹；否则 <out>/<group>/<modality>/"""
        if args.flat:
            ensure_dir(args.out)
            return os.path.join(args.out, f"{grp}_{stem}_{modality}.png")
        d = os.path.join(args.out, grp, modality)
        ensure_dir(d)
        return os.path.join(d, f"{stem}.png")

    total = 0
    for grp, frames in work.items():
        gt_dir = os.path.join(args.baseline_root, grp, "gt")
        base_dir = os.path.join(args.baseline_root, grp, "gen")
        gs_dir = os.path.join(args.gsnet_root, grp, "gen")
        if not (os.path.isdir(gt_dir) and os.path.isdir(base_dir) and os.path.isdir(gs_dir)):
            print(f"[skip] {grp}: 缺 gt/gen 目录")
            continue

        print(f"\n[{grp}] {len(frames)} 帧")
        for fname in frames:
            gt_p = os.path.join(gt_dir, fname)
            base_p = os.path.join(base_dir, fname)
            gs_p = os.path.join(gs_dir, fname)
            if not (os.path.exists(gt_p) and os.path.exists(base_p) and os.path.exists(gs_p)):
                print(f"  [skip] {fname}: gt/baseline/gsnet 缺同名帧")
                continue

            gt_rgb = load_image(gt_p)
            base_rgb = load_image(base_p)
            gs_rgb = load_image(gs_p)
            stem = os.path.splitext(fname)[0]

            if "rgb" in args.tasks:
                save_triptych([gt_rgb, base_rgb, gs_rgb], out_path(grp, stem, "rgb"))

            if "depth" in args.tasks:
                dg = depth_model.predict(gt_rgb)
                db = depth_model.predict(base_rgb)
                ds = depth_model.predict(gs_rgb)
                # baseline/gsnet 对齐到 gt 尺度，三者共用 gt 的色阶
                db = align_depth_scale(db, dg, method="median")
                ds = align_depth_scale(ds, dg, method="median")
                vmin, vmax = np.percentile(dg, [2, 98])
                save_triptych([dg, db, ds], out_path(grp, stem, "depth"),
                              cmap="magma", vrange=(vmin, vmax))

            if "seg" in args.tasks:
                sg = colorize_seg(seg_model.predict(gt_rgb), palette)
                sb = colorize_seg(seg_model.predict(base_rgb), palette)
                ss = colorize_seg(seg_model.predict(gs_rgb), palette)
                save_triptych([sg, sb, ss], out_path(grp, stem, "seg"))

            if "sam" in args.tasks:
                eg = sam_model.get_edge_map(gt_rgb)
                eb = sam_model.get_edge_map(base_rgb)
                es = sam_model.get_edge_map(gs_rgb)
                save_triptych([eg, eb, es], out_path(grp, stem, "sam"),
                              cmap="gray", vrange=(0, 1))

            total += 1
            print(f"  ✓ {grp}/{stem}")

    print(f"\n完成：共 {total} 帧，输出在 {args.out}")
    if args.flat:
        print("目录结构（平铺）: <out>/<group>_<frame>_<modality>.png")
    else:
        print("目录结构: <out>/<group>/{rgb,depth,seg,sam}/<frame>.png")


if __name__ == "__main__":
    main()
