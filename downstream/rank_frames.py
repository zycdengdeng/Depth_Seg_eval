#!/usr/bin/env python3
"""
逐帧给 baseline vs gsnet 的下游指标排序，帮你挑"指标最好/优势最大"的帧做可视化展示。

评测 JSON 只存到 per-group 聚合，挑具体某一帧需要逐帧重算，本脚本干这个：
对每帧分别算 baseline 和 gsnet 的指标，输出
    1) 每个 group(序列/clip) 的平均优势（哪个场景整体最好）
    2) 帧级排序表（哪个序列的哪一帧 gsnet 赢最多 / gsnet 绝对值最好）

可排序的指标：
    depth_abs_rel  (越低越好, gsnet优势=base-gsnet)   ← 默认
    depth_delta_1  (越高越好)
    seg_consistency(越高越好)
    seg_miou       (越高越好)
    sam_edge_f1    (越高越好)

用法：
    python downstream/rank_frames.py \
        --baseline-root /mnt/zihanw/downstream_cse/baseline/_eval_frames \
        --gsnet-root    /mnt/zihanw/downstream_cse/gsnet/_eval_frames \
        --metric depth_abs_rel \
        --top 20 \
        --out /mnt/zihanw/downstream_cse/rank_depth.csv

挑完帧后用 visualize_compare.py 出图（支持指定帧）：
    python downstream/visualize_compare.py ... --groups 410 --frames 00012 00034
"""

import os
import sys
import glob
import csv
import argparse
import numpy as np
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_image, align_depth_scale


# metric -> (所需模型, 结果字段, 方向)
METRIC_SPEC = {
    "depth_abs_rel":   ("depth", "abs_rel",     "down"),
    "depth_delta_1":   ("depth", "delta_1",     "up"),
    "seg_consistency": ("seg",   "consistency", "up"),
    "seg_miou":        ("seg",   "miou",        "up"),
    "sam_edge_f1":     ("sam",   "edge_f1",     "up"),
}


def list_groups(baseline_root, gsnet_root, groups):
    if groups:
        return groups
    found = []
    for name in sorted(os.listdir(baseline_root)):
        if os.path.isdir(os.path.join(baseline_root, name, "gt")) and \
           os.path.isdir(os.path.join(gsnet_root, name, "gen")):
            found.append(name)
    return found


def advantage(direction, base_val, gsnet_val):
    """gsnet 相对 baseline 的优势（正数=gsnet更好），按方向归一。"""
    if base_val is None or gsnet_val is None:
        return None
    return (base_val - gsnet_val) if direction == "down" else (gsnet_val - base_val)


def main():
    parser = argparse.ArgumentParser(
        description="逐帧排序挑展示帧",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--gsnet-root", required=True)
    parser.add_argument("--metric", default="depth_abs_rel",
                        choices=list(METRIC_SPEC.keys()))
    parser.add_argument("--groups", nargs="+", default=None)
    parser.add_argument("--every", type=int, default=1,
                        help="每隔 N 帧算一次（默认1=全部帧）")
    parser.add_argument("--top", type=int, default=20, help="打印前 K 帧")
    parser.add_argument("--sort-by", default="advantage",
                        choices=["advantage", "gsnet"],
                        help="advantage=按gsnet优势排; gsnet=按gsnet绝对值好坏排")
    parser.add_argument("--out", type=str, default=None, help="完整结果写到 CSV")
    parser.add_argument("--config", type=str, default=None,
                        help="评测 config，用于读模型设置（可选）")
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()

    model_kind, field, direction = METRIC_SPEC[args.metric]

    depth_size, seg_backbone, sam_size = "large", "swin-l", "large"
    if args.config and os.path.exists(args.config):
        import yaml
        cfg = yaml.safe_load(open(args.config))
        depth_size = cfg.get("depth", {}).get("model_size", depth_size)
        seg_backbone = cfg.get("segmentation", {}).get("backbone", seg_backbone)
        sam_size = cfg.get("sam", {}).get("model_size", sam_size)

    if args.gpu is not None:
        gpu_id = args.gpu
    else:
        from gpu_utils import select_free_gpus
        gpu_id = select_free_gpus(min_free_mb=15000, max_gpus=1)[0]
    device = f"cuda:{gpu_id}"
    print(f"使用 {device}，指标 = {args.metric} ({'越低越好' if direction=='down' else '越高越好'})")

    # 按需加载模型 + 定义"对一帧算指标"的函数
    if model_kind == "depth":
        from depth_eval import DepthAnythingV2Estimator
        from metrics import compute_depth_metrics
        model = DepthAnythingV2Estimator(model_size=depth_size, device=device)

        def score(gen_rgb, gt_rgb):
            dgt = model.predict(gt_rgb)
            dgen = align_depth_scale(model.predict(gen_rgb), dgt, method="median")
            return compute_depth_metrics(dgen, dgt).get(field)

    elif model_kind == "seg":
        from seg_eval import Mask2FormerSegmentor
        from metrics import (compute_segmentation_consistency,
                             compute_segmentation_metrics)
        model = Mask2FormerSegmentor(backbone=seg_backbone, device=device)

        def score(gen_rgb, gt_rgb):
            sg = model.predict(gen_rgb)
            sgt = model.predict(gt_rgb)
            if field == "consistency":
                return compute_segmentation_consistency(sg, sgt).get("consistency")
            return compute_segmentation_metrics(sg, sgt,
                                                num_classes=model.num_classes).get(field)

    else:  # sam
        from sam_eval import SAMSegmentorFast, compute_edge_consistency
        model = SAMSegmentorFast(model_size=sam_size, device=device)

        def score(gen_rgb, gt_rgb):
            eg = model.get_edge_map(gen_rgb)
            egt = model.get_edge_map(gt_rgb)
            return compute_edge_consistency(eg, egt, tolerance=3).get(field)

    groups = list_groups(args.baseline_root, args.gsnet_root, args.groups)
    print(f"将处理 {len(groups)} 个 group: {groups}\n")

    rows = []  # (group, frame, base_val, gsnet_val, adv)
    for grp in groups:
        gt_dir = os.path.join(args.baseline_root, grp, "gt")
        base_dir = os.path.join(args.baseline_root, grp, "gen")
        gs_dir = os.path.join(args.gsnet_root, grp, "gen")
        if not (os.path.isdir(gt_dir) and os.path.isdir(base_dir) and os.path.isdir(gs_dir)):
            print(f"[skip] {grp}: 缺目录")
            continue

        frames = [os.path.basename(f) for f in sorted(glob.glob(os.path.join(gt_dir, "*.png")))]
        frames = frames[::max(1, args.every)]
        print(f"[{grp}] {len(frames)} 帧...")

        for fname in frames:
            base_p = os.path.join(base_dir, fname)
            gs_p = os.path.join(gs_dir, fname)
            if not (os.path.exists(base_p) and os.path.exists(gs_p)):
                continue
            gt_rgb = load_image(os.path.join(gt_dir, fname))
            bval = score(load_image(base_p), gt_rgb)
            gval = score(load_image(gs_p), gt_rgb)
            if bval is None or gval is None or np.isnan(bval) or np.isnan(gval):
                continue
            adv = advantage(direction, bval, gval)
            rows.append((grp, os.path.splitext(fname)[0], float(bval), float(gval), float(adv)))

    if not rows:
        print("没有可用结果。")
        return

    # ---- 每个 group 的平均（哪个场景整体好）----
    print("\n" + "=" * 60)
    print(f"各场景平均（指标={args.metric}）")
    print("=" * 60)
    print(f"{'group':<16}{'baseline':>12}{'gsnet':>12}{'gsnet优势':>12}{'帧数':>8}")
    by_group = {}
    for grp, _f, b, g, a in rows:
        by_group.setdefault(grp, []).append((b, g, a))
    group_summ = []
    for grp, vals in by_group.items():
        mb = np.mean([v[0] for v in vals])
        mg = np.mean([v[1] for v in vals])
        ma = np.mean([v[2] for v in vals])
        group_summ.append((grp, mb, mg, ma, len(vals)))
    # 优势大的场景排前面
    group_summ.sort(key=lambda x: x[3], reverse=True)
    for grp, mb, mg, ma, n in group_summ:
        print(f"{grp:<16}{mb:>12.4f}{mg:>12.4f}{ma:>+12.4f}{n:>8}")
    best_grp = group_summ[0][0]
    print(f"\n→ gsnet 优势最大的场景: {best_grp}")

    # ---- 帧级排序 ----
    if args.sort_by == "advantage":
        rows.sort(key=lambda r: r[4], reverse=True)          # 优势从大到小
        crit = "gsnet优势从大到小"
    else:
        rows.sort(key=lambda r: r[3], reverse=(direction == "up"))  # gsnet绝对值好->差
        crit = "gsnet绝对指标从好到差"

    print("\n" + "=" * 60)
    print(f"帧级排序（{crit}）— 前 {args.top}")
    print("=" * 60)
    print(f"{'#':<4}{'group':<14}{'frame':<10}{'baseline':>12}{'gsnet':>12}{'gsnet优势':>12}")
    for i, (grp, f, b, g, a) in enumerate(rows[:args.top], 1):
        print(f"{i:<4}{grp:<14}{f:<10}{b:>12.4f}{g:>12.4f}{a:>+12.4f}")

    print(f"\n建议：挑上面靠前的几帧做可视化，例如\n"
          f"  python downstream/visualize_compare.py \\\n"
          f"    --baseline-root {args.baseline_root} \\\n"
          f"    --gsnet-root {args.gsnet_root} \\\n"
          f"    --out <out> --groups {rows[0][0]} --frames {rows[0][1]}")

    if args.out:
        with open(args.out, "w", newline="") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["group", "frame", "baseline", "gsnet", "gsnet_advantage"])
            w.writerows(rows)
        print(f"\n[已写入] 完整结果 {args.out}")


if __name__ == "__main__":
    main()
