#!/usr/bin/env python3
"""
汇总公共帧评测结果为论文主表（GS 方法 + Ours）

用法：
    python common_table.py --gs-dir ./results/gs_eval_common --tf-dir ./results/tf_eval_common
"""

import os
import json
import argparse

# (任务, 指标key, 表头, 越高越好, 小数位)
COLUMNS = [
    ("image_metrics", "psnr",        "PSNR↑",          True,  2),
    ("image_metrics", "ssim",        "SSIM↑",          True,  3),
    ("image_metrics", "lpips",       "LPIPS↓",         False, 3),
    ("depth",         "abs_rel",     "Abs-Rel↓",       False, 3),
    ("depth",         "delta_1",     "δ1(%)↑",         True,  1),
    ("seg",           "consistency", "Seg-Cons.(%)↑",  True,  1),
    ("seg",           "coarse_miou", "Coarse-mIoU(%)↑", True, 1),
    ("sam",           "edge_f1",     "SAM-F1(%)↑",     True,  1),
]

METHOD_ORDER = ["DDGS", "DropGaussian", "Co-Adaptation", "AD-GS",
                "SparseGS", "S3Gaussian", "Ours"]


def load_task(result_dir, task):
    path = os.path.join(result_dir, f"{task}_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="公共帧评测主表")
    parser.add_argument("--gs-dir", default="./results/gs_eval_common")
    parser.add_argument("--tf-dir", default="./results/tf_eval_common")
    args = parser.parse_args()

    # table[method][header] = value；num[method] = 帧对数
    table = {m: {} for m in METHOD_ORDER}
    for task, key, header, _, _ in COLUMNS:
        gs = load_task(args.gs_dir, task) or {}
        for method, res in gs.items():
            table.setdefault(method, {})[header] = res.get("overall", {}).get(key)
        tf = load_task(args.tf_dir, task)
        if tf:
            table["Ours"][header] = tf.get("overall", {}).get(key)

    headers = [c[2] for c in COLUMNS]
    print(f"{'方法':<16}" + "".join(f"{h:>17}" for h in headers))
    for method in [m for m in table if table[m]]:
        row = f"{method:<16}"
        for _, _, header, _, prec in COLUMNS:
            v = table[method].get(header)
            row += f"{'N/A':>17}" if v is None else f"{v:>17.{prec}f}"
        print(row)

    # 每列最优（不含缺失）
    print()
    for _, _, header, higher, _ in COLUMNS:
        vals = {m: t[header] for m, t in table.items() if t.get(header) is not None}
        if vals:
            best = (max if higher else min)(vals, key=vals.get)
            print(f"  {header:<16} 最优: {best}")


if __name__ == "__main__":
    main()
