"""
按 manifest 计算全部指标

输出（out_dir 下）：
    per_sample.csv   逐样本指标
    summary.json     各方法均值 + clip 级 bootstrap 95% CI（含分组）、KID/FID
    table.md / table.tex
"""

import csv
import os
import shutil
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from .loader import ImageLoader
from .manifest import read_manifest, check_alignment
from .report import build_summary, format_table, save_summary
from .scorers import Scorer, compute_distribution_metrics


def _sort_key(row: Dict):
    ts = row["timestamp"]
    return (row["clip"], row["camera"], float(ts) if ts else 0.0, row["sample_id"])


def run(manifest_path: str, out_dir: str,
        groups: List[str], device: str = "cuda",
        eval_size=(1280, 720), gt_undistort: str = "none",
        max_depth_ratio: float = 100.0, allow_partial: bool = False,
        keep_images: bool = False, methods: Optional[List[str]] = None) -> Dict:
    rows = read_manifest(manifest_path)
    if methods:
        rows = [r for r in rows if r["method"] in methods]
    if not rows:
        raise ValueError("manifest 中没有可评测的样本")

    align = check_alignment(rows)
    print("\n样本对齐检查:")
    for method, info in align["per_method"].items():
        print(f"  {method:<20} {info['n']:>6} 个样本, 缺少 {info['missing']}")
    if not allow_partial:
        dropped = len(rows)
        rows = [r for r in rows if r["sample_id"] in align["common"]]
        dropped -= len(rows)
        print(f"  只评测所有方法共有的 {len(align['common'])} 个样本"
              f"（丢弃 {dropped} 行；如需各评各的用 --allow-partial，结果不可直接比较）")

    by_sample = defaultdict(list)
    for r in rows:
        by_sample[r["sample_id"]].append(r)
    samples = sorted(by_sample, key=lambda s: _sort_key(by_sample[s][0]))

    os.makedirs(out_dir, exist_ok=True)
    image_dir = os.path.join(out_dir, "_images")
    loader = ImageLoader(eval_size=eval_size, gt_undistort=gt_undistort)
    scorer = Scorer(groups, device=device, max_depth_ratio=max_depth_ratio)

    records = []
    conf = defaultdict(dict)
    for idx, sample_id in enumerate(tqdm(samples, desc="评测样本")):
        sample_rows = sorted(by_sample[sample_id], key=lambda r: r["method"])
        first = sample_rows[0]
        gt_img = loader.load_gt(first["gt"], first["camera"] or None)
        ref = scorer.reference(gt_img)

        for row in sample_rows:
            gen_img = loader.load_gen(row["gen"])
            values, extras = scorer.score(gen_img, gt_img, ref)
            records.append({"method": row["method"], "sample_id": sample_id,
                            "clip": row["clip"], "camera": row["camera"],
                            "timestamp": row["timestamp"], "group": row["group"], **values})
            if "coarse_conf_matrix" in extras:
                conf[row["method"]][sample_id] = extras["coarse_conf_matrix"]

            if "dist" in scorer.groups:
                # 每个方法一对目录：gen 与其样本对应的 GT（同名），保证 KID/FID 的两组样本数相同
                for sub, img in (("gen", gen_img), ("gt", gt_img)):
                    d = os.path.join(image_dir, row["method"], sub)
                    os.makedirs(d, exist_ok=True)
                    Image.fromarray(img).save(os.path.join(d, f"{idx:06d}.png"))

    dist = {}
    if "dist" in scorer.groups:
        for method in sorted({r["method"] for r in records}):
            print(f"\n计算 KID/FID: {method}")
            d = os.path.join(image_dir, method)
            dist[method] = compute_distribution_metrics(
                os.path.join(d, "gen"), os.path.join(d, "gt"), device)
        if not keep_images:
            shutil.rmtree(image_dir, ignore_errors=True)

    _write_per_sample(records, os.path.join(out_dir, "per_sample.csv"))
    summary = build_summary(records, conf, dist)
    save_summary(summary, os.path.join(out_dir, "summary.json"))
    for fmt, ext in (("markdown", "md"), ("latex", "tex")):
        with open(os.path.join(out_dir, f"table.{ext}"), "w") as f:
            f.write(format_table(summary, fmt=fmt) + "\n")

    print("\n" + format_table(summary, fmt="markdown"))
    print(f"\n结果已保存到: {out_dir}")
    return summary


def _write_per_sample(records: List[Dict], path: str) -> None:
    keys = []
    for r in records:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
