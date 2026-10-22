#!/usr/bin/env python3
"""
指标参数敏感性分析（公共帧，GS 方法 + Ours）

对没有标准出处的参数，在一组合理取值下重算指标，检查各方法排名是否稳定：
    深度    max_depth_ratio        : 50 / 100(默认) / 200
    分割    是否排除地面类           : 否(默认) / 是
    SAM-F1  边缘阈值 × 容差(对角线比例): {0.2, 0.3(默认), 0.4} × {0.5%(默认), 1%}

每对图的模型推理只做一次（GT 推理按 (clip, 相机) 缓存，所有方法共用），
再在推理结果上换参数重算指标。

用法：
    python sensitivity_eval.py --device cuda:0 --output ./results/sensitivity.json
"""

import os
import json
import argparse
from collections import defaultdict
from typing import Dict, List

import numpy as np
from tqdm import tqdm

DEPTH_RATIOS = [50.0, 100.0, 200.0]
DEFAULT_DEPTH_RATIO = 100.0

SEG_EXCLUDE_FLAT = [False, True]
DEFAULT_EXCLUDE_FLAT = False

SAM_THRESHOLDS = [0.2, 0.3, 0.4]
SAM_TOL_FRACS = [0.005, 0.01]
DEFAULT_SAM = (0.3, 0.005)


def collect_all_pairs() -> Dict[str, List[Dict]]:
    """
    收集所有方法的公共帧对，统一为
        {method: [{"key": (clip_full, camera), "gt_path": str, "load": callable}, ...]}
    """
    import gs_eval
    import tf_eval
    from common_frames import get_common_targets

    targets = get_common_targets()
    keep = {(t["clip"], t["distance"]) for t in targets}
    num_to_full = {t["clip_num"]: t["clip"] for t in targets}

    gs_eval.COMMON_MODE = True
    all_pairs = {}

    gs_pairs = gs_eval.collect_image_pairs()
    for method, pairs in gs_pairs.items():
        items = []
        for p in pairs:
            if (p["clip"], p["distance"]) not in keep:
                continue
            items.append({
                "key": (p["clip"], p["camera"]),
                "gt_path": p["gt_path"],
                "load": (lambda p=p: gs_eval.load_image_pair(
                    p["gen_path"], p["gt_path"], camera=p["camera"])),
            })
        if items:
            all_pairs[method] = items

    tf_pairs = tf_eval.collect_image_pairs(common=True)
    all_pairs["Ours"] = [{
        "key": (num_to_full[p["clip"]], p["camera"]),
        "gt_path": p["gt_path"],
        "load": (lambda p=p: tf_eval.load_pair(p)),
    } for p in tf_pairs]

    # 所有方法同一 (clip, 相机) 必须对应同一张 GT
    gt_of_key = {}
    for method, items in all_pairs.items():
        for it in items:
            prev = gt_of_key.setdefault(it["key"], it["gt_path"])
            if prev != it["gt_path"]:
                raise RuntimeError(f"{method} {it['key']} 的GT与其他方法不一致: "
                                   f"{it['gt_path']} vs {prev}")

    print("\n各方法帧对数:")
    for method, items in all_pairs.items():
        print(f"  {method:<16} {len(items)}")
    return all_pairs


def main():
    parser = argparse.ArgumentParser(description="指标参数敏感性分析")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="./results/sensitivity.json")
    args = parser.parse_args()

    from depth_eval import get_depth_estimator
    from seg_eval import get_segmentor
    from sam_eval import SAMSegmentorFast, compute_edge_consistency
    from metrics import (evaluate_depth_pair, compute_segmentation_metrics_multilevel,
                         compute_segmentation_consistency, aggregate_metrics)

    all_pairs = collect_all_pairs()

    depth_model = get_depth_estimator({'depth': {
        'model': 'depth_anything_v2', 'model_size': 'large', 'device': args.device}})
    seg_model = get_segmentor({'segmentation': {
        'model': 'mask2former', 'backbone': 'swin-l', 'dataset': 'cityscapes',
        'device': args.device}})
    sam = SAMSegmentorFast(model_size="large", device=args.device)

    gt_cache = {}

    def gt_outputs(key, gt_img):
        if key not in gt_cache:
            gt_cache[key] = {
                "depth": depth_model.predict(gt_img),
                "seg": seg_model.predict(gt_img).astype(np.uint8),
                "edge": sam.get_edge_map(gt_img),
            }
        return gt_cache[key]

    # per_frame[method][setting] = [逐帧指标dict]
    per_frame = defaultdict(lambda: defaultdict(list))

    for method, items in all_pairs.items():
        for it in tqdm(items, desc=f"  {method}"):
            gen_img, gt_img = it["load"]()
            gt = gt_outputs(it["key"], gt_img)

            d_gen = depth_model.predict(gen_img)
            for r in DEPTH_RATIOS:
                m = evaluate_depth_pair(d_gen, gt["depth"], max_depth_ratio=r,
                                        with_tolerant=False)
                per_frame[method][f"depth|ratio={r:g}"].append(
                    {"abs_rel": m["abs_rel"], "delta_1": m["delta_1"]})

            s_gen = seg_model.predict(gen_img)
            s_gt = gt["seg"].astype(np.int64)
            for ex in SEG_EXCLUDE_FLAT:
                m = compute_segmentation_metrics_multilevel(s_gen, s_gt, exclude_flat=ex)
                m.update(compute_segmentation_consistency(s_gen, s_gt, exclude_flat=ex))
                per_frame[method][f"seg|exclude_flat={ex}"].append(
                    {k: m[k] for k in ("consistency", "coarse_miou", "coarse_conf_matrix")})

            e_gen = sam.get_edge_map(gen_img)
            h, w = e_gen.shape[:2]
            diag = (h ** 2 + w ** 2) ** 0.5
            for th in SAM_THRESHOLDS:
                for frac in SAM_TOL_FRACS:
                    tol = max(3, int(round(diag * frac)))
                    m = compute_edge_consistency(e_gen, gt["edge"], tolerance=tol, threshold=th)
                    per_frame[method][f"sam|th={th:g},tol={frac:.1%}"].append(
                        {"edge_f1": m["edge_f1"]})

    # 汇总：results[setting][metric][method] = value
    results = defaultdict(lambda: defaultdict(dict))
    for method, settings in per_frame.items():
        for setting, frames in settings.items():
            agg = aggregate_metrics(frames)
            for metric in ("abs_rel", "delta_1", "consistency", "coarse_miou", "edge_f1"):
                if metric in agg:
                    results[setting][metric][method] = float(agg[metric])

    higher_better = {"abs_rel": False, "delta_1": True, "consistency": True,
                     "coarse_miou": True, "edge_f1": True}
    defaults = {
        "depth": f"depth|ratio={DEFAULT_DEPTH_RATIO:g}",
        "seg": f"seg|exclude_flat={DEFAULT_EXCLUDE_FLAT}",
        "sam": f"sam|th={DEFAULT_SAM[0]:g},tol={DEFAULT_SAM[1]:.1%}",
    }

    def ranking(values, metric):
        return sorted(values, key=values.get, reverse=higher_better[metric])

    methods = list(all_pairs.keys())
    summary = {}
    for family in ("depth", "seg", "sam"):
        settings = sorted(s for s in results if s.startswith(family + "|"))
        metrics = sorted({m for s in settings for m in results[s]})
        for metric in metrics:
            ref_rank = ranking(results[defaults[family]][metric], metric)
            print(f"\n=== {metric} ({'越高越好' if higher_better[metric] else '越低越好'}) ===")
            print(f"{'设置':<28}" + "".join(f"{m:>15}" for m in methods) + "   排名与默认一致")
            for s in settings:
                vals = results[s][metric]
                rank = ranking(vals, metric)
                same = "默认" if s == defaults[family] else ("✓" if rank == ref_rank else "✗")
                print(f"{s:<28}" + "".join(
                    f"{vals.get(m, float('nan')):>15.4f}" for m in methods) + f"   {same}")
                summary[f"{s}|{metric}"] = {"values": vals, "ranking": rank,
                                            "same_as_default": rank == ref_rank}
            print(f"  默认排名: {' > '.join(ref_rank)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {args.output}")


if __name__ == "__main__":
    main()
