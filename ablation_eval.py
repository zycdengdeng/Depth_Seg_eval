#!/usr/bin/env python3
"""
消融实验评测脚本

数据结构：
    /mnt/zihanw/inference/
      ablation_all_controls/031_seg01/generated.mp4 + ground_truth.mp4
      ablation_blur_only/033_seg01/generated.mp4 + ground_truth.mp4
      ...

指标与 gs_eval / tf_eval 一致：
    image_metrics (PSNR/SSIM/LPIPS), depth, seg, sam, nta, ntl, fid

使用：
    python ablation_eval.py --task all --device cuda:0
    python ablation_eval.py --task image_metrics --variants ablation_all_controls ablation_blur_only
    python ablation_eval.py --task fid --device cuda:1
"""

import os
import sys
import glob
import json
import tempfile
import shutil
import argparse
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# ============== 配置 ==============

INFERENCE_ROOT = "/mnt/zihanw/inference"

VARIANTS = [
    "ablation_all_controls",
    "ablation_blur_depth",
    "ablation_blur_only",
    "ablation_depth_only",
    "ablation_hdmap_blur",
    "ablation_hdmap_depth",
    "ablation_hdmap_only",
]

VARIANT_DISPLAY = {
    "ablation_all_controls": "All Controls",
    "ablation_blur_depth":   "Blur+Depth",
    "ablation_blur_only":    "Blur Only",
    "ablation_depth_only":   "Depth Only",
    "ablation_hdmap_blur":   "HDMap+Blur",
    "ablation_hdmap_depth":  "HDMap+Depth",
    "ablation_hdmap_only":   "HDMap Only",
}


# ============== 数据加载 ==============

def extract_frames(video_path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def discover_clips(variant_dir: str) -> List[str]:
    clips = []
    if not os.path.isdir(variant_dir):
        return clips
    for name in sorted(os.listdir(variant_dir)):
        gen_path = os.path.join(variant_dir, name, "generated.mp4")
        gt_path = os.path.join(variant_dir, name, "ground_truth.mp4")
        if os.path.exists(gen_path) and os.path.exists(gt_path):
            clips.append(name)
    return clips


def collect_pairs(variant: str, clips: List[str] = None,
                  root: str = INFERENCE_ROOT) -> List[Dict]:
    """收集所有帧对: [{gen_frame, gt_frame, clip, frame_idx}, ...]"""
    variant_dir = os.path.join(root, variant)
    available_clips = discover_clips(variant_dir)
    if clips:
        available_clips = [c for c in available_clips if c in clips]

    pairs = []
    for clip_name in available_clips:
        gen_path = os.path.join(variant_dir, clip_name, "generated.mp4")
        gt_path = os.path.join(variant_dir, clip_name, "ground_truth.mp4")

        gen_frames = extract_frames(gen_path)
        gt_frames = extract_frames(gt_path)

        n = min(len(gen_frames), len(gt_frames))
        for i in range(n):
            pairs.append({
                "gen_frame": gen_frames[i],
                "gt_frame": gt_frames[i],
                "clip": clip_name,
                "camera": "all",
                "frame_idx": i,
            })

    return pairs


def load_pair(pair: Dict) -> Tuple[np.ndarray, np.ndarray]:
    return pair["gen_frame"], pair["gt_frame"]


# ============== 评测任务（复用已有模块） ==============

def run_image_metrics(pairs: List[Dict], device: str = "cuda") -> List[Dict]:
    from image_metrics_eval import ImageMetricsEvaluator
    from tqdm import tqdm
    evaluator = ImageMetricsEvaluator(device=device)
    results = []
    for pair in tqdm(pairs, desc="    图像质量"):
        gen_img, gt_img = load_pair(pair)
        metrics = evaluator.evaluate_pair(gen_img, gt_img)
        metrics.update({"clip": pair["clip"], "frame_idx": pair["frame_idx"]})
        results.append(metrics)
    return results


def run_depth_eval(pairs: List[Dict], device: str = "cuda") -> List[Dict]:
    from depth_eval import get_depth_estimator
    from metrics import compute_depth_metrics, compute_depth_correlation
    from utils import align_depth_scale
    from tqdm import tqdm
    depth_config = {'depth': {'model': 'depth_anything_v2', 'model_size': 'large', 'device': device}}
    model = get_depth_estimator(depth_config)
    results = []
    for pair in tqdm(pairs, desc="    深度一致性"):
        gen_img, gt_img = load_pair(pair)
        d_gen = model.predict(gen_img)
        d_gt = model.predict(gt_img)
        d_gen = align_depth_scale(d_gen, d_gt, method="median")
        metrics = compute_depth_metrics(d_gen, d_gt)
        metrics.update(compute_depth_correlation(d_gen, d_gt))
        metrics.update({"clip": pair["clip"], "frame_idx": pair["frame_idx"]})
        results.append(metrics)
    return results


def run_seg_eval(pairs: List[Dict], device: str = "cuda") -> List[Dict]:
    from seg_eval import get_segmentor
    from metrics import compute_segmentation_metrics_multilevel, compute_segmentation_consistency
    from tqdm import tqdm
    seg_config = {'segmentation': {
        'model': 'mask2former', 'backbone': 'swin-l',
        'dataset': 'cityscapes', 'device': device,
        'class_merging': {'enabled': True},
    }}
    model = get_segmentor(seg_config)
    results = []
    for pair in tqdm(pairs, desc="    语义分割"):
        gen_img, gt_img = load_pair(pair)
        s_gen = model.predict(gen_img)
        s_gt = model.predict(gt_img)
        metrics = compute_segmentation_metrics_multilevel(s_gen, s_gt, compute_coarse=True)
        metrics.update(compute_segmentation_consistency(s_gen, s_gt))
        metrics.update({"clip": pair["clip"], "frame_idx": pair["frame_idx"]})
        results.append(metrics)
    return results


def run_sam_eval(pairs: List[Dict], device: str = "cuda") -> List[Dict]:
    from sam_eval import SAMSegmentorFast, compute_edge_consistency
    from tqdm import tqdm
    sam = SAMSegmentorFast(model_size="large", device=device)
    results = []
    for pair in tqdm(pairs, desc="    SAM边缘"):
        gen_img, gt_img = load_pair(pair)
        edge_gen = sam.get_edge_map(gen_img)
        edge_gt = sam.get_edge_map(gt_img)
        metrics = compute_edge_consistency(edge_gen, edge_gt, tolerance=3)
        metrics.update({"clip": pair["clip"], "frame_idx": pair["frame_idx"]})
        results.append(metrics)
    return results


def run_nta_eval(pairs: List[Dict], device: str = "cuda") -> List[Dict]:
    from nta_eval import YOLO11Detector, match_detections, TRAFFIC_AGENT_IDS
    from tqdm import tqdm
    detector = YOLO11Detector(model_size="l", device=device,
                              conf_threshold=0.5, detect_size=(480, 320))
    results = []
    for pair in tqdm(pairs, desc="    NTA-IoU"):
        gen_img, gt_img = load_pair(pair)
        dets_gen = detector.detect(gen_img, filter_classes=TRAFFIC_AGENT_IDS)
        dets_gt = detector.detect(gt_img, filter_classes=TRAFFIC_AGENT_IDS)
        metrics = match_detections(dets_gen, dets_gt, distance_threshold=50.0)
        metrics.update({"clip": pair["clip"], "frame_idx": pair["frame_idx"]})
        results.append(metrics)
    return results


def run_ntl_eval(pairs: List[Dict], device: str = "cuda") -> List[Dict]:
    from ntl_eval import TwinLiteNetDetector, compute_mask_iou, compute_mask_f1
    from tqdm import tqdm
    detector = TwinLiteNetDetector(device=device, input_size=(360, 640))
    results = []
    for pair in tqdm(pairs, desc="    NTL-IoU"):
        gen_img, gt_img = load_pair(pair)
        pred_gen = detector.predict(gen_img)
        pred_gt = detector.predict(gt_img)
        ntl_iou = compute_mask_iou(pred_gen['lane_mask'], pred_gt['lane_mask'])
        lane_f1 = compute_mask_f1(pred_gen['lane_mask'], pred_gt['lane_mask'], tolerance=3)
        da_iou = compute_mask_iou(pred_gen['da_mask'], pred_gt['da_mask'])
        metrics = {
            'ntl_iou': ntl_iou, 'ntl_f1': lane_f1['f1'],
            'ntl_precision': lane_f1['precision'], 'ntl_recall': lane_f1['recall'],
            'da_iou': da_iou,
            "clip": pair["clip"], "frame_idx": pair["frame_idx"],
        }
        results.append(metrics)
    return results


def run_fid_eval(pairs: List[Dict], device: str = "cuda") -> List[Dict]:
    from image_metrics_eval import compute_fid_for_camera
    from PIL import Image

    tmp_dir = tempfile.mkdtemp(prefix="ablation_fid_")
    gen_dir = os.path.join(tmp_dir, "gen")
    gt_dir = os.path.join(tmp_dir, "gt")
    os.makedirs(gen_dir)
    os.makedirs(gt_dir)

    try:
        for i, pair in enumerate(pairs):
            gen_img, gt_img = load_pair(pair)
            Image.fromarray(gen_img).save(os.path.join(gen_dir, f"{i:06d}.png"))
            Image.fromarray(gt_img).save(os.path.join(gt_dir, f"{i:06d}.png"))
        fid_score = compute_fid_for_camera(gen_dir, gt_dir, device=device)
        return [{"fid": fid_score, "clip": "all", "frame_idx": -1}]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ============== 汇总 ==============

TASK_RUNNERS = {
    "image_metrics": run_image_metrics,
    "depth": run_depth_eval,
    "seg": run_seg_eval,
    "sam": run_sam_eval,
    "nta": run_nta_eval,
    "ntl": run_ntl_eval,
    "fid": run_fid_eval,
}


def aggregate_results(raw_results: List[Dict]) -> Dict:
    from metrics import aggregate_metrics
    if not raw_results:
        return {}
    meta_keys = {"clip", "camera", "frame_idx"}
    extract = lambda recs: [{k: v for k, v in r.items() if k not in meta_keys} for r in recs]

    result = {"overall": aggregate_metrics(extract(raw_results))}

    by_clip = defaultdict(list)
    for r in raw_results:
        by_clip[r["clip"]].append(r)
    result["by_clip"] = {clip: aggregate_metrics(extract(recs))
                         for clip, recs in sorted(by_clip.items())}
    return result


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def print_comparison_table(all_variant_results: Dict, task: str):
    """横向对比表"""
    variants = list(all_variant_results.keys())
    if not variants:
        return
    sample = all_variant_results[variants[0]]
    if "overall" not in sample:
        return
    metric_keys = [k for k in sample["overall"].keys()
                   if not k.endswith('_std') and k != 'class_iou' and k != 'coarse_class_iou']

    print(f"\n{'=' * 80}")
    print(f"  {task} 消融对比")
    print(f"{'=' * 80}")
    header = f"{'Variant':<22}"
    for mk in metric_keys[:8]:
        header += f" {mk:>10}"
    print(header)
    print("-" * len(header))

    for variant in variants:
        display = VARIANT_DISPLAY.get(variant, variant)
        row = f"{display:<22}"
        overall = all_variant_results[variant].get("overall", {})
        for mk in metric_keys[:8]:
            val = overall.get(mk, float('nan'))
            if isinstance(val, float):
                row += f" {val:>10.4f}"
            else:
                row += f" {str(val):>10}"
        print(row)
    print("=" * len(header))


# ============== 主函数 ==============

def main():
    parser = argparse.ArgumentParser(description="消融实验评测")
    parser.add_argument("--task", type=str, default="all",
                        choices=["all"] + list(TASK_RUNNERS.keys()))
    parser.add_argument("--variants", nargs="+", default=None,
                        help="要评测的变体（默认全部）")
    parser.add_argument("--clips", nargs="+", default=None,
                        help="要评测的 clips（默认全部）")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--root", type=str, default=INFERENCE_ROOT)
    parser.add_argument("--output-dir", type=str, default="./results/ablation_eval")
    args = parser.parse_args()

    variants = args.variants or VARIANTS
    tasks = list(TASK_RUNNERS.keys()) if args.task == "all" else [args.task]

    print("=" * 70)
    print("消融实验评测")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"变体: {variants}")
    print(f"任务: {tasks}")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    for task in tasks:
        print(f"\n{'#' * 70}")
        print(f"# 任务: {task}")
        print(f"{'#' * 70}")

        task_results = {}
        runner = TASK_RUNNERS[task]

        for variant in variants:
            print(f"\n--- [{VARIANT_DISPLAY.get(variant, variant)}] ---")
            pairs = collect_pairs(variant, clips=args.clips, root=args.root)
            if not pairs:
                print(f"  没有找到有效帧对，跳过")
                continue
            print(f"  收集到 {len(pairs)} 个帧对")

            raw = runner(pairs, device=args.device)
            agg = aggregate_results(raw)
            task_results[variant] = agg

            overall = agg.get("overall", {})
            print(f"  结果: ", end="")
            for k, v in list(overall.items())[:5]:
                if not k.endswith('_std') and isinstance(v, float):
                    print(f"{k}={v:.4f}  ", end="")
            print()

        print_comparison_table(task_results, task)

        output_path = os.path.join(args.output_dir, f"{task}_results.json")
        with open(output_path, 'w') as f:
            json.dump(convert_to_serializable({
                "timestamp": datetime.now().isoformat(),
                "task": task,
                "results": task_results,
            }), f, indent=2)
        print(f"\n保存: {output_path}")

    print(f"\n完成！所有结果保存在 {args.output_dir}")


if __name__ == "__main__":
    main()
