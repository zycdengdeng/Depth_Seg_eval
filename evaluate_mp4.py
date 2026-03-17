#!/usr/bin/env python3
"""
MP4视频评测脚本

针对MP4视频格式的数据集进行评测，支持：
- 深度一致性评测 (Depth Anything V2)
- 语义分割一致性评测 (Mask2Former)
- SAM结构一致性评测
- 图像质量评测 (PSNR/SSIM/LPIPS/FID)
- FVD视频质量评测

使用方法：
    python evaluate_mp4.py --task all --parallel --gpus 0,1,2,3,4,5,6
    python evaluate_mp4.py --task depth
    python evaluate_mp4.py --task fvd
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from copy import deepcopy
from tqdm import tqdm
import traceback

from mp4_utils import (
    load_mp4_config, get_video_paths, get_segment_folders,
    iter_video_frame_pairs, read_video_frames, get_video_info,
    count_total_frames, ensure_dir
)
from metrics import aggregate_metrics, format_metrics_table
from evaluate import convert_to_serializable, generate_report


# ============== 深度评测 ==============

def evaluate_depth_mp4(config: Dict, camera: str, video_pairs: List[Tuple],
                       gpu_id: int = 0, save_vis: bool = True) -> Dict:
    """单相机深度评测"""
    torch.cuda.set_device(gpu_id)

    from depth_eval import get_depth_estimator
    from utils import align_depth_scale, save_depth_visualization
    from metrics import compute_depth_metrics, compute_depth_correlation

    worker_config = deepcopy(config)
    worker_config['depth']['device'] = f'cuda:{gpu_id}'
    estimator = get_depth_estimator(worker_config)

    frame_step = config.get('processing', {}).get('frame_step', 1)
    output_dir = os.path.join(config['output']['depth_maps'], camera)
    if save_vis:
        ensure_dir(output_dir)

    camera_metrics = []

    for gen_path, gt_path, segment in tqdm(video_pairs, desc=f"  [GPU:{gpu_id}] {camera}"):
        for gen_frame, gt_frame, frame_idx in iter_video_frame_pairs(gen_path, gt_path, frame_step):
            depth_gen = estimator.predict(gen_frame)
            depth_gt = estimator.predict(gt_frame)

            depth_gen_aligned = align_depth_scale(depth_gen, depth_gt, method="median")

            metrics = compute_depth_metrics(depth_gen_aligned, depth_gt)
            correlation = compute_depth_correlation(depth_gen_aligned, depth_gt)
            metrics.update(correlation)

            camera_metrics.append(metrics)

            if save_vis and frame_idx % 30 == 0:  # 每30帧保存一次
                save_depth_visualization(
                    depth_gen,
                    os.path.join(output_dir, f"{segment}_{frame_idx:04d}_gen.png")
                )

    return aggregate_metrics(camera_metrics)


# ============== 分割评测 ==============

def evaluate_seg_mp4(config: Dict, camera: str, video_pairs: List[Tuple],
                     gpu_id: int = 0, save_vis: bool = True) -> Dict:
    """单相机分割评测"""
    torch.cuda.set_device(gpu_id)

    from seg_eval import get_segmentor, apply_ego_vehicle_mask
    from utils import save_segmentation_visualization, get_cityscapes_palette
    from metrics import (compute_segmentation_metrics_multilevel,
                        compute_segmentation_consistency)

    worker_config = deepcopy(config)
    worker_config['segmentation']['device'] = f'cuda:{gpu_id}'
    segmentor = get_segmentor(worker_config)

    frame_step = config.get('processing', {}).get('frame_step', 1)
    output_dir = os.path.join(config['output']['seg_maps'], camera)
    if save_vis:
        ensure_dir(output_dir)

    palette = get_cityscapes_palette()
    camera_metrics = []

    for gen_path, gt_path, segment in tqdm(video_pairs, desc=f"  [GPU:{gpu_id}] {camera}"):
        for gen_frame, gt_frame, frame_idx in iter_video_frame_pairs(gen_path, gt_path, frame_step):
            seg_gen = segmentor.predict(gen_frame)
            seg_gt = segmentor.predict(gt_frame)

            seg_gen = apply_ego_vehicle_mask(seg_gen, camera, config)
            seg_gt = apply_ego_vehicle_mask(seg_gt, camera, config)

            consistency = compute_segmentation_consistency(seg_gen, seg_gt)
            compute_coarse = config.get('segmentation', {}).get('class_merging', {}).get('enabled', True)
            seg_metrics = compute_segmentation_metrics_multilevel(
                seg_gen, seg_gt, num_classes=segmentor.num_classes,
                compute_coarse=compute_coarse
            )

            metrics = {**consistency, **seg_metrics}
            camera_metrics.append(metrics)

            if save_vis and frame_idx % 30 == 0:
                save_segmentation_visualization(
                    seg_gen,
                    os.path.join(output_dir, f"{segment}_{frame_idx:04d}_gen.png"),
                    palette=palette
                )

    return aggregate_metrics(camera_metrics)


# ============== SAM评测 ==============

def evaluate_sam_mp4(config: Dict, camera: str, video_pairs: List[Tuple],
                     gpu_id: int = 0, save_vis: bool = True) -> Dict:
    """单相机SAM评测"""
    torch.cuda.set_device(gpu_id)

    from sam_eval import SAMSegmentorFast, compute_edge_consistency
    from PIL import Image as PILImage

    sam_config = config.get('sam', {})
    model_size = sam_config.get('model_size', 'large')
    sam = SAMSegmentorFast(model_size=model_size, device=f'cuda:{gpu_id}')

    frame_step = config.get('processing', {}).get('frame_step', 1)
    output_dir = os.path.join(config['output']['root'], 'sam_maps', camera)
    if save_vis:
        ensure_dir(output_dir)

    camera_metrics = []

    for gen_path, gt_path, segment in tqdm(video_pairs, desc=f"  [GPU:{gpu_id}] {camera}"):
        for gen_frame, gt_frame, frame_idx in iter_video_frame_pairs(gen_path, gt_path, frame_step):
            edge_gen = sam.get_edge_map(gen_frame)
            edge_gt = sam.get_edge_map(gt_frame)

            metrics = compute_edge_consistency(edge_gen, edge_gt, tolerance=3)
            camera_metrics.append(metrics)

            if save_vis and frame_idx % 30 == 0:
                PILImage.fromarray((edge_gen * 255).astype(np.uint8)).save(
                    os.path.join(output_dir, f"{segment}_{frame_idx:04d}_gen.png"))

    return aggregate_metrics(camera_metrics)


# ============== 图像质量评测 ==============

def evaluate_image_metrics_mp4(config: Dict, camera: str, video_pairs: List[Tuple],
                               gpu_id: int = 0, save_vis: bool = False) -> Dict:
    """单相机图像质量评测 (PSNR/SSIM/LPIPS)"""
    torch.cuda.set_device(gpu_id)

    from image_metrics_eval import ImageMetricsEvaluator

    evaluator = ImageMetricsEvaluator(device=f'cuda:{gpu_id}')
    frame_step = config.get('processing', {}).get('frame_step', 1)

    camera_metrics = []

    for gen_path, gt_path, segment in tqdm(video_pairs, desc=f"  [GPU:{gpu_id}] {camera}"):
        for gen_frame, gt_frame, frame_idx in iter_video_frame_pairs(gen_path, gt_path, frame_step):
            metrics = evaluator.evaluate_pair(gen_frame, gt_frame)
            camera_metrics.append(metrics)

    return aggregate_metrics(camera_metrics)


# ============== FVD评测 ==============

def evaluate_fvd_mp4(config: Dict) -> Dict:
    """FVD评测（需要完整视频片段）"""
    from fvd_eval import I3DFeatureExtractor, compute_fvd

    fvd_config = config.get('fvd', {})
    clip_length = fvd_config.get('clip_length', 16)
    resolution = fvd_config.get('resolution', 224)
    device = 'cuda:0'

    video_pairs = get_video_paths(config)
    extractor = I3DFeatureExtractor(device=device)

    results = {}

    for camera, pairs in video_pairs.items():
        print(f"\n处理相机: {camera} ({len(pairs)} 个视频)")

        gen_features_list = []
        gt_features_list = []

        for gen_path, gt_path, segment in tqdm(pairs, desc=f"  {camera}"):
            # 读取完整视频
            gen_frames = read_video_frames(gen_path)
            gt_frames = read_video_frames(gt_path)

            # 切分成clips
            num_clips = min(len(gen_frames), len(gt_frames)) // clip_length

            for i in range(num_clips):
                start = i * clip_length
                end = start + clip_length

                # 构建clip tensor
                gen_clip = []
                gt_clip = []
                for j in range(start, end):
                    from PIL import Image
                    # Resize to resolution
                    gen_img = Image.fromarray(gen_frames[j]).resize((resolution, resolution))
                    gt_img = Image.fromarray(gt_frames[j]).resize((resolution, resolution))

                    gen_clip.append(torch.from_numpy(np.array(gen_img)).float() / 255.0)
                    gt_clip.append(torch.from_numpy(np.array(gt_img)).float() / 255.0)

                gen_clip = torch.stack(gen_clip).permute(0, 3, 1, 2).unsqueeze(0)  # (1, T, C, H, W)
                gt_clip = torch.stack(gt_clip).permute(0, 3, 1, 2).unsqueeze(0)

                # 提取特征
                gen_feat = extractor.extract_features(gen_clip)
                gt_feat = extractor.extract_features(gt_clip)

                gen_features_list.append(gen_feat)
                gt_features_list.append(gt_feat)

        if gen_features_list:
            gen_features = np.concatenate(gen_features_list, axis=0)
            gt_features = np.concatenate(gt_features_list, axis=0)

            fvd_score = compute_fvd(gen_features, gt_features)

            results[camera] = {
                'fvd': fvd_score,
                'num_clips': len(gen_features_list),
                'clip_length': clip_length,
            }
            print(f"  {camera} FVD = {fvd_score:.2f} ({len(gen_features_list)} clips)")

    # 计算总体FVD
    all_fvds = [r['fvd'] for r in results.values() if 'fvd' in r]
    if all_fvds:
        results['overall'] = {
            'fvd': np.mean(all_fvds),
            'fvd_std': np.std(all_fvds),
            'num_cameras': len(all_fvds),
        }

    return results


# ============== 并行评测 ==============

def _worker_wrapper(task_fn, camera, video_pairs, config, gpu_id, save_vis, result_dict):
    """Worker包装器"""
    try:
        result = task_fn(config, camera, video_pairs, gpu_id, save_vis)
        result_dict[camera] = result
        print(f"\n  [GPU:{gpu_id}] {camera} 完成!")
    except Exception as e:
        print(f"\n  [GPU:{gpu_id}] {camera} 出错: {e}")
        traceback.print_exc()
        result_dict[camera] = {}


def evaluate_parallel_mp4(config: Dict, task: str,
                          gpu_ids: Optional[List[int]] = None,
                          save_vis: bool = True) -> Dict:
    """多GPU并行评测"""
    if gpu_ids is None:
        num_gpus = torch.cuda.device_count()
        gpu_ids = list(range(num_gpus))

    print(f"\n可用GPU: {gpu_ids}")

    video_pairs = get_video_paths(config)
    if not video_pairs:
        raise ValueError("没有找到有效的视频对")

    cameras = list(video_pairs.keys())
    print(f"相机数量: {len(cameras)}, GPU数量: {len(gpu_ids)}")

    # 分配相机到GPU
    camera_gpu_map = {}
    for i, camera in enumerate(cameras):
        camera_gpu_map[camera] = gpu_ids[i % len(gpu_ids)]

    print("GPU分配:")
    for camera, gpu_id in camera_gpu_map.items():
        print(f"  {camera} -> GPU:{gpu_id}")

    # 选择任务函数
    task_fn = {
        'depth': evaluate_depth_mp4,
        'seg': evaluate_seg_mp4,
        'segmentation': evaluate_seg_mp4,
        'sam': evaluate_sam_mp4,
        'image_metrics': evaluate_image_metrics_mp4,
    }.get(task)

    if task_fn is None:
        raise ValueError(f"不支持的并行任务: {task}")

    # 预加载模型
    _preload_models(task, config)

    # 创建输出目录
    ensure_dir(config['output']['metrics'])
    if task == 'depth':
        ensure_dir(config['output']['depth_maps'])
    elif task in ['seg', 'segmentation']:
        ensure_dir(config['output']['seg_maps'])
    elif task == 'sam':
        ensure_dir(os.path.join(config['output']['root'], 'sam_maps'))

    # 并行处理
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    result_dict = manager.dict()

    processes = []
    for camera in cameras:
        gpu_id = camera_gpu_map[camera]
        p = mp.Process(
            target=_worker_wrapper,
            args=(task_fn, camera, video_pairs[camera], config,
                  gpu_id, save_vis, result_dict)
        )
        processes.append(p)

    # 依次启动，间隔2秒
    import time
    for i, p in enumerate(processes):
        p.start()
        if i < len(processes) - 1:
            time.sleep(2)

    for p in processes:
        p.join()

    results = dict(result_dict)

    # 计算总体平均
    all_metrics = []
    for camera_results in results.values():
        if camera_results:
            filtered = {k: v for k, v in camera_results.items()
                       if not k.endswith('_std') and not isinstance(v, list)}
            all_metrics.append(filtered)

    if all_metrics:
        results['overall'] = aggregate_metrics(all_metrics)

    return results


def _preload_models(task: str, config: Dict):
    """预加载模型到缓存"""
    print("预下载模型到缓存...")

    if task == 'depth':
        from transformers import AutoModelForDepthEstimation, AutoImageProcessor
        model_name = "depth-anything/Depth-Anything-V2-Large-hf"
        print(f"  预加载 Depth Anything V2: {model_name}")
        AutoImageProcessor.from_pretrained(model_name)
        AutoModelForDepthEstimation.from_pretrained(model_name)

    elif task in ['seg', 'segmentation']:
        from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor
        model_name = "facebook/mask2former-swin-large-cityscapes-semantic"
        print(f"  预加载 Mask2Former: {model_name}")
        AutoImageProcessor.from_pretrained(model_name)
        Mask2FormerForUniversalSegmentation.from_pretrained(model_name)

    elif task == 'sam':
        from transformers import SamModel, SamProcessor
        model_name = "facebook/sam-vit-large"
        print(f"  预加载 SAM: {model_name}")
        SamProcessor.from_pretrained(model_name)
        SamModel.from_pretrained(model_name)

    elif task == 'image_metrics':
        import lpips
        print("  预加载 LPIPS (AlexNet)...")
        _ = lpips.LPIPS(net='alex')

    print("模型缓存就绪!\n")


# ============== 主函数 ==============

def main():
    parser = argparse.ArgumentParser(description="MP4视频评测")
    parser.add_argument("--config", type=str, default="config_mp4.yaml",
                       help="配置文件路径")
    parser.add_argument("--task", type=str, default="all",
                       choices=["all", "depth", "seg", "segmentation", "sam", "image_metrics", "fvd"],
                       help="评测任务")
    parser.add_argument("--no-vis", action="store_true",
                       help="不保存可视化结果")
    parser.add_argument("--parallel", action="store_true",
                       help="启用多GPU并行评测")
    parser.add_argument("--gpus", type=str, default=None,
                       help="指定GPU ID，逗号分隔")

    args = parser.parse_args()

    print(f"加载配置文件: {args.config}")
    config = load_mp4_config(args.config)

    # 解析GPU
    gpu_ids = None
    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]

    # 确保输出目录存在
    ensure_dir(config['output']['metrics'])

    # 打印数据集信息
    print("\n" + "=" * 60)
    print("数据集信息")
    print("=" * 60)
    segments = get_segment_folders(config)
    print(f"数据段数量: {len(segments)}")
    frame_counts = count_total_frames(config)
    total_frames = sum(frame_counts.values())
    print(f"总帧数: {total_frames}")
    print(f"输出目录: {config['output']['root']}")

    # 运行评测
    depth_results = None
    seg_results = None
    sam_results = None
    image_metrics_results = None
    fvd_results = None
    save_vis = not args.no_vis

    def _run_task(task_name, serial_fn=None):
        """运行单个任务"""
        label = {
            'depth': '深度一致性', 'seg': '语义分割一致性',
            'sam': 'SAM结构一致性', 'image_metrics': '图像质量',
            'fvd': 'FVD视频距离'
        }
        print(f"\n{'=' * 70}")
        print(f"开始{label.get(task_name, task_name)}评测...")
        print("=" * 70)

        try:
            if args.parallel and task_name != 'fvd':
                return evaluate_parallel_mp4(config, task=task_name,
                                            gpu_ids=gpu_ids, save_vis=save_vis)
            elif serial_fn:
                return serial_fn(config)
            else:
                # 串行单GPU
                video_pairs = get_video_paths(config)
                task_fn = {
                    'depth': evaluate_depth_mp4,
                    'seg': evaluate_seg_mp4,
                    'sam': evaluate_sam_mp4,
                    'image_metrics': evaluate_image_metrics_mp4,
                }[task_name]

                results = {}
                for camera, pairs in video_pairs.items():
                    results[camera] = task_fn(config, camera, pairs, 0, save_vis)

                all_metrics = [v for v in results.values() if v]
                if all_metrics:
                    results['overall'] = aggregate_metrics(all_metrics)

                return results

        except Exception as e:
            print(f"{label.get(task_name, task_name)}评测出错: {e}")
            traceback.print_exc()
            return None

    # 执行各任务
    if args.task in ["all", "depth"]:
        depth_results = _run_task("depth")
        if depth_results:
            output_path = os.path.join(config['output']['metrics'], 'depth_results.json')
            with open(output_path, 'w') as f:
                json.dump(convert_to_serializable(depth_results), f, indent=2)
            print(f"深度评测结果已保存到: {output_path}")

    if args.task in ["all", "seg", "segmentation"]:
        seg_results = _run_task("seg")
        if seg_results:
            output_path = os.path.join(config['output']['metrics'], 'seg_results.json')
            with open(output_path, 'w') as f:
                json.dump(convert_to_serializable(seg_results), f, indent=2)
            print(f"分割评测结果已保存到: {output_path}")

    if args.task in ["all", "sam"]:
        sam_results = _run_task("sam")
        if sam_results:
            output_path = os.path.join(config['output']['metrics'], 'sam_results.json')
            with open(output_path, 'w') as f:
                json.dump(convert_to_serializable(sam_results), f, indent=2)
            print(f"SAM评测结果已保存到: {output_path}")

    if args.task in ["all", "image_metrics"]:
        image_metrics_results = _run_task("image_metrics")
        if image_metrics_results:
            output_path = os.path.join(config['output']['metrics'], 'image_metrics_results.json')
            with open(output_path, 'w') as f:
                json.dump(convert_to_serializable(image_metrics_results), f, indent=2)
            print(f"图像质量评测结果已保存到: {output_path}")

    if args.task in ["all", "fvd"]:
        print(f"\n{'=' * 70}")
        print("开始FVD(视频Fréchet距离)评测...")
        print("  模式: 串行 (I3D要求cuda:0)")
        print("=" * 70)
        try:
            fvd_results = evaluate_fvd_mp4(config)
        except Exception as e:
            print(f"FVD评测出错: {e}")
            traceback.print_exc()
            fvd_results = None

        if fvd_results:
            output_path = os.path.join(config['output']['metrics'], 'fvd_results.json')
            with open(output_path, 'w') as f:
                json.dump(convert_to_serializable(fvd_results), f, indent=2)
            print(f"FVD评测结果已保存到: {output_path}")

    # 生成综合报告（自动加载已有结果）
    metrics_dir = config['output']['metrics']

    def _load_existing(filename, current_result):
        if current_result is not None:
            return current_result
        path = os.path.join(metrics_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                print(f"  已加载历史结果: {filename}")
                return data
            except Exception:
                pass
        return None

    depth_results = _load_existing('depth_results.json', depth_results)
    seg_results = _load_existing('seg_results.json', seg_results)
    sam_results = _load_existing('sam_results.json', sam_results)
    image_metrics_results = _load_existing('image_metrics_results.json', image_metrics_results)
    fvd_results = _load_existing('fvd_results.json', fvd_results)

    if depth_results or seg_results or sam_results or image_metrics_results or fvd_results:
        report_path = os.path.join(metrics_dir, 'evaluation_report.txt')
        generate_report(depth_results, seg_results, report_path,
                       sam_results=sam_results,
                       image_metrics_results=image_metrics_results,
                       fvd_results=fvd_results)
        print(f"\n综合报告已保存到: {report_path}")

        full_results = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'depth': depth_results,
            'segmentation': seg_results,
            'sam': sam_results,
            'image_metrics': image_metrics_results,
            'fvd': fvd_results
        }
        full_output = os.path.join(metrics_dir, 'full_results.json')
        with open(full_output, 'w') as f:
            json.dump(convert_to_serializable(full_results), f, indent=2)
        print(f"完整结果已保存到: {full_output}")


if __name__ == "__main__":
    main()
