"""
多GPU并行评测封装

将不同相机分配到不同GPU上并行处理，大幅加速评测。
"""

import os
import torch
import torch.multiprocessing as mp
import numpy as np
from typing import Dict, List, Tuple, Optional
from copy import deepcopy
import json
import traceback

from utils import load_config, get_image_pairs, ensure_dir
from metrics import aggregate_metrics, format_metrics_table


def _worker_depth(camera: str, pairs: List[Tuple[str, str]],
                  config: Dict, gpu_id: int, save_vis: bool,
                  result_dict: dict):
    """单GPU深度评测worker"""
    try:
        import torch
        torch.cuda.set_device(gpu_id)

        from depth_eval import get_depth_estimator
        from utils import (load_image, align_depth_scale, align_spatial,
                          compute_tolerant_metrics, save_depth_visualization, ensure_dir)
        from metrics import (compute_depth_metrics, compute_depth_correlation,
                            compute_depth_ssim)
        from tqdm import tqdm

        # 创建该GPU上的estimator
        worker_config = deepcopy(config)
        worker_config['depth']['device'] = f'cuda:{gpu_id}'
        estimator = get_depth_estimator(worker_config)

        output_dir = config['output']['depth_maps']
        camera_out_dir = os.path.join(output_dir, camera)
        if save_vis:
            ensure_dir(camera_out_dir)

        camera_metrics = []

        for gen_path, gt_path in tqdm(pairs, desc=f"  [GPU:{gpu_id}] {camera}"):
            filename = os.path.basename(gen_path).replace('.png', '')

            gen_img = load_image(gen_path)
            gt_img = load_image(gt_path)

            depth_gen = estimator.predict(gen_img)
            depth_gt = estimator.predict(gt_img)

            depth_gen_aligned = align_depth_scale(depth_gen, depth_gt, method="median")

            # 标准指标
            metrics = compute_depth_metrics(depth_gen_aligned, depth_gt)
            correlation = compute_depth_correlation(depth_gen_aligned, depth_gt)
            metrics.update(correlation)

            # 容差指标
            depth_gen_spatial, (dy, dx) = align_spatial(
                depth_gen_aligned, depth_gt, max_shift=20
            )
            aligned_metrics = compute_depth_metrics(depth_gen_spatial, depth_gt)
            metrics['aligned_abs_rel'] = aligned_metrics['abs_rel']
            metrics['aligned_delta_1'] = aligned_metrics['delta_1']
            metrics['aligned_rmse'] = aligned_metrics['rmse']
            metrics['shift_dx'] = float(dx)
            metrics['shift_dy'] = float(dy)

            best_match_gt = compute_tolerant_metrics(
                depth_gen_aligned, depth_gt, window_size=5
            )
            tolerant_metrics = compute_depth_metrics(depth_gen_aligned, best_match_gt)
            metrics['tolerant_abs_rel'] = tolerant_metrics['abs_rel']
            metrics['tolerant_delta_1'] = tolerant_metrics['delta_1']
            metrics['tolerant_rmse'] = tolerant_metrics['rmse']

            ssim = compute_depth_ssim(depth_gen_aligned, depth_gt)
            metrics.update(ssim)

            # 转换为Python原生类型
            metrics = {k: float(v) if isinstance(v, (np.floating,)) else v
                      for k, v in metrics.items()}

            camera_metrics.append(metrics)

            if save_vis:
                save_depth_visualization(
                    depth_gen,
                    os.path.join(camera_out_dir, f"{filename}_gen_depth.png")
                )
                save_depth_visualization(
                    depth_gt,
                    os.path.join(camera_out_dir, f"{filename}_gt_depth.png")
                )

        result_dict[camera] = aggregate_metrics(camera_metrics)
        print(f"\n  [GPU:{gpu_id}] {camera} 完成!")

    except Exception as e:
        print(f"\n  [GPU:{gpu_id}] {camera} 出错: {e}")
        traceback.print_exc()
        result_dict[camera] = {}


def _worker_seg(camera: str, pairs: List[Tuple[str, str]],
                config: Dict, gpu_id: int, save_vis: bool,
                result_dict: dict):
    """单GPU分割评测worker"""
    try:
        import torch
        torch.cuda.set_device(gpu_id)

        from seg_eval import get_segmentor, apply_ego_vehicle_mask
        from utils import (load_image, save_segmentation_visualization,
                          get_cityscapes_palette, ensure_dir)
        from metrics import (compute_segmentation_metrics_multilevel,
                            compute_segmentation_consistency)
        from tqdm import tqdm

        worker_config = deepcopy(config)
        worker_config['segmentation']['device'] = f'cuda:{gpu_id}'
        segmentor = get_segmentor(worker_config)

        output_dir = config['output']['seg_maps']
        camera_out_dir = os.path.join(output_dir, camera)
        if save_vis:
            ensure_dir(camera_out_dir)

        palette = get_cityscapes_palette()
        camera_metrics = []

        for gen_path, gt_path in tqdm(pairs, desc=f"  [GPU:{gpu_id}] {camera}"):
            filename = os.path.basename(gen_path).replace('.png', '')

            gen_img = load_image(gen_path)
            gt_img = load_image(gt_path)

            seg_gen = segmentor.predict(gen_img)
            seg_gt = segmentor.predict(gt_img)

            # 应用自车车身mask处理（边缘区域 building -> car）
            seg_gen = apply_ego_vehicle_mask(seg_gen, camera, config)
            seg_gt = apply_ego_vehicle_mask(seg_gt, camera, config)

            consistency = compute_segmentation_consistency(seg_gen, seg_gt)
            compute_coarse = worker_config.get('segmentation', {}).get('class_merging', {}).get('enabled', True)
            seg_metrics = compute_segmentation_metrics_multilevel(
                seg_gen, seg_gt, num_classes=segmentor.num_classes,
                compute_coarse=compute_coarse
            )
            metrics = {**consistency, **seg_metrics}
            camera_metrics.append(metrics)

            if save_vis:
                save_segmentation_visualization(
                    seg_gen,
                    os.path.join(camera_out_dir, f"{filename}_gen_seg.png"),
                    palette=palette
                )
                save_segmentation_visualization(
                    seg_gt,
                    os.path.join(camera_out_dir, f"{filename}_gt_seg.png"),
                    palette=palette
                )

        result_dict[camera] = aggregate_metrics(camera_metrics)
        print(f"\n  [GPU:{gpu_id}] {camera} 完成!")

    except Exception as e:
        print(f"\n  [GPU:{gpu_id}] {camera} 出错: {e}")
        traceback.print_exc()
        result_dict[camera] = {}


def _worker_sam(camera: str, pairs: List[Tuple[str, str]],
                config: Dict, gpu_id: int, save_vis: bool,
                result_dict: dict):
    """单GPU SAM评测worker"""
    try:
        import torch
        torch.cuda.set_device(gpu_id)

        from sam_eval import SAMSegmentorFast, compute_edge_consistency
        from utils import load_image, ensure_dir
        from tqdm import tqdm
        from PIL import Image as PILImage

        sam_config = config.get('sam', {})
        model_size = sam_config.get('model_size', 'large')
        sam = SAMSegmentorFast(model_size=model_size, device=f'cuda:{gpu_id}')

        output_dir = os.path.join(config['output'].get('root', './results'), 'sam_maps')
        camera_out_dir = os.path.join(output_dir, camera)
        if save_vis:
            ensure_dir(camera_out_dir)

        camera_metrics = []

        for gen_path, gt_path in tqdm(pairs, desc=f"  [GPU:{gpu_id}] {camera}"):
            filename = os.path.basename(gen_path).replace('.png', '')

            gen_img = load_image(gen_path)
            gt_img = load_image(gt_path)

            edge_gen = sam.get_edge_map(gen_img)
            edge_gt = sam.get_edge_map(gt_img)

            metrics = compute_edge_consistency(edge_gen, edge_gt, tolerance=3)
            camera_metrics.append(metrics)

            if save_vis:
                PILImage.fromarray((edge_gen * 255).astype(np.uint8)).save(
                    os.path.join(camera_out_dir, f"{filename}_gen_edge.png"))
                PILImage.fromarray((edge_gt * 255).astype(np.uint8)).save(
                    os.path.join(camera_out_dir, f"{filename}_gt_edge.png"))

        result_dict[camera] = aggregate_metrics(camera_metrics)
        print(f"\n  [GPU:{gpu_id}] {camera} 完成!")

    except Exception as e:
        print(f"\n  [GPU:{gpu_id}] {camera} 出错: {e}")
        traceback.print_exc()
        result_dict[camera] = {}


def _worker_image_metrics(camera: str, pairs: List[Tuple[str, str]],
                          config: Dict, gpu_id: int, save_vis: bool,
                          result_dict: dict):
    """单GPU图像质量指标worker (PSNR/SSIM/LPIPS)，FID在主进程计算"""
    try:
        import torch
        torch.cuda.set_device(gpu_id)

        from image_metrics_eval import ImageMetricsEvaluator
        from utils import load_image
        from tqdm import tqdm

        device = f'cuda:{gpu_id}'
        evaluator = ImageMetricsEvaluator(device=device)

        camera_metrics = []

        for gen_path, gt_path in tqdm(pairs, desc=f"  [GPU:{gpu_id}] {camera}"):
            gen_img = load_image(gen_path)
            gt_img = load_image(gt_path)

            metrics = evaluator.evaluate_pair(gen_img, gt_img)
            camera_metrics.append(metrics)

        result_dict[camera] = aggregate_metrics(camera_metrics)
        print(f"\n  [GPU:{gpu_id}] {camera} 完成!")

    except Exception as e:
        print(f"\n  [GPU:{gpu_id}] {camera} 出错: {e}")
        traceback.print_exc()
        result_dict[camera] = {}


def _worker_nta(camera: str, pairs: List[Tuple[str, str]],
                config: Dict, gpu_id: int, save_vis: bool,
                result_dict: dict):
    """单GPU NTA-IoU评测worker"""
    try:
        import torch
        torch.cuda.set_device(gpu_id)

        from nta_eval import (get_nta_detector, match_detections,
                              compute_nta_per_class, TRAFFIC_AGENT_IDS,
                              _save_detection_vis)
        from utils import load_image, ensure_dir
        from tqdm import tqdm

        worker_config = deepcopy(config)
        worker_config['nta']['device'] = f'cuda:{gpu_id}'
        detector = get_nta_detector(worker_config)

        nta_config = config.get('nta', {})
        match_iou_threshold = nta_config.get('match_iou_threshold', 0.5)
        compute_per_class = nta_config.get('per_class', True)

        output_dir = os.path.join(config['output'].get('root', './results'), 'nta_maps')
        camera_out_dir = os.path.join(output_dir, camera)
        if save_vis:
            ensure_dir(camera_out_dir)

        camera_metrics = []

        for gen_path, gt_path in tqdm(pairs, desc=f"  [GPU:{gpu_id}] {camera}"):
            filename = os.path.basename(gen_path).replace('.png', '')

            gen_img = load_image(gen_path)
            gt_img = load_image(gt_path)

            dets_gen = detector.detect(gen_img, filter_classes=TRAFFIC_AGENT_IDS)
            dets_gt = detector.detect(gt_img, filter_classes=TRAFFIC_AGENT_IDS)

            metrics = match_detections(dets_gen, dets_gt, match_iou_threshold)
            if compute_per_class:
                per_class = compute_nta_per_class(dets_gen, dets_gt, match_iou_threshold)
                metrics.update(per_class)

            camera_metrics.append(metrics)

            if save_vis:
                _save_detection_vis(gen_img, dets_gen,
                    os.path.join(camera_out_dir, f"{filename}_gen_det.png"))
                _save_detection_vis(gt_img, dets_gt,
                    os.path.join(camera_out_dir, f"{filename}_gt_det.png"))

        result_dict[camera] = aggregate_metrics(camera_metrics)
        print(f"\n  [GPU:{gpu_id}] {camera} 完成!")

    except Exception as e:
        print(f"\n  [GPU:{gpu_id}] {camera} 出错: {e}")
        traceback.print_exc()
        result_dict[camera] = {}


def _worker_ntl(camera: str, pairs: List[Tuple[str, str]],
                config: Dict, gpu_id: int, save_vis: bool,
                result_dict: dict):
    """单GPU NTL-IoU评测worker"""
    try:
        import torch
        torch.cuda.set_device(gpu_id)

        from ntl_eval import (get_ntl_detector, compute_mask_iou,
                              compute_mask_f1, _save_lane_vis)
        from utils import load_image, ensure_dir
        from tqdm import tqdm

        worker_config = deepcopy(config)
        worker_config['ntl']['device'] = f'cuda:{gpu_id}'
        detector = get_ntl_detector(worker_config)

        ntl_config = config.get('ntl', {})
        lane_tolerance = ntl_config.get('lane_tolerance', 3)
        compute_da = ntl_config.get('compute_drivable_area', True)

        output_dir = os.path.join(config['output'].get('root', './results'), 'ntl_maps')
        camera_out_dir = os.path.join(output_dir, camera)
        if save_vis:
            ensure_dir(camera_out_dir)

        camera_metrics = []

        for gen_path, gt_path in tqdm(pairs, desc=f"  [GPU:{gpu_id}] {camera}"):
            filename = os.path.basename(gen_path).replace('.png', '')

            gen_img = load_image(gen_path)
            gt_img = load_image(gt_path)

            pred_gen = detector.predict(gen_img)
            pred_gt = detector.predict(gt_img)

            ntl_iou = compute_mask_iou(pred_gen['lane_mask'], pred_gt['lane_mask'])
            lane_f1 = compute_mask_f1(
                pred_gen['lane_mask'], pred_gt['lane_mask'],
                tolerance=lane_tolerance
            )

            metrics = {
                'ntl_iou': ntl_iou,
                'ntl_precision': lane_f1['precision'],
                'ntl_recall': lane_f1['recall'],
                'ntl_f1': lane_f1['f1'],
            }

            if compute_da:
                da_iou = compute_mask_iou(pred_gen['da_mask'], pred_gt['da_mask'])
                metrics['da_iou'] = da_iou

            camera_metrics.append(metrics)

            if save_vis:
                _save_lane_vis(gen_img, pred_gen,
                    os.path.join(camera_out_dir, f"{filename}_gen_lane.png"))
                _save_lane_vis(gt_img, pred_gt,
                    os.path.join(camera_out_dir, f"{filename}_gt_lane.png"))

        result_dict[camera] = aggregate_metrics(camera_metrics)
        print(f"\n  [GPU:{gpu_id}] {camera} 完成!")

    except Exception as e:
        print(f"\n  [GPU:{gpu_id}] {camera} 出错: {e}")
        traceback.print_exc()
        result_dict[camera] = {}


def _preload_models(task: str, config: Dict):
    """
    在主进程中预下载模型到缓存，避免多进程同时下载导致冲突

    Args:
        task: 评测任务类型
        config: 配置字典
    """
    print("预下载模型到缓存...")

    if task == 'depth':
        from transformers import AutoModelForDepthEstimation, AutoImageProcessor
        model_mapping = {
            "small": "depth-anything/Depth-Anything-V2-Small-hf",
            "base": "depth-anything/Depth-Anything-V2-Base-hf",
            "large": "depth-anything/Depth-Anything-V2-Large-hf",
        }
        model_size = config.get('depth', {}).get('model_size', 'large')
        model_name = model_mapping.get(model_size, model_mapping["large"])
        print(f"  预加载 Depth Anything V2: {model_name}")
        AutoImageProcessor.from_pretrained(model_name, use_fast=False)
        AutoModelForDepthEstimation.from_pretrained(model_name)

    elif task in ['seg', 'segmentation']:
        from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor
        model_name = "facebook/mask2former-swin-large-cityscapes-semantic"
        print(f"  预加载 Mask2Former: {model_name}")
        AutoImageProcessor.from_pretrained(model_name, use_fast=False)
        Mask2FormerForUniversalSegmentation.from_pretrained(model_name)

    elif task == 'sam':
        from transformers import SamModel, SamProcessor
        model_mapping = {
            "base": "facebook/sam-vit-base",
            "large": "facebook/sam-vit-large",
            "huge": "facebook/sam-vit-huge",
        }
        model_size = config.get('sam', {}).get('model_size', 'large')
        model_name = model_mapping.get(model_size, model_mapping["large"])
        print(f"  预加载 SAM: {model_name}")
        SamProcessor.from_pretrained(model_name, use_fast=False)
        SamModel.from_pretrained(model_name)

    elif task == 'image_metrics':
        import lpips
        print("  预加载 LPIPS (AlexNet)...")
        _ = lpips.LPIPS(net='alex')
        print("  LPIPS 缓存就绪")

    elif task == 'nta':
        print("  预加载 YOLO11...")
        from ultralytics import YOLO
        model_size = config.get('nta', {}).get('model_size', 'l')
        _ = YOLO(f"yolo11{model_size}.pt")
        print("  YOLO11 缓存就绪")

    elif task == 'ntl':
        print("  预加载 TwinLiteNet (ResNet18 backbone)...")
        from torchvision.models import resnet18, ResNet18_Weights
        _ = resnet18(weights=ResNet18_Weights.DEFAULT)
        print("  TwinLiteNet backbone 缓存就绪")

    print("模型缓存就绪!\n")


def evaluate_parallel(config: Dict, task: str = "depth",
                      gpu_ids: Optional[List[int]] = None,
                      save_vis: bool = True) -> Dict[str, Dict]:
    """
    多GPU并行评测

    Args:
        config: 配置字典
        task: 评测任务 (depth, seg, sam)
        gpu_ids: 使用的GPU列表，None则自动检测
        save_vis: 是否保存可视化

    Returns:
        评测结果
    """
    # 自动检测可用GPU
    if gpu_ids is None:
        num_gpus = torch.cuda.device_count()
        gpu_ids = list(range(num_gpus))

    print(f"\n可用GPU: {gpu_ids}")

    # 预下载模型，避免多进程并发下载冲突
    _preload_models(task, config)

    # 获取图像对
    image_pairs = get_image_pairs(config)
    if not image_pairs:
        raise ValueError("没有找到有效的图像对")

    cameras = list(image_pairs.keys())
    print(f"相机数量: {len(cameras)}, GPU数量: {len(gpu_ids)}")

    # 分配相机到GPU
    camera_gpu_map = {}
    for i, camera in enumerate(cameras):
        camera_gpu_map[camera] = gpu_ids[i % len(gpu_ids)]

    print("GPU分配:")
    for camera, gpu_id in camera_gpu_map.items():
        print(f"  {camera} -> GPU:{gpu_id}")

    # 选择worker函数
    worker_fn = {
        'depth': _worker_depth,
        'seg': _worker_seg,
        'segmentation': _worker_seg,
        'sam': _worker_sam,
        'image_metrics': _worker_image_metrics,
        'nta': _worker_nta,
        'ntl': _worker_ntl,
    }[task]

    # 创建输出目录
    if task == 'depth':
        ensure_dir(config['output']['depth_maps'])
    elif task in ['seg', 'segmentation']:
        ensure_dir(config['output']['seg_maps'])
    elif task == 'sam':
        ensure_dir(os.path.join(config['output'].get('root', './results'), 'sam_maps'))
    elif task == 'nta':
        ensure_dir(os.path.join(config['output'].get('root', './results'), 'nta_maps'))
    elif task == 'ntl':
        ensure_dir(os.path.join(config['output'].get('root', './results'), 'ntl_maps'))

    # 使用multiprocessing并行
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    result_dict = manager.dict()

    processes = []
    for camera in cameras:
        gpu_id = camera_gpu_map[camera]
        p = mp.Process(
            target=worker_fn,
            args=(camera, image_pairs[camera], config,
                  gpu_id, save_vis, result_dict)
        )
        processes.append(p)

    # 依次启动进程，间隔2秒避免资源竞争
    import time
    for i, p in enumerate(processes):
        p.start()
        if i < len(processes) - 1:
            time.sleep(2)

    # 等待所有进程完成
    for p in processes:
        p.join()

    # 收集结果
    results = dict(result_dict)

    # image_metrics任务：在主进程顺序计算FID（clean-fid要求cuda:0）
    if task == 'image_metrics':
        from image_metrics_eval import compute_fid_for_camera
        root = config['data']['root']
        print(f"\n在主进程计算FID (cuda:0)...")
        for camera in cameras:
            if camera in results and results[camera]:
                gen_dir = os.path.join(root, camera, config['data']['gen_folder'])
                gt_dir = os.path.join(root, camera, config['data']['gt_folder'])
                try:
                    print(f"  计算FID: {camera}")
                    fid_score = compute_fid_for_camera(gen_dir, gt_dir, device='cuda:0')
                    results[camera] = dict(results[camera])
                    results[camera]['fid'] = fid_score
                    print(f"    FID = {fid_score:.2f}")
                except Exception as e:
                    print(f"    FID失败: {e}")
                    results[camera] = dict(results[camera])
                    results[camera]['fid'] = float('nan')

    # 计算总体平均
    all_metrics = []
    for camera_results in results.values():
        if camera_results:
            filtered = {k: v for k, v in camera_results.items()
                       if not k.endswith('_std') and not isinstance(v, list)}
            all_metrics.append(filtered)

    if all_metrics:
        results['overall'] = aggregate_metrics(all_metrics)

    # 打印结果
    task_name = {'depth': '深度', 'seg': '分割', 'segmentation': '分割', 'sam': 'SAM结构',
                 'image_metrics': '图像质量', 'nta': 'NTA-IoU', 'ntl': 'NTL-IoU'}
    print(f"\n{'=' * 60}")
    print(f"总体{task_name.get(task, task)}评测结果:")
    if 'overall' in results:
        print(format_metrics_table(results['overall']))

    return results
