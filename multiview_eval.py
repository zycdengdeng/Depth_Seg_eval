#!/usr/bin/env python3
"""
多视角一致性评测模块

评测相邻视角间的几何一致性，包含两种方法：
1. 深度重投影一致性：用深度图将一个视角 warp 到另一个视角，比较像素差异
2. 跨视角特征匹配：用 LoFTR 找匹配点，计算 epipolar error 和 inlier 比例

相邻视角对（有视野重叠）：
  前方: FL↔FW, FR↔FW, FN↔FW
  后方: RL↔RN, RR↔RN

使用方法：
    # GS 方法
    python multiview_eval.py --source gs --clip 031 --methods DDGS AD-GS --device cuda:0
    # TF (Ours)
    python multiview_eval.py --source tf --clip 031 --device cuda:0

需要安装：
    pip install kornia  # LoFTR
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import yaml
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime

from undistort import CAMERA_SHORT_TO_ID, DEFAULT_CALIB_DIR


# ============== 相邻视角对 ==============

ADJACENT_PAIRS = [
    ("FL", "FW"),
    ("FR", "FW"),
    ("FN", "FW"),
    ("RL", "RN"),
    ("RR", "RN"),
]


# ============== 标定加载 ==============

_extr_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
_intr_cache: Dict[int, Tuple[np.ndarray, np.ndarray, int, int]] = {}


def load_intrinsics(cam_short: str, calib_dir: str = DEFAULT_CALIB_DIR,
                    img_w: int = 1280, img_h: int = 720
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """加载内参并缩放到目标分辨率，返回去畸变后的 nK

    Returns:
        nK: 去畸变后的 3x3 内参矩阵（对应 1280x720 无畸变图像）
        K_raw: 原始缩放后的内参
    """
    cam_id = CAMERA_SHORT_TO_ID[cam_short]

    if cam_id in _intr_cache:
        return _intr_cache[cam_id]

    intr_path = os.path.join(calib_dir, "camera",
                              f"camera_{cam_id:02d}_intrinsics.yaml")
    with open(intr_path) as f:
        intr = yaml.safe_load(f)
    K = np.array(intr["K"], dtype=np.float64).reshape(3, 3)
    D = np.array(intr["D"], dtype=np.float64)
    calib_w, calib_h = intr["width"], intr["height"]

    # 缩放到目标分辨率
    sx = img_w / calib_w
    sy = img_h / calib_h
    K_scaled = K.copy()
    K_scaled[0] *= sx
    K_scaled[1] *= sy

    # 去畸变后的内参（alpha=0）
    if cam_id == 1:  # FN 跳过
        nK = K_scaled
    else:
        nK, _ = cv2.getOptimalNewCameraMatrix(
            K_scaled, D, (img_w, img_h), alpha=0, newImgSize=(img_w, img_h))

    _intr_cache[cam_id] = (nK, K_scaled)
    return nK, K_scaled


def load_extrinsics(cam_short: str, calib_dir: str = DEFAULT_CALIB_DIR
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """加载外参: R_c2l, t_c2l (相机→LiDAR坐标系)

    p_lidar = R_c2l @ p_cam + t_c2l
    """
    cam_id = CAMERA_SHORT_TO_ID[cam_short]

    if cam_id in _extr_cache:
        return _extr_cache[cam_id]

    from scipy.spatial.transform import Rotation

    extr_path = os.path.join(calib_dir, "camera",
                              f"camera_{cam_id:02d}_extrinsics.yaml")
    with open(extr_path) as f:
        extr = yaml.safe_load(f)
    q = extr["transform"]["rotation"]
    t = extr["transform"]["translation"]
    R_c2l = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    t_c2l = np.array([t["x"], t["y"], t["z"]], dtype=np.float64)

    _extr_cache[cam_id] = (R_c2l, t_c2l)
    return R_c2l, t_c2l


def get_relative_pose(cam_a: str, cam_b: str, calib_dir: str = DEFAULT_CALIB_DIR
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """计算相机 A → 相机 B 的相对位姿

    Returns:
        R_a2b: 3x3 旋转矩阵
        t_a2b: 3 平移向量
    满足: p_b = R_a2b @ p_a + t_a2b
    """
    R_a2l, t_a2l = load_extrinsics(cam_a, calib_dir)
    R_b2l, t_b2l = load_extrinsics(cam_b, calib_dir)

    # cam_a → lidar → cam_b
    # p_lidar = R_a2l @ p_a + t_a2l
    # p_b = R_b2l^{-1} @ (p_lidar - t_b2l) = R_b2l^T @ (R_a2l @ p_a + t_a2l - t_b2l)
    R_l2b = R_b2l.T
    t_l2b = -R_b2l.T @ t_b2l

    R_a2b = R_l2b @ R_a2l
    t_a2b = R_l2b @ t_a2l + t_l2b

    return R_a2b, t_a2b


def compute_fundamental_matrix(cam_a: str, cam_b: str,
                                calib_dir: str = DEFAULT_CALIB_DIR
                                ) -> np.ndarray:
    """计算 Fundamental Matrix F，满足 x_b^T F x_a = 0"""
    R, t = get_relative_pose(cam_a, cam_b, calib_dir)
    nK_a, _ = load_intrinsics(cam_a, calib_dir)
    nK_b, _ = load_intrinsics(cam_b, calib_dir)

    # Essential matrix E = [t]_x R
    tx = np.array([[0, -t[2], t[1]],
                    [t[2], 0, -t[0]],
                    [-t[1], t[0], 0]])
    E = tx @ R

    # F = K_b^{-T} E K_a^{-1}
    F = np.linalg.inv(nK_b).T @ E @ np.linalg.inv(nK_a)
    return F


# ============== 深度重投影一致性 ==============

def depth_reprojection_consistency(img_a: np.ndarray, img_b: np.ndarray,
                                    depth_a: np.ndarray, depth_b: np.ndarray,
                                    cam_a: str, cam_b: str,
                                    calib_dir: str = DEFAULT_CALIB_DIR
                                    ) -> Dict[str, float]:
    """计算视角A→B的深度重投影一致性

    1. 用 depth_a + K_a + 外参把 A 的像素反投影到 3D
    2. 用 K_b + 外参把 3D 投影到 B 的像素
    3. 在重叠区域比较 warp 后的图像和 B 的原始图像

    Returns:
        warp_psnr: warp 后与 B 的 PSNR
        warp_ssim: warp 后与 B 的 SSIM
        overlap_ratio: 重叠区域比例
        depth_consistency: 深度一致性（重投影深度 vs depth_b 的相对误差）
    """
    h, w = img_a.shape[:2]
    nK_a, _ = load_intrinsics(cam_a, calib_dir, w, h)
    nK_b, _ = load_intrinsics(cam_b, calib_dir, w, h)
    R_a2b, t_a2b = get_relative_pose(cam_a, cam_b, calib_dir)

    # 像素网格
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    ones = np.ones_like(u)
    uv1 = np.stack([u, v, ones], axis=-1).reshape(-1, 3).T  # (3, N)

    # 反投影到相机A坐标系
    depth_flat = depth_a.flatten()
    valid = depth_flat > 0.1
    pts_a = np.linalg.inv(nK_a) @ uv1  # (3, N) normalized
    pts_a = pts_a * depth_flat[None, :]  # (3, N) in cam_a 3D

    # 变换到相机B坐标系
    pts_b = R_a2b @ pts_a + t_a2b[:, None]  # (3, N)

    # 投影到相机B像素
    uv_b = nK_b @ pts_b  # (3, N)
    z_b = uv_b[2, :]
    front = z_b > 0.1
    uv_b = uv_b[:2, :] / (z_b[None, :] + 1e-8)

    # 找在B图像范围内的点
    u_b = uv_b[0, :].astype(int)
    v_b = uv_b[1, :].astype(int)
    in_bounds = (u_b >= 0) & (u_b < w) & (v_b >= 0) & (v_b < h) & valid & front

    if in_bounds.sum() < 100:
        return {
            'warp_psnr': float('nan'),
            'warp_ssim': float('nan'),
            'overlap_ratio': 0.0,
            'depth_consistency': float('nan'),
        }

    overlap_ratio = in_bounds.sum() / (h * w)

    # 构建 warp 图像
    warp_img = np.zeros_like(img_b)
    src_idx = np.where(in_bounds)[0]
    src_v = (src_idx // w).astype(int)
    src_u = (src_idx % w).astype(int)
    dst_u = u_b[in_bounds]
    dst_v = v_b[in_bounds]
    warp_img[dst_v, dst_u] = img_a[src_v, src_u]

    # 创建有效 mask（warp 过去的像素）
    warp_mask = np.zeros((h, w), dtype=bool)
    warp_mask[dst_v, dst_u] = True

    if warp_mask.sum() < 100:
        return {
            'warp_psnr': float('nan'),
            'warp_ssim': float('nan'),
            'overlap_ratio': overlap_ratio,
            'depth_consistency': float('nan'),
        }

    # 在重叠区域计算 PSNR
    warp_pixels = warp_img[warp_mask].astype(np.float64)
    ref_pixels = img_b[warp_mask].astype(np.float64)
    mse = np.mean((warp_pixels - ref_pixels) ** 2)
    warp_psnr = 10 * np.log10(255.0 ** 2 / (mse + 1e-8)) if mse > 0 else 100.0

    # 深度一致性：重投影深度 vs B 视角的深度
    reproj_depth = z_b[in_bounds]
    actual_depth = depth_b[dst_v, dst_u]
    valid_depth = actual_depth > 0.1
    if valid_depth.sum() > 0:
        depth_ratio = reproj_depth[valid_depth] / (actual_depth[valid_depth] + 1e-8)
        depth_consistency = np.mean(np.abs(depth_ratio - 1.0))
    else:
        depth_consistency = float('nan')

    return {
        'warp_psnr': warp_psnr,
        'overlap_ratio': overlap_ratio * 100,
        'depth_consistency': depth_consistency,
    }


# ============== LoFTR 跨视角特征匹配 ==============

_loftr_model = None


def _get_loftr(device: str):
    global _loftr_model
    if _loftr_model is None:
        import torch
        try:
            from kornia.feature import LoFTR
            _loftr_model = LoFTR(pretrained='outdoor').to(device).eval()
            print("  LoFTR 模型加载完成 (outdoor)")
        except Exception as e:
            print(f"  LoFTR 加载失败: {e}")
            print("  请安装: pip install kornia")
            return None
    return _loftr_model


def loftr_matching_consistency(img_a: np.ndarray, img_b: np.ndarray,
                                cam_a: str, cam_b: str,
                                device: str = "cuda",
                                calib_dir: str = DEFAULT_CALIB_DIR
                                ) -> Dict[str, float]:
    """用 LoFTR 做跨视角特征匹配，计算匹配质量

    Returns:
        num_matches: 总匹配点数
        inlier_ratio: epipolar error < 3px 的比例
        mean_epipolar_error: 平均 epipolar error (px)
        median_epipolar_error: 中位 epipolar error (px)
    """
    import torch

    matcher = _get_loftr(device)
    if matcher is None:
        return {
            'num_matches': 0,
            'inlier_ratio': float('nan'),
            'mean_epipolar_error': float('nan'),
            'median_epipolar_error': float('nan'),
        }

    # 转灰度 + resize 到 LoFTR 输入
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)

    # LoFTR 需要 (1, 1, H, W) float tensor
    t_a = torch.from_numpy(gray_a).float()[None, None].to(device) / 255.0
    t_b = torch.from_numpy(gray_b).float()[None, None].to(device) / 255.0

    with torch.no_grad():
        result = matcher({"image0": t_a, "image1": t_b})

    kpts_a = result["keypoints0"].cpu().numpy()  # (N, 2)
    kpts_b = result["keypoints1"].cpu().numpy()  # (N, 2)
    confidence = result["confidence"].cpu().numpy()  # (N,)

    num_matches = len(kpts_a)
    if num_matches < 5:
        return {
            'num_matches': num_matches,
            'inlier_ratio': 0.0,
            'mean_epipolar_error': float('nan'),
            'median_epipolar_error': float('nan'),
        }

    # 计算 Fundamental Matrix
    F = compute_fundamental_matrix(cam_a, cam_b, calib_dir)

    # 计算 epipolar error: |x_b^T F x_a| / ||Fx_a||
    pts_a_h = np.hstack([kpts_a, np.ones((num_matches, 1))])  # (N, 3)
    pts_b_h = np.hstack([kpts_b, np.ones((num_matches, 1))])  # (N, 3)

    Fxa = (F @ pts_a_h.T).T  # (N, 3) epipolar lines in image B
    numerator = np.abs(np.sum(pts_b_h * Fxa, axis=1))  # |x_b^T F x_a|
    denominator = np.sqrt(Fxa[:, 0] ** 2 + Fxa[:, 1] ** 2) + 1e-8
    epipolar_errors = numerator / denominator  # in pixels

    inlier_threshold = 3.0  # pixels
    inlier_ratio = np.mean(epipolar_errors < inlier_threshold) * 100

    return {
        'num_matches': num_matches,
        'inlier_ratio': inlier_ratio,
        'mean_epipolar_error': float(np.mean(epipolar_errors)),
        'median_epipolar_error': float(np.median(epipolar_errors)),
    }


# ============== 综合评测 ==============

def evaluate_multiview_consistency(images: Dict[str, np.ndarray],
                                    device: str = "cuda",
                                    calib_dir: str = DEFAULT_CALIB_DIR
                                    ) -> Dict[str, Dict]:
    """对一组7视角图像评测多视角一致性

    Args:
        images: {camera_short: rgb_image} 字典，如 {"FL": img, "FW": img, ...}
        device: 推理设备
        calib_dir: 标定目录

    Returns:
        每个相邻对的一致性指标
    """
    # 加载深度模型
    from depth_eval import get_depth_estimator

    depth_config = {'depth': {'model': 'depth_anything_v2', 'model_size': 'large', 'device': device}}
    depth_model = get_depth_estimator(depth_config)

    # 预测所有视角的深度图
    depths = {}
    for cam, img in images.items():
        depths[cam] = depth_model.predict(img)

    results = {}
    for cam_a, cam_b in ADJACENT_PAIRS:
        if cam_a not in images or cam_b not in images:
            continue

        pair_name = f"{cam_a}_{cam_b}"

        # 深度重投影 (双向)
        reproj_ab = depth_reprojection_consistency(
            images[cam_a], images[cam_b], depths[cam_a], depths[cam_b],
            cam_a, cam_b, calib_dir
        )
        reproj_ba = depth_reprojection_consistency(
            images[cam_b], images[cam_a], depths[cam_b], depths[cam_a],
            cam_b, cam_a, calib_dir
        )

        # 取双向平均
        pair_metrics = {}
        for key in reproj_ab:
            val_ab = reproj_ab[key]
            val_ba = reproj_ba[key]
            if np.isnan(val_ab) or np.isnan(val_ba):
                pair_metrics[f"reproj_{key}"] = float('nan')
            else:
                pair_metrics[f"reproj_{key}"] = (val_ab + val_ba) / 2

        # LoFTR 匹配
        loftr_res = loftr_matching_consistency(
            images[cam_a], images[cam_b], cam_a, cam_b, device, calib_dir
        )
        for key, val in loftr_res.items():
            pair_metrics[f"loftr_{key}"] = val

        results[pair_name] = pair_metrics

    return results


# ============== 图像收集（复用已有 dataloader）==============

def collect_gs_images(clip: str, distance: str) -> Dict[str, Dict[str, np.ndarray]]:
    """收集 GS 方法的7视角图像

    Returns: {method_name: {camera: rgb_image}}
    """
    from gs_eval import METHODS, CAMERAS, get_gen_path
    from PIL import Image

    result = {}
    for method in METHODS:
        images = {}
        for cam in CAMERAS:
            path = get_gen_path(method, clip, distance, cam)
            if os.path.exists(path):
                images[cam] = np.array(Image.open(path).convert('RGB'))
        if images:
            result[method] = images
    return result


def collect_gt_images(clip: str, distance: str) -> Dict[str, np.ndarray]:
    """收集 GT 的7视角图像（去畸变）"""
    from gs_eval import CAMERAS, find_gt_path
    from undistort import load_gt_undistorted

    images = {}
    for cam in CAMERAS:
        gt_path = find_gt_path(clip, distance, cam)
        if gt_path:
            images[cam] = load_gt_undistorted(gt_path, cam, target_size=(1280, 720))
    return images


def collect_tf_images(clip_num: str) -> Dict[str, np.ndarray]:
    """收集 TF (Ours) 的7视角图像（取中间帧）"""
    from tf_eval import (CLIP_TS_RANGES, CAMERA_SHORT_TO_LONG,
                          GEN_ROOT, extract_frames)
    from gs_eval import CAMERAS

    if clip_num not in CLIP_TS_RANGES:
        return {}

    seg = CLIP_TS_RANGES[clip_num]["seg"]
    clip_dir = f"{clip_num}_{seg}"
    images = {}

    for cam in CAMERAS:
        cam_long = CAMERA_SHORT_TO_LONG.get(cam)
        if not cam_long:
            continue
        video_path = os.path.join(GEN_ROOT, clip_dir, f"{cam_long}_generated.mp4")
        if not os.path.exists(video_path):
            continue
        frames = extract_frames(video_path)
        if frames:
            images[cam] = frames[len(frames) // 2]  # 取中间帧

    return images


# ============== 主函数 ==============

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
    else:
        return obj


def main():
    parser = argparse.ArgumentParser(description="多视角一致性评测")
    parser.add_argument("--source", type=str, required=True,
                        choices=["gs", "tf", "gt"],
                        help="数据来源: gs(GS方法), tf(TF/Ours), gt(GT)")
    parser.add_argument("--clip", type=str, required=True,
                        help="clip 编号或完整名")
    parser.add_argument("--distance", type=str, default="middle",
                        choices=["near", "middle", "far"],
                        help="距离（GS和GT用，TF忽略）")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="GS方法名（默认全部）")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str,
                        default="./results/multiview_eval")
    args = parser.parse_args()

    from gs_eval import CLIP_TIMESTAMPS

    # 解析 clip 名
    matching = [c for c in CLIP_TIMESTAMPS
                if c == args.clip or c.startswith(f"{args.clip}_")]
    if not matching:
        print(f"错误: clip '{args.clip}' 不在 CLIP_TIMESTAMPS 中")
        return
    clip_full = matching[0]
    clip_num = clip_full.split("_")[0]

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("多视角一致性评测")
    print(f"Clip: {clip_full}, Distance: {args.distance}")
    print(f"相邻对: {[f'{a}↔{b}' for a, b in ADJACENT_PAIRS]}")
    print("=" * 70)

    all_results = {}

    if args.source in ["gs", "gt"]:
        # GT
        if args.source == "gt":
            print("\n收集 GT 图像...")
            gt_imgs = collect_gt_images(clip_full, args.distance)
            if gt_imgs:
                print(f"  GT: {len(gt_imgs)} 视角")
                print("\n评测 GT 多视角一致性...")
                all_results["GT"] = evaluate_multiview_consistency(
                    gt_imgs, args.device)

        # GS 方法
        if args.source == "gs":
            print("\n收集 GS 方法图像...")
            gs_all = collect_gs_images(clip_full, args.distance)
            methods = args.methods or list(gs_all.keys())
            for method in methods:
                if method not in gs_all:
                    continue
                imgs = gs_all[method]
                print(f"\n评测 {method} 多视角一致性 ({len(imgs)} 视角)...")
                all_results[method] = evaluate_multiview_consistency(
                    imgs, args.device)

    elif args.source == "tf":
        print("\n收集 TF (Ours) 图像...")
        tf_imgs = collect_tf_images(clip_num)
        if tf_imgs:
            print(f"  TF_Ours: {len(tf_imgs)} 视角")
            print("\n评测 TF_Ours 多视角一致性...")
            all_results["TF_Ours"] = evaluate_multiview_consistency(
                tf_imgs, args.device)

    # 打印结果
    if all_results:
        print("\n" + "=" * 70)
        print("结果汇总")
        print("=" * 70)

        for source_name, pair_results in all_results.items():
            print(f"\n[{source_name}]")
            for pair_name, metrics in pair_results.items():
                print(f"  {pair_name}:")
                for k, v in metrics.items():
                    if isinstance(v, float):
                        print(f"    {k:<30} {v:.4f}")
                    else:
                        print(f"    {k:<30} {v}")

        # 保存
        output_path = os.path.join(
            args.output_dir,
            f"multiview_{args.source}_{clip_num}_{args.distance}.json")
        with open(output_path, 'w') as f:
            json.dump(convert_to_serializable({
                "timestamp": datetime.now().isoformat(),
                "clip": clip_full,
                "distance": args.distance,
                "results": all_results,
            }), f, indent=2)
        print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()
