"""
测试集清单（render list）与 manifest 构建（路侧→车端 Tianjin 数据）

render list 以 Ours 的测试集为准：每个测试 clip 的 29 帧时间戳 × 7 相机。
所有方法都按 render list 生成/渲染，再用 build_manifest 对应到 GT。

GS 类方法的输出目录约定：
    {root}/{clip}/{timestamp}/{camera}.png
    clip      = 完整 clip 名（如 031_car0402_road0402_t9）
    timestamp = 路侧毫秒时间戳
    camera    = FL / FN / FR / FW / RL / RN / RR
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .loader import EVAL_SIZE
from .manifest import make_sample_id

GS_LAYOUT = "{root}/{clip}/{timestamp}/{camera}.png"


def eval_intrinsics(camera: str, gt_path: str,
                    eval_size: Tuple[int, int] = EVAL_SIZE) -> Dict:
    """
    评测分辨率下、GT 去畸变后的相机内参，交给各方法按此内参渲染

    与 undistort.undistort_image 完全一致：原分辨率下 getOptimalNewCameraMatrix(alpha=0)，
    再缩放到评测分辨率；FN 不做去畸变。
    """
    import cv2
    from PIL import Image
    from undistort import CAMERA_SHORT_TO_ID, DEFAULT_CALIB_DIR, load_camera_intrinsics

    cam_id = CAMERA_SHORT_TO_ID[camera]
    K, D, calib_w, calib_h = load_camera_intrinsics(DEFAULT_CALIB_DIR, cam_id)
    with Image.open(gt_path) as im:
        w, h = im.size

    K_scaled = K.copy()
    K_scaled[0] *= w / calib_w
    K_scaled[1] *= h / calib_h
    undistorted = cam_id != 1
    if undistorted:
        new_K, _ = cv2.getOptimalNewCameraMatrix(K_scaled, D, (w, h), alpha=0, newImgSize=(w, h))
    else:
        new_K = K_scaled

    new_K = new_K.copy()
    new_K[0] *= eval_size[0] / w
    new_K[1] *= eval_size[1] / h
    return {"K": np.round(new_K, 6).tolist(), "width": eval_size[0], "height": eval_size[1],
            "undistorted": undistorted, "gt_native_size": [w, h]}


def build_render_list(clips: Optional[List[str]] = None,
                      cameras: Optional[List[str]] = None) -> Dict:
    """以 Ours 测试集（tf_eval.CLIP_TS_RANGES）生成测试集清单"""
    import tf_eval

    clips = clips or list(tf_eval.CLIP_TS_RANGES)
    cameras = cameras or tf_eval.CAMERAS

    samples, missing_gt = [], 0
    for clip_num in clips:
        clip_full = tf_eval.find_clip_full_name(clip_num)
        if clip_full is None:
            continue
        frame_ts = tf_eval.compute_frame_timestamps(clip_num, tf_eval.FRAMES_PER_VIDEO)
        for camera in cameras:
            for frame_idx, ts in enumerate(frame_ts):
                gt = tf_eval.find_gt_path(clip_full, camera, ts)
                if gt is None:
                    missing_gt += 1
                    continue
                samples.append({"clip": clip_full, "clip_num": clip_num, "camera": camera,
                                "timestamp": ts, "frame_idx": frame_idx, "gt": gt})

    intrinsics = {}
    for camera in cameras:
        first = next((s for s in samples if s["camera"] == camera), None)
        if first is not None:
            intrinsics[camera] = eval_intrinsics(camera, first["gt"])

    print(f"测试集清单: {len(clips)} clips, {len(cameras)} 相机, {len(samples)} 个样本"
          f"（GT 缺失 {missing_gt}）")
    return {"eval_size": list(EVAL_SIZE), "gs_layout": GS_LAYOUT,
            "intrinsics": intrinsics, "samples": samples}


def build_manifest(render_list: Dict, gs_methods: Dict[str, str],
                   ours: Optional[Tuple[str, str]] = None) -> List[Dict]:
    """
    Args:
        render_list: build_render_list 的输出
        gs_methods: {方法名: 输出根目录}，目录结构为 GS_LAYOUT
        ours: (方法名, 视频根目录)，结构同 tf_eval：{root}/{clip_num}_{seg}/{相机长名}_generated.mp4
    """
    import tf_eval

    rows, missing = [], {}
    for s in render_list["samples"]:
        base = {"sample_id": make_sample_id(s["clip"], s["camera"], s["timestamp"]),
                "clip": s["clip"], "camera": s["camera"], "timestamp": s["timestamp"],
                "group": s.get("group", ""), "gt": s["gt"]}

        for method, root in gs_methods.items():
            gen = GS_LAYOUT.format(root=root, clip=s["clip"], timestamp=s["timestamp"],
                                   camera=s["camera"])
            if os.path.exists(gen):
                rows.append({**base, "method": method, "gen": gen})
            else:
                missing[method] = missing.get(method, 0) + 1

        if ours is not None:
            name, root = ours
            seg = tf_eval.CLIP_TS_RANGES[s["clip_num"]]["seg"]
            video = os.path.join(root, f"{s['clip_num']}_{seg}",
                                 f"{tf_eval.CAMERA_SHORT_TO_LONG[s['camera']]}_generated.mp4")
            if os.path.exists(video):
                rows.append({**base, "method": name, "gen": f"{video}#{s['frame_idx']}"})
            else:
                missing[name] = missing.get(name, 0) + 1

    n = len(render_list["samples"])
    for method in list(gs_methods) + ([ours[0]] if ours else []):
        print(f"  {method:<20} {n - missing.get(method, 0)}/{n} 个样本")
    return rows


def build_legacy_common_manifest(ours_name: str = "Ours") -> List[Dict]:
    """
    旧数据（gs_eval 的 near/middle/far 渲染 + tf_eval 的视频）在公共帧上的 manifest，
    用于验证 evalkit 与 gs_eval/tf_eval --common 结果一致
    """
    import gs_eval
    import tf_eval
    from common_frames import get_common_targets

    targets = get_common_targets()
    keep = {(t["clip"], t["distance"]): t for t in targets}
    rows = []

    for method, pairs in gs_eval.collect_image_pairs().items():
        for p in pairs:
            t = keep.get((p["clip"], p["distance"]))
            if t is None:
                continue
            rows.append({"sample_id": make_sample_id(p["clip"], p["camera"], t["road_ts"]),
                         "method": method, "clip": p["clip"], "camera": p["camera"],
                         "timestamp": t["road_ts"], "group": p["distance"],
                         "gen": p["gen_path"], "gt": p["gt_path"]})

    for t in targets:
        frame_ts = tf_eval.compute_frame_timestamps(t["clip_num"], tf_eval.FRAMES_PER_VIDEO)
        frame_idx = int(np.argmin([abs(x - t["road_ts"]) for x in frame_ts]))
        seg = tf_eval.CLIP_TS_RANGES[t["clip_num"]]["seg"]
        for camera in tf_eval.CAMERAS:
            video = os.path.join(tf_eval.GEN_ROOT, f"{t['clip_num']}_{seg}",
                                 f"{tf_eval.CAMERA_SHORT_TO_LONG[camera]}_generated.mp4")
            gt = tf_eval.find_gt_path(t["clip"], camera, t["road_ts"])
            if not os.path.exists(video) or gt is None:
                continue
            rows.append({"sample_id": make_sample_id(t["clip"], camera, t["road_ts"]),
                         "method": ours_name, "clip": t["clip"], "camera": camera,
                         "timestamp": t["road_ts"], "group": t["distance"],
                         "gen": f"{video}#{frame_idx}", "gt": gt})
    return rows


def save_render_list(render_list: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(render_list, f, indent=2, ensure_ascii=False)
    csv_path = os.path.splitext(path)[0] + ".csv"
    with open(csv_path, "w") as f:
        f.write("clip,camera,timestamp,frame_idx,gt\n")
        for s in render_list["samples"]:
            f.write(f"{s['clip']},{s['camera']},{s['timestamp']},{s['frame_idx']},{s['gt']}\n")
    print(f"已保存: {path}\n已保存: {csv_path}")


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def _index_images(root: str) -> Dict[str, str]:
    """{相对路径去扩展名: 绝对路径}"""
    index = {}
    for dirpath, _, files in os.walk(root):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                full = os.path.join(dirpath, name)
                key = os.path.splitext(os.path.relpath(full, root))[0]
                if key in index:
                    raise ValueError(f"{root} 中存在同名不同扩展名的文件: {key}")
                index[key] = full
    return index


def build_pairs_manifest(gen_dirs: Dict[str, str], gt_dir: str,
                         clip_depth: int = 1) -> List[Dict]:
    """
    按相对路径（忽略扩展名）把各方法的推理结果与 GT 配对

    Args:
        gen_dirs: {方法名: 推理结果目录}
        gt_dir: GT 目录（目录结构与推理结果一致）
        clip_depth: 相对路径前几级目录作为 clip（置信区间按 clip 重采样）；
            0 表示每个样本自成一个 clip（样本间独立时使用）
    """
    gt_index = _index_images(gt_dir)
    if not gt_index:
        raise ValueError(f"GT 目录中没有图像: {gt_dir}")

    rows = []
    for method, gen_dir in gen_dirs.items():
        gen_index = _index_images(gen_dir)
        matched = sorted(set(gen_index) & set(gt_index))
        print(f"  {method:<20} 配对 {len(matched)}/{len(gt_index)}"
              f"（推理结果多出 {len(set(gen_index) - set(gt_index))} 个未匹配文件）")
        for key in matched:
            parts = key.split(os.sep)
            clip = os.sep.join(parts[:clip_depth]) if 0 < clip_depth < len(parts) else key
            rows.append({"sample_id": key, "method": method, "clip": clip,
                         "gen": gen_index[key], "gt": gt_index[key]})
    return rows
