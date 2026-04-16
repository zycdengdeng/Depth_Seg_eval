"""
GT 图像去畸变模块

车端 GT 图像带有镜头畸变，而 GS 方法和 Ours 的渲染都是无畸变的 1280x720。
评测前需要对 GT 图像做去畸变 + resize，使其与 gen 图像具有可比性。

标定文件：/mnt/car_road_data_TianJin/support_info/NoEER705_v3/camera/
- camera_{cam_id:02d}_intrinsics.yaml → K, D, width, height
- camera_{cam_id:02d}_extrinsics.yaml → R, t (本模块不需要)

注意：cam_id=1 (FN) 的畸变参数异常，跳过去畸变。
"""

import os
import yaml
import cv2
import numpy as np
from typing import Dict, Tuple, Optional

# 相机短名 → 标定文件中的 cam_id
CAMERA_SHORT_TO_ID = {
    "FN": 1,
    "FW": 2,
    "FL": 3,
    "FR": 4,
    "RL": 5,
    "RR": 6,
    "RN": 7,
}

# 默认标定目录
DEFAULT_CALIB_DIR = "/mnt/car_road_data_TianJin/support_info/NoEER705_v3"

# 缓存已加载的标定参数
_calib_cache: Dict[int, Tuple[np.ndarray, np.ndarray, int, int]] = {}


def load_camera_intrinsics(calib_dir: str, cam_id: int
                           ) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """加载相机内参和畸变系数

    Returns:
        K: 3x3 内参矩阵
        D: 畸变系数
        width: 标定时的图像宽度
        height: 标定时的图像高度
    """
    if cam_id in _calib_cache:
        return _calib_cache[cam_id]

    intr_path = os.path.join(calib_dir, "camera",
                              f"camera_{cam_id:02d}_intrinsics.yaml")
    with open(intr_path) as f:
        intr = yaml.safe_load(f)

    K = np.array(intr["K"], dtype=np.float64).reshape(3, 3)
    D = np.array(intr["D"], dtype=np.float64)
    w, h = intr["width"], intr["height"]

    _calib_cache[cam_id] = (K, D, w, h)
    return K, D, w, h


def undistort_image(image: np.ndarray, camera_short: str,
                    calib_dir: str = DEFAULT_CALIB_DIR) -> np.ndarray:
    """对单张图像做去畸变

    Args:
        image: BGR 或 RGB 图像 (H, W, 3), uint8
        camera_short: 相机短名 (FL/FN/FR/FW/RL/RR/RN)
        calib_dir: 标定文件根目录

    Returns:
        去畸变后的图像（同尺寸）
    """
    cam_id = CAMERA_SHORT_TO_ID.get(camera_short)
    if cam_id is None:
        print(f"  警告: 未知相机 {camera_short}，跳过去畸变")
        return image

    # cam_id=1 (FN) 畸变参数异常，跳过
    if cam_id == 1:
        return image

    K, D, calib_w, calib_h = load_camera_intrinsics(calib_dir, cam_id)

    h_img, w_img = image.shape[:2]

    # 缩放 K 到实际图像分辨率
    sx = w_img / calib_w
    sy = h_img / calib_h
    K_scaled = K.copy()
    K_scaled[0] *= sx
    K_scaled[1] *= sy

    # alpha=0: 裁掉去畸变后的黑边
    nK, _ = cv2.getOptimalNewCameraMatrix(
        K_scaled, D, (w_img, h_img), alpha=0, newImgSize=(w_img, h_img))

    undistorted = cv2.undistort(image, K_scaled, D, None, nK)
    return undistorted


def undistort_and_resize(image: np.ndarray, camera_short: str,
                         target_size: Tuple[int, int] = (1280, 720),
                         calib_dir: str = DEFAULT_CALIB_DIR) -> np.ndarray:
    """去畸变 + resize 到目标尺寸

    Args:
        image: BGR 或 RGB 图像
        camera_short: 相机短名
        target_size: (width, height) 目标尺寸，默认 1280x720
        calib_dir: 标定文件根目录

    Returns:
        去畸变并 resize 后的图像
    """
    undistorted = undistort_image(image, camera_short, calib_dir)
    if undistorted.shape[1] != target_size[0] or undistorted.shape[0] != target_size[1]:
        undistorted = cv2.resize(undistorted, target_size, interpolation=cv2.INTER_LINEAR)
    return undistorted


def load_gt_undistorted(gt_path: str, camera_short: str,
                        target_size: Tuple[int, int] = None,
                        calib_dir: str = DEFAULT_CALIB_DIR) -> np.ndarray:
    """加载 GT 图像并去畸变（返回 RGB numpy）

    Args:
        gt_path: GT 图像路径
        camera_short: 相机短名
        target_size: (width, height) 目标尺寸，None 则不 resize
        calib_dir: 标定文件根目录

    Returns:
        去畸变后的 RGB 图像 (H, W, 3), uint8
    """
    img_bgr = cv2.imread(gt_path)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图像: {gt_path}")

    img_bgr = undistort_image(img_bgr, camera_short, calib_dir)

    if target_size is not None:
        img_bgr = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_LINEAR)

    # BGR → RGB
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
