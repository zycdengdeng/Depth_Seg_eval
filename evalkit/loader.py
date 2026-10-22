"""
图像加载与预处理：所有方法的 gen 与 GT 统一到同一分辨率
"""

import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image

EVAL_SIZE = (1280, 720)  # (W, H)
_VIDEO_CACHE_SIZE = 4


class ImageLoader:
    """
    Args:
        eval_size: 评测分辨率 (W, H)
        gt_undistort: GT 预处理方式。"none" 不处理；"tianjin" 用 undistort.py 的标定去畸变
    """

    def __init__(self, eval_size: Tuple[int, int] = EVAL_SIZE, gt_undistort: str = "none"):
        self.eval_size = tuple(eval_size)
        self.gt_undistort = gt_undistort
        self._video_cache = {}  # path -> frames，最多保留 _VIDEO_CACHE_SIZE 个视频

    def _resize(self, img: np.ndarray, resample) -> np.ndarray:
        if (img.shape[1], img.shape[0]) == self.eval_size:
            return img
        return np.array(Image.fromarray(img).resize(self.eval_size, resample))

    def _video_frame(self, path: str, index: int) -> np.ndarray:
        # 样本按 (clip, 相机, 时间) 顺序访问，同一视频的帧是连续读取的，少量缓存即可
        if path not in self._video_cache:
            if len(self._video_cache) >= _VIDEO_CACHE_SIZE:
                self._video_cache.pop(next(iter(self._video_cache)))
            import cv2
            cap = cv2.VideoCapture(path)
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            if not frames:
                raise FileNotFoundError(f"无法读取视频: {path}")
            self._video_cache[path] = frames
        return self._video_cache[path][index]

    def load_gen(self, spec: str) -> np.ndarray:
        """加载 gen：图像路径或 "video.mp4#帧序号"，resize 到评测分辨率（bicubic）"""
        if "#" in spec and os.path.splitext(spec.split("#")[0])[1].lower() in (".mp4", ".avi", ".mov"):
            path, index = spec.rsplit("#", 1)
            img = self._video_frame(path, int(index))
        else:
            img = np.array(Image.open(spec).convert("RGB"))
        return self._resize(img, Image.BICUBIC)

    def load_gt(self, path: str, camera: Optional[str] = None) -> np.ndarray:
        """加载 GT：可选去畸变，resize 到评测分辨率（bilinear）"""
        if self.gt_undistort == "tianjin":
            from undistort import load_gt_undistorted
            return load_gt_undistorted(path, camera, target_size=self.eval_size)
        img = np.array(Image.open(path).convert("RGB"))
        return self._resize(img, Image.BILINEAR)
