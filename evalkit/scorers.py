"""
逐样本指标计算

指标组：
    pixel  PSNR↑ / SSIM↑ / LPIPS↓              (image_metrics_eval.ImageMetricsEvaluator)
    depth  δ1↑ / AbsRel↓                         (Depth-Anything-V2-Large，视差空间 scale+shift 对齐)
    seg    Seg-Cons↑ / Coarse-mIoU↑ 的混淆矩阵    (Mask2Former Swin-L Cityscapes，含地面类)
    dist   KID↓ / FID↓                           (集合级，由 runner 落盘图像后计算)

GT 侧的模型输出与方法无关，由 reference() 计算一次，供同一样本的所有方法共用。
"""

from typing import Dict, Iterable, Tuple

import numpy as np

ALL_GROUPS = ("pixel", "depth", "seg", "dist")


class Scorer:
    def __init__(self, groups: Iterable[str], device: str = "cuda",
                 max_depth_ratio: float = 100.0):
        self.groups = set(groups)
        unknown = self.groups - set(ALL_GROUPS)
        if unknown:
            raise ValueError(f"未知指标组: {unknown}，可选 {ALL_GROUPS}")
        self.device = device
        self.max_depth_ratio = max_depth_ratio

        if "pixel" in self.groups:
            from image_metrics_eval import ImageMetricsEvaluator
            self.pixel = ImageMetricsEvaluator(device=device)
        if "depth" in self.groups:
            from depth_eval import get_depth_estimator
            self.depth_model = get_depth_estimator({'depth': {
                'model': 'depth_anything_v2', 'model_size': 'large', 'device': device}})
        if "seg" in self.groups:
            from seg_eval import get_segmentor
            self.seg_model = get_segmentor({'segmentation': {
                'model': 'mask2former', 'backbone': 'swin-l', 'dataset': 'cityscapes',
                'device': device}})

    def reference(self, gt_img: np.ndarray) -> Dict:
        """GT 侧模型输出"""
        ref = {}
        if "depth" in self.groups:
            ref["depth"] = self.depth_model.predict(gt_img)
        if "seg" in self.groups:
            ref["seg"] = self.seg_model.predict(gt_img)
        return ref

    def score(self, gen_img: np.ndarray, gt_img: np.ndarray,
              ref: Dict) -> Tuple[Dict[str, float], Dict]:
        """
        Returns:
            values: 逐样本标量指标
            extras: 需要集合级汇总的量（coarse_conf_matrix）
        """
        from metrics import (evaluate_depth_pair, compute_segmentation_metrics_multilevel,
                             compute_segmentation_consistency)

        values, extras = {}, {}
        if "pixel" in self.groups:
            values.update(self.pixel.evaluate_pair(gen_img, gt_img))
        if "depth" in self.groups:
            m = evaluate_depth_pair(self.depth_model.predict(gen_img), ref["depth"],
                                    max_depth_ratio=self.max_depth_ratio, with_tolerant=False)
            values["delta_1"] = float(m["delta_1"])
            values["abs_rel"] = float(m["abs_rel"])
        if "seg" in self.groups:
            seg_gen = self.seg_model.predict(gen_img)
            values["seg_cons"] = float(
                compute_segmentation_consistency(seg_gen, ref["seg"])["consistency"])
            m = compute_segmentation_metrics_multilevel(seg_gen, ref["seg"])
            extras["coarse_conf_matrix"] = np.asarray(m["coarse_conf_matrix"], dtype=np.int64)
        return values, extras


def compute_distribution_metrics(gen_dir: str, gt_dir: str, device: str) -> Dict[str, float]:
    """KID / FID（clean-fid, clean 模式）"""
    import torch
    from cleanfid import fid

    kwargs = dict(mode="clean", device=torch.device(device), num_workers=0)
    results = {}
    for name, fn in (("kid", fid.compute_kid), ("fid", fid.compute_fid)):
        try:
            results[name] = float(fn(gen_dir, gt_dir, use_dataparallel=False, **kwargs))
        except TypeError:  # 旧版 clean-fid 没有 use_dataparallel 参数
            results[name] = float(fn(gen_dir, gt_dir, **kwargs))
    return results
