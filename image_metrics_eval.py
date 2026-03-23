"""
图像质量直接评测模块

直接比较生成图像和真值图像的像素/感知相似度。
不需要中间模型（深度/分割），直接对原始图像计算。

评测指标：
- PSNR: 峰值信噪比 (越高越好)
- SSIM: 结构相似性指数 (越高越好, 0-1)
- LPIPS: 感知相似度 (越低越好, 0-1)
- FID: Fréchet Inception Distance (越低越好, 分布级指标)
"""

import os
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from utils import load_config, get_image_pairs, load_image, load_image_pair, ensure_dir
from metrics import aggregate_metrics, format_metrics_table


class ImageMetricsEvaluator:
    """图像质量指标计算器（PSNR, SSIM, LPIPS）"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._lpips_fn = None

    def _load_lpips(self):
        """懒加载LPIPS模型"""
        if self._lpips_fn is None:
            import lpips
            self._lpips_fn = lpips.LPIPS(net='alex').to(self.device)
            self._lpips_fn.eval()
            print("LPIPS模型加载完成 (AlexNet)")
        return self._lpips_fn

    @staticmethod
    def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        计算PSNR (Peak Signal-to-Noise Ratio)

        Args:
            img1, img2: RGB图像 (H, W, 3), uint8

        Returns:
            PSNR值 (dB), 越高越好
        """
        mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
        if mse == 0:
            return 100.0  # 完全相同
        return float(10 * np.log10(255.0 ** 2 / mse))

    @staticmethod
    def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        计算SSIM (Structural Similarity Index)

        Args:
            img1, img2: RGB图像 (H, W, 3), uint8

        Returns:
            SSIM值 (0-1), 越高越好
        """
        from skimage.metrics import structural_similarity
        return float(structural_similarity(
            img1, img2, channel_axis=2, data_range=255
        ))

    @torch.no_grad()
    def compute_lpips(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        计算LPIPS (Learned Perceptual Image Patch Similarity)

        Args:
            img1, img2: RGB图像 (H, W, 3), uint8

        Returns:
            LPIPS值 (0-1), 越低越好
        """
        lpips_fn = self._load_lpips()

        # 转换为tensor: (H,W,3) uint8 -> (1,3,H,W) float [-1,1]
        def to_tensor(img):
            t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
            t = t / 127.5 - 1.0  # [0,255] -> [-1,1]
            return t.to(self.device)

        t1 = to_tensor(img1)
        t2 = to_tensor(img2)

        # LPIPS要求相同尺寸
        if t1.shape != t2.shape:
            h = min(t1.shape[2], t2.shape[2])
            w = min(t1.shape[3], t2.shape[3])
            t1 = t1[:, :, :h, :w]
            t2 = t2[:, :, :h, :w]

        score = lpips_fn(t1, t2)
        return float(score.item())

    def evaluate_pair(self, gen_img: np.ndarray, gt_img: np.ndarray) -> Dict[str, float]:
        """计算一对图像的所有指标"""
        return {
            'psnr': self.compute_psnr(gen_img, gt_img),
            'ssim': self.compute_ssim(gen_img, gt_img),
            'lpips': self.compute_lpips(gen_img, gt_img),
        }


def compute_fid_for_camera(gen_dir: str, gt_dir: str,
                            device: str = "cuda") -> float:
    """
    计算单个相机的FID (Fréchet Inception Distance)

    FID是分布级指标，需要对整个图像集合计算。

    Args:
        gen_dir: 生成图像目录
        gt_dir: 真值图像目录
        device: 计算设备

    Returns:
        FID分数, 越低越好
    """
    try:
        from cleanfid import fid
        score = fid.compute_fid(gen_dir, gt_dir,
                                device=torch.device(device),
                                num_workers=0)  # 禁止嵌套多进程，避免spawn下pickle冲突
        return float(score)
    except ImportError:
        print("警告: cleanfid未安装，使用torch-fidelity计算FID...")
        try:
            from torch_fidelity import calculate_metrics
            metrics = calculate_metrics(
                input1=gen_dir,
                input2=gt_dir,
                cuda=(device != 'cpu'),
                fid=True,
            )
            return float(metrics['frechet_inception_distance'])
        except ImportError:
            print("错误: 需要安装 cleanfid 或 torch-fidelity 来计算FID")
            print("  pip install clean-fid  或  pip install torch-fidelity")
            return float('nan')


def evaluate_image_metrics(config: Dict, save_vis: bool = False) -> Dict[str, Dict]:
    """
    评测图像质量直接指标 (PSNR, SSIM, LPIPS, FID)

    Args:
        config: 配置字典
        save_vis: 未使用（保持接口一致）

    Returns:
        每个相机的评测结果
    """
    print("\n" + "=" * 60)
    print("图像质量直接评测 (PSNR / SSIM / LPIPS / FID)")
    print("=" * 60)

    device = config.get('image_metrics', {}).get('device',
                config.get('depth', {}).get('device', 'cuda'))

    evaluator = ImageMetricsEvaluator(device=device)

    # 获取图像对
    image_pairs = get_image_pairs(config)
    if not image_pairs:
        raise ValueError("没有找到有效的图像对")

    results = {}
    root = config['data']['root']

    for camera, pairs in image_pairs.items():
        print(f"\n处理相机: {camera}")
        camera_metrics = []

        for gen_path, gt_path in tqdm(pairs, desc=f"  {camera}"):
            gen_img, gt_img = load_image_pair(gen_path, gt_path)

            metrics = evaluator.evaluate_pair(gen_img, gt_img)
            camera_metrics.append(metrics)

        # 汇总per-image指标
        results[camera] = aggregate_metrics(camera_metrics)

        # 计算FID（分布级指标）
        gen_dir = os.path.join(root, camera, config['data']['gen_folder'])
        gt_dir = os.path.join(root, camera, config['data']['gt_folder'])
        print(f"  计算FID: {camera}")
        fid_score = compute_fid_for_camera(gen_dir, gt_dir, device=device)
        results[camera]['fid'] = fid_score

        print(f"\n  {camera} 图像质量指标:")
        print(f"    PSNR:  {results[camera].get('psnr', 0):.2f} dB")
        print(f"    SSIM:  {results[camera].get('ssim', 0):.4f}")
        print(f"    LPIPS: {results[camera].get('lpips', 0):.4f}")
        print(f"    FID:   {fid_score:.2f}")

    # 总体平均
    all_metrics = []
    all_fids = []
    for camera_results in results.values():
        filtered = {k: v for k, v in camera_results.items()
                   if not k.endswith('_std') and k != 'fid'}
        all_metrics.append(filtered)
        if 'fid' in camera_results and not np.isnan(camera_results['fid']):
            all_fids.append(camera_results['fid'])

    results['overall'] = aggregate_metrics(all_metrics)
    if all_fids:
        results['overall']['fid'] = np.mean(all_fids)
        results['overall']['fid_std'] = np.std(all_fids)

    print("\n" + "=" * 60)
    print("总体图像质量评测结果:")
    overall = results['overall']
    print(f"  PSNR:  {overall.get('psnr', 0):.2f} ± {overall.get('psnr_std', 0):.2f} dB")
    print(f"  SSIM:  {overall.get('ssim', 0):.4f} ± {overall.get('ssim_std', 0):.4f}")
    print(f"  LPIPS: {overall.get('lpips', 0):.4f} ± {overall.get('lpips_std', 0):.4f}")
    print(f"  FID:   {overall.get('fid', float('nan')):.2f} ± {overall.get('fid_std', 0):.2f}")

    return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="图像质量直接评测")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="配置文件路径")
    parser.add_argument("--output", type=str, default=None,
                       help="结果输出路径")
    args = parser.parse_args()

    config = load_config(args.config)
    results = evaluate_image_metrics(config)

    output_path = args.output or os.path.join(
        config['output']['metrics'], "image_metrics_results.json"
    )
    ensure_dir(os.path.dirname(output_path))

    import json
    from evaluate import convert_to_serializable
    with open(output_path, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
