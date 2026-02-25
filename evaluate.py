#!/usr/bin/env python3
"""
主评测脚本

评测生成图像与真值图像在深度估计和语义分割上的一致性。

使用方法：
    python evaluate.py --config config.yaml --task all
    python evaluate.py --task depth
    python evaluate.py --task segmentation
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, Optional

from utils import load_config, ensure_dir


def convert_to_serializable(obj):
    """将numpy类型转换为Python原生类型，以便JSON序列化"""
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
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def run_depth_evaluation(config: Dict, save_vis: bool = True) -> Dict:
    """运行深度评测"""
    from depth_eval import evaluate_depth_consistency
    return evaluate_depth_consistency(config, save_vis=save_vis)


def run_segmentation_evaluation(config: Dict, save_vis: bool = True) -> Dict:
    """运行分割评测"""
    from seg_eval import evaluate_segmentation_consistency
    return evaluate_segmentation_consistency(config, save_vis=save_vis)


def run_sam_evaluation(config: Dict, save_vis: bool = True) -> Dict:
    """运行SAM结构一致性评测"""
    from sam_eval import evaluate_sam_consistency
    return evaluate_sam_consistency(config, save_vis=save_vis)


def generate_report(depth_results: Optional[Dict],
                    seg_results: Optional[Dict],
                    output_path: str,
                    sam_results: Optional[Dict] = None):
    """生成综合评测报告"""
    report = []
    report.append("=" * 70)
    report.append("生成图像质量评测报告")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)

    if depth_results:
        report.append("\n## 深度一致性评测")
        report.append("-" * 50)
        report.append("评测指标说明:")
        report.append("  - abs_rel: 绝对相对误差 (越低越好)")
        report.append("  - rmse: 均方根误差 (越低越好)")
        report.append("  - delta_1/2/3: 阈值准确率 (越高越好)")
        report.append("  - pearson/spearman: 相关系数 (越高越好)")
        report.append("-" * 50)

        if 'overall' in depth_results:
            overall = depth_results['overall']
            report.append("\n总体结果:")
            for key in ['abs_rel', 'rmse', 'delta_1', 'delta_2', 'delta_3', 'pearson', 'spearman']:
                if key in overall:
                    value = overall[key]
                    std_key = f'{key}_std'
                    if std_key in overall:
                        report.append(f"  {key:<12}: {value:.4f} ± {overall[std_key]:.4f}")
                    else:
                        report.append(f"  {key:<12}: {value:.4f}")

        report.append("\n各相机结果:")
        for camera, results in depth_results.items():
            if camera != 'overall':
                report.append(f"\n  [{camera}]")
                for key in ['abs_rel', 'delta_1', 'pearson']:
                    if key in results:
                        report.append(f"    {key:<12}: {results[key]:.4f}")

    if seg_results:
        report.append("\n\n## 语义分割一致性评测")
        report.append("-" * 50)
        report.append("评测指标说明:")
        report.append("  - consistency: 分割结果一致率 (越高越好)")
        report.append("  - miou: 平均交并比 (越高越好)")
        report.append("  - pixel_acc: 像素准确率 (越高越好)")
        report.append("-" * 50)

        if 'overall' in seg_results:
            overall = seg_results['overall']
            report.append("\n总体结果:")
            for key in ['consistency', 'miou', 'pixel_acc', 'fwiou']:
                if key in overall:
                    value = overall[key]
                    std_key = f'{key}_std'
                    if std_key in overall:
                        report.append(f"  {key:<12}: {value:.2f}% ± {overall[std_key]:.2f}%")
                    else:
                        report.append(f"  {key:<12}: {value:.2f}%")

        report.append("\n各相机结果:")
        for camera, results in seg_results.items():
            if camera != 'overall':
                report.append(f"\n  [{camera}]")
                for key in ['consistency', 'miou']:
                    if key in results:
                        report.append(f"    {key:<12}: {results[key]:.2f}%")

    if sam_results:
        report.append("\n\n## SAM 结构一致性评测")
        report.append("-" * 50)
        report.append("评测指标说明:")
        report.append("  - edge_f1: 边缘F1分数 (越高越好)")
        report.append("  - edge_correlation: 边缘相关系数 (越高越好)")
        report.append("  - edge_precision/recall: 边缘精确率/召回率 (越高越好)")
        report.append("-" * 50)

        if 'overall' in sam_results:
            overall = sam_results['overall']
            report.append("\n总体结果:")
            for key in ['edge_f1', 'edge_correlation', 'edge_precision', 'edge_recall']:
                if key in overall:
                    value = overall[key]
                    std_key = f'{key}_std'
                    if key == 'edge_correlation':
                        fmt = f"  {key:<20}: {value:.4f}"
                        if std_key in overall:
                            fmt += f" ± {overall[std_key]:.4f}"
                    else:
                        fmt = f"  {key:<20}: {value:.2f}%"
                        if std_key in overall:
                            fmt += f" ± {overall[std_key]:.2f}%"
                    report.append(fmt)

        report.append("\n各相机结果:")
        for camera, results in sam_results.items():
            if camera != 'overall':
                report.append(f"\n  [{camera}]")
                for key in ['edge_f1', 'edge_correlation']:
                    if key in results:
                        if key == 'edge_correlation':
                            report.append(f"    {key:<20}: {results[key]:.4f}")
                        else:
                            report.append(f"    {key:<20}: {results[key]:.2f}%")

    report.append("\n" + "=" * 70)
    report.append("评测完成")
    report.append("=" * 70)

    report_text = "\n".join(report)
    print(report_text)

    with open(output_path, 'w') as f:
        f.write(report_text)

    return report_text


def main():
    parser = argparse.ArgumentParser(
        description="评测生成图像与真值图像的深度和分割一致性"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--task", type=str, default="all",
        choices=["all", "depth", "segmentation", "seg", "sam"],
        help="评测任务: all(全部), depth(深度), segmentation/seg(语义分割), sam(SAM结构)"
    )
    parser.add_argument(
        "--no-vis", action="store_true",
        help="不保存可视化结果"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="输出目录 (覆盖配置文件中的设置)"
    )

    args = parser.parse_args()

    # 加载配置
    print(f"加载配置文件: {args.config}")
    config = load_config(args.config)

    # 覆盖输出目录
    if args.output_dir:
        config['output']['root'] = args.output_dir
        config['output']['depth_maps'] = os.path.join(args.output_dir, 'depth_maps')
        config['output']['seg_maps'] = os.path.join(args.output_dir, 'seg_maps')
        config['output']['metrics'] = os.path.join(args.output_dir, 'metrics')

    # 确保输出目录存在
    ensure_dir(config['output']['metrics'])

    # 运行评测
    depth_results = None
    seg_results = None
    sam_results = None
    save_vis = not args.no_vis

    if args.task in ["all", "depth"]:
        print("\n" + "=" * 70)
        print("开始深度一致性评测...")
        print("=" * 70)
        try:
            depth_results = run_depth_evaluation(config, save_vis=save_vis)
            # 保存深度结果
            depth_output = os.path.join(config['output']['metrics'], 'depth_results.json')
            with open(depth_output, 'w') as f:
                json.dump(convert_to_serializable(depth_results), f, indent=2)
            print(f"深度评测结果已保存到: {depth_output}")
        except Exception as e:
            print(f"深度评测出错: {e}")
            import traceback
            traceback.print_exc()

    if args.task in ["all", "segmentation", "seg"]:
        print("\n" + "=" * 70)
        print("开始语义分割一致性评测...")
        print("=" * 70)
        try:
            seg_results = run_segmentation_evaluation(config, save_vis=save_vis)
            # 保存分割结果
            seg_output = os.path.join(config['output']['metrics'], 'seg_results.json')
            with open(seg_output, 'w') as f:
                json.dump(convert_to_serializable(seg_results), f, indent=2)
            print(f"分割评测结果已保存到: {seg_output}")
        except Exception as e:
            print(f"分割评测出错: {e}")
            import traceback
            traceback.print_exc()

    if args.task in ["all", "sam"]:
        print("\n" + "=" * 70)
        print("开始SAM结构一致性评测...")
        print("=" * 70)
        try:
            sam_results = run_sam_evaluation(config, save_vis=save_vis)
            # 保存SAM结果
            sam_output = os.path.join(config['output']['metrics'], 'sam_results.json')
            with open(sam_output, 'w') as f:
                json.dump(convert_to_serializable(sam_results), f, indent=2)
            print(f"SAM评测结果已保存到: {sam_output}")
        except Exception as e:
            print(f"SAM评测出错: {e}")
            import traceback
            traceback.print_exc()

    # 生成综合报告
    if depth_results or seg_results or sam_results:
        report_path = os.path.join(config['output']['metrics'], 'evaluation_report.txt')
        generate_report(depth_results, seg_results, report_path, sam_results=sam_results)
        print(f"\n综合报告已保存到: {report_path}")

        # 保存完整JSON结果
        full_results = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'depth': depth_results,
            'segmentation': seg_results,
            'sam': sam_results
        }
        full_output = os.path.join(config['output']['metrics'], 'full_results.json')
        with open(full_output, 'w') as f:
            json.dump(convert_to_serializable(full_results), f, indent=2)
        print(f"完整结果已保存到: {full_output}")


if __name__ == "__main__":
    main()
