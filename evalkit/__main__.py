"""
用法：
    # 1. 测试集清单（交给各方法渲染；含评测分辨率下的相机内参）
    python -m evalkit render-list --out ./results/render_list.json

    # 2. manifest：各方法输出 ↔ GT
    python -m evalkit manifest --render-list ./results/render_list.json \\
        --gs DDGS=/path/to/ddgs_renders --gs SparseGS=/path/... \\
        --ours Ours=/mnt/zihanw/tf2.5_verion2_test_evaluation \\
        --out ./results/manifest.csv

    # 3. 评测
    python -m evalkit run --manifest ./results/manifest.csv --out ./results/evalkit \\
        --gt-undistort tianjin --device cuda:0

    # 小批量：推理结果目录 vs GT 目录（按相对路径/文件名配对，扩展名可不同）
    python -m evalkit pairs --gen MyMethod=/data/pred --gt /data/gt --out ./results/manifest.csv
    python -m evalkit run --manifest ./results/manifest.csv --out ./results/eval

    # 新服务器环境/权重检查
    python -m evalkit check --device cuda:0

    # 回归验证：旧数据公共帧（应与 gs_eval/tf_eval --common 结果一致）
    python -m evalkit legacy-common --out ./results/manifest_common.csv
"""

import argparse
import os
import sys

# 仓库根目录（复用 metrics / depth_eval / seg_eval 等模块）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .scorers import ALL_GROUPS  # noqa: E402


def _parse_named(items):
    named = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"格式应为 名称=路径: {item}")
        name, path = item.split("=", 1)
        named[name] = path
    return named


def main():
    parser = argparse.ArgumentParser(prog="python -m evalkit", description="生成图 vs 真值图统一评测")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("render-list", help="生成测试集清单（以 Ours 测试集为准）")
    p.add_argument("--out", required=True)
    p.add_argument("--clips", nargs="+", default=None, help="clip 编号，默认 Ours 全部测试 clip")
    p.add_argument("--cameras", nargs="+", default=None)

    p = sub.add_parser("manifest", help="由测试集清单和各方法输出生成 manifest")
    p.add_argument("--render-list", required=True)
    p.add_argument("--gs", action="append", help="GS 类方法：名称=根目录（可重复）")
    p.add_argument("--ours", default=None, help="视频方法：名称=视频根目录")
    p.add_argument("--out", required=True)

    p = sub.add_parser("pairs", help="推理结果目录 vs GT 目录，按相对路径配对生成 manifest")
    p.add_argument("--gen", action="append", required=True, help="方法名=推理结果目录（可重复）")
    p.add_argument("--gt", required=True, help="GT 目录")
    p.add_argument("--clip-depth", type=int, default=1,
                   help="相对路径前几级目录作为 clip（默认1）；0 表示每个样本独立")
    p.add_argument("--out", required=True)

    p = sub.add_parser("check", help="检查依赖、GPU 与模型权重（无网络时提示需拷贝的缓存）")
    p.add_argument("--device", default="cuda")
    p.add_argument("--metrics", nargs="+", default=list(ALL_GROUPS), choices=ALL_GROUPS)

    p = sub.add_parser("legacy-common", help="旧数据公共帧 manifest（回归验证用）")
    p.add_argument("--out", required=True)

    p = sub.add_parser("run", help="按 manifest 计算指标")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--metrics", nargs="+", default=list(ALL_GROUPS), choices=ALL_GROUPS)
    p.add_argument("--methods", nargs="+", default=None, help="只评测这些方法")
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-size", nargs=2, type=int, default=[1280, 720], metavar=("W", "H"))
    p.add_argument("--gt-undistort", default="none", choices=["none", "tianjin"])
    p.add_argument("--max-depth-ratio", type=float, default=100.0)
    p.add_argument("--allow-partial", action="store_true",
                   help="不要求各方法样本一致（结果不可直接比较）")
    p.add_argument("--keep-images", action="store_true", help="保留 KID/FID 用的中间图像")
    p.add_argument("--force-dist", action="store_true",
                   help="样本数不足时也计算 KID/FID（默认少于 50 个样本时跳过）")

    args = parser.parse_args()

    if args.cmd == "render-list":
        from .build import build_render_list, save_render_list
        save_render_list(build_render_list(args.clips, args.cameras), args.out)

    elif args.cmd == "manifest":
        import json
        from .build import build_manifest
        from .manifest import write_manifest
        with open(args.render_list) as f:
            render_list = json.load(f)
        ours = next(iter(_parse_named([args.ours]).items())) if args.ours else None
        rows = build_manifest(render_list, _parse_named(args.gs), ours)
        write_manifest(rows, args.out)
        print(f"已保存: {args.out}（{len(rows)} 行）")

    elif args.cmd == "pairs":
        from .build import build_pairs_manifest
        from .manifest import write_manifest
        rows = build_pairs_manifest(_parse_named(args.gen), args.gt, clip_depth=args.clip_depth)
        write_manifest(rows, args.out)
        print(f"已保存: {args.out}（{len(rows)} 行）")

    elif args.cmd == "check":
        from .check import check_environment
        sys.exit(0 if check_environment(args.device, args.metrics) else 1)

    elif args.cmd == "legacy-common":
        from .build import build_legacy_common_manifest
        from .manifest import write_manifest
        rows = build_legacy_common_manifest()
        write_manifest(rows, args.out)
        print(f"已保存: {args.out}（{len(rows)} 行）")

    elif args.cmd == "run":
        from .runner import run
        run(args.manifest, args.out, args.metrics, device=args.device,
            eval_size=tuple(args.eval_size), gt_undistort=args.gt_undistort,
            max_depth_ratio=args.max_depth_ratio, allow_partial=args.allow_partial,
            keep_images=args.keep_images, methods=args.methods, force_dist=args.force_dist)


if __name__ == "__main__":
    main()
