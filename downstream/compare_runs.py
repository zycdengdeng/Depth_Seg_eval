#!/usr/bin/env python3
"""
对比两次评测 run（baseline vs gsnet），输出 delta 表。

用途：回应审稿"只在 3DGS 内评估、没证 data reuse 价值"。
对每个数据集，把 Depth_Seg_eval 库各跑两次（baseline / gsnet，gt 完全相同），
本脚本读取两套结果的 overall 指标，并排打印，标出 gsnet 是否更优。

读取的文件（在每个 run 的 metrics 目录下）：
    depth_results.json   -> overall.{abs_rel, rmse, delta_1, pearson, ...}
    seg_results.json     -> overall.{consistency, miou, fwiou, ...}
    sam_results.json     -> overall.{edge_f1, edge_correlation, ...}

用法：
    # 单个数据集
    python compare_runs.py \
        --pair "CARLA CSE" \
            /mnt/zihanw/downstream_cse/baseline/results/metrics \
            /mnt/zihanw/downstream_cse/gsnet/results/metrics

    # 多个数据集一起出论文表
    python compare_runs.py \
        --pair "CARLA CSE" \
            /mnt/zihanw/downstream_cse/baseline/results/metrics \
            /mnt/zihanw/downstream_cse/gsnet/results/metrics \
        --pair "nuScenes SSE" \
            /mnt/zihanw/downstream_nusc/sfm/results/metrics \
            /mnt/zihanw/downstream_nusc/gsnet/results/metrics \
        --out comparison.md

    # 传 run root 也行（自动找其下的 metrics/ 子目录）
    python compare_runs.py --pair "CARLA CSE" \
        /mnt/zihanw/downstream_cse/baseline/results \
        /mnt/zihanw/downstream_cse/gsnet/results

可选：
    --per-group   额外打印每个 group(序列/clip) 的对比
    --out FILE    同时把 Markdown 表写到文件
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple


# ============== 指标定义 ==============
# (task, json_key, 显示名, 方向, 格式)
# 方向 'down' = 越低越好；'up' = 越高越好
HEADLINE_METRICS = [
    ("depth", "abs_rel",         "depth abs_rel",   "down", "{:.4f}"),
    ("depth", "rmse",            "depth rmse",      "down", "{:.4f}"),
    ("depth", "delta_1",         "depth δ<1.25 %",  "up",   "{:.2f}"),
    ("depth", "pearson",         "depth pearson",   "up",   "{:.4f}"),
    ("seg",   "consistency",     "seg consistency %", "up", "{:.2f}"),
    ("seg",   "miou",            "seg mIoU %",      "up",   "{:.2f}"),
    ("seg",   "coarse_miou",     "seg mIoU-7 %",    "up",   "{:.2f}"),
    ("seg",   "fwiou",           "seg fwIoU %",     "up",   "{:.2f}"),
    ("sam",   "edge_f1",         "SAM edge_f1 %",   "up",   "{:.2f}"),
    ("sam",   "edge_correlation","SAM edge_corr",   "up",   "{:.4f}"),
]

# 论文主表用的精简指标（section 5）。箭头会自动按 direction 加，不用写在名字里
PAPER_METRICS = [
    ("depth", "abs_rel",  "depth abs_rel", "down", "{:.4f}"),
    ("seg",   "miou",     "seg mIoU",      "up",   "{:.2f}"),
    ("sam",   "edge_f1",  "SAM edge_f1",   "up",   "{:.2f}"),
]

TASK_FILES = {
    "depth": "depth_results.json",
    "seg":   "seg_results.json",
    "sam":   "sam_results.json",
}


def resolve_metrics_dir(path: str) -> str:
    """传入 metrics 目录或 run root 都可以；自动定位含 *_results.json 的目录。"""
    if not os.path.isdir(path):
        raise FileNotFoundError(f"目录不存在: {path}")
    # 直接含结果文件
    if any(os.path.exists(os.path.join(path, f)) for f in TASK_FILES.values()):
        return path
    # 常见的 metrics 子目录
    cand = os.path.join(path, "metrics")
    if os.path.isdir(cand):
        return cand
    return path


def load_run(metrics_dir: str) -> Dict[str, dict]:
    """读取一个 run 的 depth/seg/sam 结果 JSON，返回 {task: full_dict}。缺失则跳过。"""
    metrics_dir = resolve_metrics_dir(metrics_dir)
    run = {}
    for task, fname in TASK_FILES.items():
        fpath = os.path.join(metrics_dir, fname)
        if os.path.exists(fpath):
            try:
                with open(fpath, "r") as f:
                    run[task] = json.load(f)
            except Exception as e:
                print(f"  [warn] 读取失败 {fpath}: {e}")
    if not run:
        print(f"  [warn] {metrics_dir} 下没找到任何 *_results.json")
    return run


def get_value(run: Dict[str, dict], task: str, key: str,
              group: str = "overall") -> Optional[float]:
    """取某 task 在 group(默认 overall) 下某指标的值。"""
    if task not in run:
        return None
    section = run[task].get(group)
    if not isinstance(section, dict):
        return None
    val = section.get(key)
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def is_better(direction: str, base: float, new: float) -> Optional[bool]:
    """gsnet(new) 相对 baseline(base) 是否更优。"""
    if base is None or new is None:
        return None
    if direction == "down":
        return new < base
    return new > base


def arrow(direction: str) -> str:
    """↓ = 越低越好；↑ = 越高越好。"""
    return "↓" if direction == "down" else "↑"


def rel_improve(direction: str, base: Optional[float],
                new: Optional[float]) -> Optional[float]:
    """gsnet 相对 baseline 的提升百分比（正数=更好，已按方向归一）。"""
    if base is None or new is None or base == 0:
        return None
    if direction == "down":
        return (base - new) / abs(base) * 100.0
    return (new - base) / abs(base) * 100.0


def fmt(v: Optional[float], pattern: str) -> str:
    return pattern.format(v) if v is not None else "—"


def list_groups(run_b: Dict[str, dict], run_g: Dict[str, dict]) -> List[str]:
    """收集两套结果里出现过的 group(排除 overall)，按出现顺序。"""
    seen = []
    for run in (run_b, run_g):
        for task_dict in run.values():
            for k in task_dict.keys():
                if k != "overall" and k not in seen:
                    seen.append(k)
    return seen


def compare_one(name: str, base_dir: str, gsnet_dir: str,
                metrics_spec, group: str = "overall") -> Tuple[List[str], int, int]:
    """对比一组，返回 (markdown行列表, gsnet更优数, 有效指标数)。"""
    run_b = load_run(base_dir)
    run_g = load_run(gsnet_dir)

    lines = []
    title = f"{name}" if group == "overall" else f"{name} — [{group}]"
    lines.append(f"### {title}")
    lines.append("")
    lines.append("> 箭头: ↑=越高越好, ↓=越低越好 ｜ 提升%: gsnet 相对 baseline，正数=更好")
    lines.append("")
    lines.append("| 指标 | baseline | gsnet | Δ | 提升% | 结果 |")
    lines.append("|---|---|---|---|---|---|")

    wins = 0
    valid = 0
    for task, key, disp, direction, pattern in metrics_spec:
        b = get_value(run_b, task, key, group)
        g = get_value(run_g, task, key, group)
        if b is None and g is None:
            continue
        better = is_better(direction, b, g)
        if better is not None:
            valid += 1
            if better:
                wins += 1
        delta = (g - b) if (b is not None and g is not None) else None
        delta_str = ("{:+.4f}".format(delta) if delta is not None else "—")
        imp = rel_improve(direction, b, g)
        imp_str = ("{:+.1f}%".format(imp) if imp is not None else "—")
        mark = "—" if better is None else ("✅ 更优" if better else "❌ 更差")
        name_with_arrow = f"{disp} {arrow(direction)}"
        lines.append(
            f"| {name_with_arrow} | {fmt(b, pattern)} | {fmt(g, pattern)} | "
            f"{delta_str} | {imp_str} | {mark} |"
        )

    lines.append("")
    if valid:
        avg_imp = None
        # 平均提升%（仅对两边都有值的指标）
        imps = [rel_improve(d, get_value(run_b, t, k, group), get_value(run_g, t, k, group))
                for t, k, _disp, d, _p in metrics_spec]
        imps = [x for x in imps if x is not None]
        if imps:
            avg_imp = sum(imps) / len(imps)
        summary = f"> **gsnet 在 {wins}/{valid} 个指标上更优**"
        if avg_imp is not None:
            summary += f"，平均提升 {avg_imp:+.1f}%"
        summary += "。"
        lines.append(summary)
        lines.append("")
    return lines, wins, valid


def build_paper_table(pairs_data: List[Tuple[str, str, str]]) -> List[str]:
    """汇总所有数据集的论文主表（section 5 风格）。"""
    cols = [f"{disp} {arrow(direction)}" for _, _, disp, direction, _ in PAPER_METRICS]
    lines = []
    lines.append("## 论文主表（下游感知，gsnet 更优的数加粗）")
    lines.append("> 表头箭头: ↑=越高越好, ↓=越低越好")
    lines.append("")
    header = "| 数据集 | init | " + " | ".join(cols) + " |"
    sep = "|---|---|" + "|".join(["---"] * len(cols)) + "|"
    lines.append(header)
    lines.append(sep)

    for name, base_dir, gsnet_dir in pairs_data:
        run_b = load_run(base_dir)
        run_g = load_run(gsnet_dir)
        b_cells, g_cells = [], []
        for task, key, _disp, direction, pattern in PAPER_METRICS:
            b = get_value(run_b, task, key)
            g = get_value(run_g, task, key)
            better = is_better(direction, b, g)
            b_str = fmt(b, pattern)
            g_str = fmt(g, pattern)
            # gsnet 更优则加粗
            if better is True:
                g_str = f"**{g_str}**"
            elif better is False:
                b_str = f"**{b_str}**"
            b_cells.append(b_str)
            g_cells.append(g_str)
        lines.append(f"| {name} | baseline | " + " | ".join(b_cells) + " |")
        lines.append(f"| {name} | gsnet | " + " | ".join(g_cells) + " |")
    lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="对比 baseline vs gsnet 两次下游评测，输出 delta 表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pair", action="append", nargs=3,
        metavar=("NAME", "BASELINE_DIR", "GSNET_DIR"),
        required=True,
        help="一组对比：名称 baseline结果目录 gsnet结果目录（可多次指定）",
    )
    parser.add_argument("--per-group", action="store_true",
                        help="额外打印每个 group(序列/clip) 的对比")
    parser.add_argument("--out", type=str, default=None,
                        help="把 Markdown 结果写到该文件")
    args = parser.parse_args()

    all_lines = []
    all_lines.append("# 下游感知评测对比 (baseline vs gsnet)")
    all_lines.append("")

    pairs_data = [(p[0], p[1], p[2]) for p in args.pair]

    # 各数据集详细对比
    for name, base_dir, gsnet_dir in pairs_data:
        all_lines.append("=" * 60)
        all_lines.append(f"## {name}")
        all_lines.append("")
        lines, wins, valid = compare_one(name, base_dir, gsnet_dir, HEADLINE_METRICS)
        all_lines.extend(lines)

        if args.per_group:
            run_b = load_run(base_dir)
            run_g = load_run(gsnet_dir)
            for grp in list_groups(run_b, run_g):
                glines, _, _ = compare_one(name, base_dir, gsnet_dir,
                                           HEADLINE_METRICS, group=grp)
                all_lines.extend(glines)

    # 论文主表
    all_lines.extend(build_paper_table(pairs_data))

    text = "\n".join(all_lines)
    print(text)

    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"\n[已写入] {args.out}")


if __name__ == "__main__":
    main()
