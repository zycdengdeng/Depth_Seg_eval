"""
汇总与报告：均值 + clip 级 bootstrap 95% 置信区间，按分组拆分，输出 JSON / Markdown / LaTeX
"""

import json
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

# (key, 表头, 越高越好, 小数位, 显示倍率)
METRICS = [
    ("psnr",        "PSNR↑",            True,  2, 1),
    ("ssim",        "SSIM↑",            True,  3, 1),
    ("lpips",       "LPIPS↓",           False, 3, 1),
    ("kid",         "KID×10³↓",         False, 2, 1e3),
    ("fid",         "FID↓",             False, 1, 1),
    ("delta_1",     "δ1(%)↑",           True,  1, 1),
    ("abs_rel",     "AbsRel↓",          False, 3, 1),
    ("seg_cons",    "Seg-Cons.(%)↑",    True,  1, 1),
    ("coarse_miou", "Coarse-mIoU(%)↑",  True,  1, 1),
]
PER_SAMPLE_METRICS = ("psnr", "ssim", "lpips", "delta_1", "abs_rel", "seg_cons")

N_BOOTSTRAP = 1000
CI_LEVEL = 0.95


def _coarse_miou(conf_matrix: np.ndarray) -> float:
    from metrics import segmentation_metrics_from_confusion
    return float(segmentation_metrics_from_confusion(conf_matrix)["miou"])


def summarize(records: List[Dict], conf: Dict[str, np.ndarray],
              seed: int = 0) -> Dict:
    """
    Args:
        records: 一个方法（或一个分组）的逐样本记录，含 clip 和各指标
        conf: {sample_id: coarse 混淆矩阵}

    Returns:
        {"n_samples", "n_clips", metric: {"mean", "ci_low", "ci_high"}, ...}
    """
    clips = sorted({r["clip"] for r in records})
    clip_index = {c: i for i, c in enumerate(clips)}
    n_clips = len(clips)
    out = {"n_samples": len(records), "n_clips": n_clips}
    if not records:
        return out

    rng = np.random.default_rng(seed)
    boot = rng.integers(0, n_clips, size=(N_BOOTSTRAP, n_clips))
    # 每次重采样中每个 clip 被抽中的次数
    weights = np.zeros((N_BOOTSTRAP, n_clips))
    for b in range(N_BOOTSTRAP):
        np.add.at(weights[b], boot[b], 1)
    alpha = (1 - CI_LEVEL) / 2

    for key in PER_SAMPLE_METRICS:
        sums, counts = np.zeros(n_clips), np.zeros(n_clips)
        for r in records:
            v = r.get(key)
            if v is None or v == "" or np.isnan(float(v)):
                continue
            sums[clip_index[r["clip"]]] += float(v)
            counts[clip_index[r["clip"]]] += 1
        if counts.sum() == 0:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            boot_means = (weights @ sums) / (weights @ counts)
        out[key] = {"mean": float(sums.sum() / counts.sum()),
                    "ci_low": float(np.nanquantile(boot_means, alpha)),
                    "ci_high": float(np.nanquantile(boot_means, 1 - alpha))}

    sample_ids = [r["sample_id"] for r in records if r["sample_id"] in conf]
    if sample_ids:
        shape = conf[sample_ids[0]].shape
        clip_cm = np.zeros((n_clips,) + shape, dtype=np.int64)
        for r in records:
            if r["sample_id"] in conf:
                clip_cm[clip_index[r["clip"]]] += conf[r["sample_id"]]
        boot_vals = [_coarse_miou(np.tensordot(weights[b], clip_cm, axes=1))
                     for b in range(N_BOOTSTRAP)]
        out["coarse_miou"] = {"mean": _coarse_miou(clip_cm.sum(axis=0)),
                              "ci_low": float(np.nanquantile(boot_vals, alpha)),
                              "ci_high": float(np.nanquantile(boot_vals, 1 - alpha))}
    return out


def build_summary(records: List[Dict], conf: Dict[str, Dict[str, np.ndarray]],
                  dist: Dict[str, Dict[str, float]]) -> Dict:
    """
    Args:
        records: 全部逐样本记录（含 method / group）
        conf: {method: {sample_id: coarse 混淆矩阵}}
        dist: {method: {"kid":..., "fid":...}}
    """
    by_method = defaultdict(list)
    for r in records:
        by_method[r["method"]].append(r)

    summary = {}
    for method, recs in by_method.items():
        s = summarize(recs, conf.get(method, {}))
        for key, value in dist.get(method, {}).items():
            s[key] = {"mean": value}  # 集合级指标，无置信区间
        groups = sorted({r["group"] for r in recs if r.get("group")})
        if groups:
            s["by_group"] = {g: summarize([r for r in recs if r["group"] == g],
                                          conf.get(method, {}))
                             for g in groups}
        summary[method] = s
    return summary


def format_table(summary: Dict, fmt: str = "markdown", with_ci: bool = True,
                 methods: Optional[List[str]] = None) -> str:
    """主表：每个方法一行，最优值加粗"""
    methods = methods or list(summary)
    cols = [m for m in METRICS if any(m[0] in summary[x] for x in methods)]

    best = {}
    for key, _, higher, _, _ in cols:
        vals = {x: summary[x][key]["mean"] for x in methods
                if key in summary[x] and not np.isnan(summary[x][key]["mean"])}
        if vals:
            best[key] = (max if higher else min)(vals, key=vals.get)

    def cell(method, key, prec, scale):
        if key not in summary[method]:
            return "N/A"
        s = summary[method][key]
        text = f"{s['mean'] * scale:.{prec}f}"
        if with_ci and "ci_low" in s:
            half = (s["ci_high"] - s["ci_low"]) * scale / 2
            text += f" ± {half:.{prec}f}" if fmt == "markdown" else f" {{\\scriptsize$\\pm${half:.{prec}f}}}"
        if best.get(key) == method:
            text = f"**{text}**" if fmt == "markdown" else f"\\textbf{{{text}}}"
        return text

    if fmt == "markdown":
        lines = ["| 方法 | N | " + " | ".join(c[1] for c in cols) + " |",
                 "|---|---|" + "---|" * len(cols)]
        for x in methods:
            lines.append(f"| {x} | {summary[x]['n_samples']} | "
                         + " | ".join(cell(x, k, p, sc) for k, _, _, p, sc in cols) + " |")
    else:
        lines = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
                 "Method & " + " & ".join(c[1] for c in cols) + " \\\\", "\\midrule"]
        for x in methods:
            lines.append(f"{x} & " + " & ".join(cell(x, k, p, sc) for k, _, _, p, sc in cols) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def save_summary(summary: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
