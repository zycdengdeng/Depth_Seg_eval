"""
manifest：评测样本清单（CSV）

每行一个 (方法, 样本)：
    sample_id  样本唯一标识，跨方法相同（同一 sample_id 必须对应同一张 GT）
    method     方法名
    clip       场景/片段名（bootstrap 置信区间按 clip 重采样）
    camera     相机名（GT 去畸变时使用；不需要可留空）
    timestamp  时间戳（仅记录用，可留空）
    group      分组标签（如 near/middle/far，可留空）
    gen        生成结果：图像路径，或视频帧 "path/to/video.mp4#帧序号"
    gt         真值图像路径
"""

import csv
from collections import defaultdict
from typing import Dict, List

FIELDS = ["sample_id", "method", "clip", "camera", "timestamp", "group", "gen", "gt"]


def make_sample_id(clip: str, camera: str, timestamp) -> str:
    return f"{clip}|{camera}|{timestamp}"


def write_manifest(rows: List[Dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def read_manifest(path: str) -> List[Dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    missing = [k for k in ("sample_id", "method", "gen", "gt") if rows and k not in rows[0]]
    if missing:
        raise ValueError(f"manifest 缺少列: {missing}")
    for row in rows:
        for k in FIELDS:
            row.setdefault(k, "")
        if not row["clip"]:
            row["clip"] = row["sample_id"]
    return rows


def check_alignment(rows: List[Dict]) -> Dict[str, Dict]:
    """
    检查各方法的样本集合与 GT 是否一致

    Returns:
        {"common": 所有方法共有的 sample_id 集合,
         "per_method": {method: {"n": 样本数, "missing": 缺少的共有外样本数}}}
    Raises:
        ValueError: 同一 sample_id 在不同方法中对应不同 GT
    """
    by_method = defaultdict(set)
    gt_of_sample = {}
    for row in rows:
        by_method[row["method"]].add(row["sample_id"])
        prev = gt_of_sample.setdefault(row["sample_id"], row["gt"])
        if prev != row["gt"]:
            raise ValueError(f"样本 {row['sample_id']} 的 GT 不一致: {prev} vs {row['gt']}")

    all_samples = set(gt_of_sample)
    common = set.intersection(*by_method.values()) if by_method else set()
    per_method = {m: {"n": len(s), "missing": len(all_samples - s)}
                  for m, s in sorted(by_method.items())}
    return {"common": common, "per_method": per_method}
