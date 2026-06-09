"""
GPU 自动选卡工具

共用服务器时，开跑前查 nvidia-smi，只挑显存空闲足够的卡，
别人占满的卡自动跳过，不去挤。

- select_free_gpus(): 返回可用 GPU 的 torch 序号列表（已处理 CUDA_VISIBLE_DEVICES 重映射）
- 评测每个 worker 大致需要 10~20GB 显存（Depth-Anything-Large / Mask2Former-swin-L / SAM-large），
  默认要求单卡至少 20GB 空闲、且已用显存不超过总量的一半。
"""

import os
import subprocess
from typing import List, Optional, Dict


def query_gpus() -> Optional[List[Dict]]:
    """
    用 nvidia-smi 查询每张卡的显存/利用率。
    返回 [{index, mem_used, mem_total, mem_free, util}, ...]（index 为物理卡号）。
    查询失败返回 None。
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
            timeout=15,
        )
    except Exception as e:
        print(f"[gpu_utils] nvidia-smi 查询失败: {e}")
        return None

    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append({
                "index": int(parts[0]),
                "mem_used": int(parts[1]),
                "mem_total": int(parts[2]),
                "mem_free": int(parts[3]),
                "util": int(parts[4]),
            })
        except ValueError:
            continue
    return gpus or None


def _apply_visible_devices(gpus: List[Dict]) -> List[Dict]:
    """
    处理 CUDA_VISIBLE_DEVICES：把物理卡号映射成 torch 看到的序号。
    例如 CUDA_VISIBLE_DEVICES=2,3,5 时，物理 2/3/5 对应 torch cuda:0/1/2。
    未设置则 torch 序号==物理卡号。
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.strip() == "":
        for g in gpus:
            g["torch_index"] = g["index"]
        return gpus

    try:
        vis_ids = [int(x) for x in visible.split(",") if x.strip() != ""]
    except ValueError:
        for g in gpus:
            g["torch_index"] = g["index"]
        return gpus

    phys_to_ord = {p: i for i, p in enumerate(vis_ids)}
    filtered = []
    for g in gpus:
        if g["index"] in phys_to_ord:
            g["torch_index"] = phys_to_ord[g["index"]]
            filtered.append(g)
    return filtered


def select_free_gpus(min_free_mb: int = 20000,
                     max_used_frac: float = 0.5,
                     max_gpus: Optional[int] = None,
                     verbose: bool = True) -> List[int]:
    """
    选出空闲足够的 GPU（torch 序号），按空闲显存从多到少排序。

    Args:
        min_free_mb: 单卡至少需要的空闲显存(MB)，低于此视为被占用、跳过
        max_used_frac: 已用显存占比超过此值视为被占用、跳过
        max_gpus: 最多用几张卡（None=不限）
        verbose: 打印每张卡状态

    Returns:
        可用 GPU 的 torch 序号列表。若没有满足条件的卡，则退而使用最空的一张。
    """
    gpus = query_gpus()
    if not gpus:
        # 回退：用 torch 报告的全部卡
        try:
            import torch
            ids = list(range(torch.cuda.device_count())) or [0]
        except Exception:
            ids = [0]
        if verbose:
            print(f"[gpu_utils] 无法查询 nvidia-smi，回退使用: {ids}")
        return ids

    gpus = _apply_visible_devices(gpus)

    free = [
        g for g in gpus
        if g["mem_free"] >= min_free_mb
        and (g["mem_used"] / max(g["mem_total"], 1)) <= max_used_frac
    ]
    free.sort(key=lambda g: g["mem_free"], reverse=True)
    ids = [g["torch_index"] for g in free]
    if max_gpus is not None:
        ids = ids[:max_gpus]

    if verbose:
        print(f"[gpu_utils] 选卡条件: 空闲显存 ≥ {min_free_mb}MB 且 已用 ≤ {int(max_used_frac*100)}%")
        for g in sorted(gpus, key=lambda x: x["torch_index"]):
            chosen = g["torch_index"] in ids
            tag = "✓ 选用" if chosen else "· 跳过(占用)"
            print(f"  cuda:{g['torch_index']} (物理GPU{g['index']}): "
                  f"已用 {g['mem_used']}/{g['mem_total']}MB, 空闲 {g['mem_free']}MB, "
                  f"util {g['util']}%  {tag}")

    if not ids:
        gpus.sort(key=lambda g: g["mem_free"], reverse=True)
        ids = [gpus[0]["torch_index"]]
        if verbose:
            print(f"[gpu_utils] 无满足条件的空闲卡，退而使用最空的 cuda:{ids[0]}")
    elif verbose:
        print(f"[gpu_utils] 最终选用: {ids}")

    return ids


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="查看并选出空闲 GPU")
    p.add_argument("--min-free-mem", type=int, default=20000, help="单卡最低空闲显存(MB)")
    p.add_argument("--max-used-frac", type=float, default=0.5, help="已用显存占比上限")
    p.add_argument("--max-gpus", type=int, default=None, help="最多用几张卡")
    args = p.parse_args()
    select_free_gpus(min_free_mb=args.min_free_mem,
                     max_used_frac=args.max_used_frac,
                     max_gpus=args.max_gpus)
