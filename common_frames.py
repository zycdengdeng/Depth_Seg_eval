"""
公共评测帧：GS 方法与 Ours 使用完全相同的 (clip, 路侧时间戳, 相机, GT帧)

GS 方法只在每个 clip 的 near/middle/far 三个路侧时间戳渲染；
Ours 是覆盖一段时间窗口的 29 帧视频。
两者重合的 clip 中，只有落在 Ours 时间窗口内、且 Ours 有帧与之对齐的时刻才可比。

用法：
    python gs_eval.py --common ...
    python tf_eval.py --common ...
    python common_frames.py          # 打印公共帧清单
"""

from typing import Dict, List

# Ours 帧与 GS 路侧时间戳的最大允许时间差（毫秒）
MAX_FRAME_DIFF_MS = 50


def get_common_targets() -> List[Dict]:
    """
    Returns:
        [{"clip": "031_car0402_road0402_t9", "clip_num": "031",
          "distance": "middle", "road_ts": 1743572675860}, ...]
    """
    from gs_eval import CLIP_TIMESTAMPS
    from tf_eval import CLIP_TS_RANGES, FRAMES_PER_VIDEO, compute_frame_timestamps

    targets = []
    for clip_full, dists in CLIP_TIMESTAMPS.items():
        clip_num = clip_full.split("_")[0]
        if clip_num not in CLIP_TS_RANGES:
            continue
        frame_ts = compute_frame_timestamps(clip_num, FRAMES_PER_VIDEO)
        for dist, ts in dists.items():
            ts = int(ts)
            if min(abs(t - ts) for t in frame_ts) <= MAX_FRAME_DIFF_MS:
                targets.append({"clip": clip_full, "clip_num": clip_num,
                                "distance": dist, "road_ts": ts})
    return targets


if __name__ == "__main__":
    targets = get_common_targets()
    for t in targets:
        print(f"  {t['clip']:<28} {t['distance']:<7} {t['road_ts']}")
    print(f"共 {len(targets)} 个公共时刻 × 7 相机")
