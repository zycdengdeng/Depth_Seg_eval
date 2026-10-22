#!/usr/bin/env python3
"""
公共帧可视化：把评测时实际使用的图像拼成对比图（GT | 各方法），用于人工核查

图像经过与评测完全相同的预处理（gen resize 到 1280x720，GT 去畸变 + resize），
所以看到的就是指标计算时的输入。

用法：
    python common_vis.py --output-dir ./results/common_vis
    python common_vis.py --cameras FW FL
"""

import os
import argparse

import numpy as np
from PIL import Image, ImageDraw

TILE_SIZE = (640, 360)


def label_tile(img: np.ndarray, text: str) -> Image.Image:
    tile = Image.fromarray(img).resize(TILE_SIZE, Image.BILINEAR)
    draw = ImageDraw.Draw(tile)
    draw.rectangle([0, 0, 8 * len(text) + 10, 22], fill=(0, 0, 0))
    draw.text((5, 5), text, fill=(255, 255, 0))
    return tile


def main():
    parser = argparse.ArgumentParser(description="公共帧对比图")
    parser.add_argument("--output-dir", default="./results/common_vis")
    parser.add_argument("--cameras", nargs="+", default=None,
                        help="只输出这些相机（默认全部）")
    args = parser.parse_args()

    from sensitivity_eval import collect_all_pairs

    all_pairs = collect_all_pairs()
    methods = list(all_pairs.keys())

    # by_key[(clip, camera)][method] = loader
    by_key = {}
    for method, items in all_pairs.items():
        for it in items:
            by_key.setdefault(it["key"], {})[method] = it["load"]

    os.makedirs(args.output_dir, exist_ok=True)
    for (clip, camera), loaders in sorted(by_key.items()):
        if args.cameras and camera not in args.cameras:
            continue
        tiles, gt_tile = [], None
        for method in methods:
            if method not in loaders:
                tiles.append(label_tile(np.zeros((720, 1280, 3), np.uint8), f"{method} (missing)"))
                continue
            gen_img, gt_img = loaders[method]()
            if gt_tile is None:
                gt_tile = label_tile(gt_img, "GT")
            tiles.append(label_tile(gen_img, method))
        tiles = [gt_tile] + tiles

        cols = 4
        rows = (len(tiles) + cols - 1) // cols
        canvas = Image.new("RGB", (cols * TILE_SIZE[0], rows * TILE_SIZE[1]))
        for i, tile in enumerate(tiles):
            canvas.paste(tile, ((i % cols) * TILE_SIZE[0], (i // cols) * TILE_SIZE[1]))

        out_path = os.path.join(args.output_dir, f"{clip.split('_')[0]}_{camera}.jpg")
        canvas.save(out_path, quality=85)
        print(f"  已保存: {out_path}")


if __name__ == "__main__":
    main()
