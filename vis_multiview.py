#!/usr/bin/env python3
"""
多视角一致性结果可视化
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

methods = ['GT', 'Ours', 'DDGS', 'DropGS', 'Co-Adapt', 'AD-GS', 'SparseGS', 'S3GS']
colors = ['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

# All-pairs summary data
mv_ssim =    [0.302, 0.347, 0.730, 0.748, 0.745, 0.709, 0.540, 0.303]
mv_psnr =    [15.8,  15.4,  19.2,  19.8,  21.4,  18.6,  16.7,  14.8]
depth_err =  [2.78,  2.34,  1.70,  1.74,  1.76,  1.74,  1.38,  1.52]
inlier_pct = [38.4,  29.1,  30.7,  48.9,  51.2,  29.4,  23.6,  18.3]
epi_med =    [6.6,   5.9,   34.9,  24.3,  16.6,  35.6,  30.5,  35.0]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Multi-View Consistency Metrics (All Pairs)', fontsize=16, fontweight='bold', y=0.98)

# 1. MV-SSIM
ax = axes[0, 0]
bars = ax.bar(methods, mv_ssim, color=colors, edgecolor='black', linewidth=0.5)
ax.set_title('MV-SSIM (higher = better)', fontsize=12, fontweight='bold')
ax.set_ylabel('MV-SSIM')
ax.set_ylim(0, 0.85)
for bar, val in zip(bars, mv_ssim):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(y=mv_ssim[0], color='#2ca02c', linestyle='--', alpha=0.5, label='GT level')
ax.tick_params(axis='x', rotation=30)

# 2. MV-PSNR
ax = axes[0, 1]
bars = ax.bar(methods, mv_psnr, color=colors, edgecolor='black', linewidth=0.5)
ax.set_title('MV-PSNR (higher = better)', fontsize=12, fontweight='bold')
ax.set_ylabel('dB')
ax.set_ylim(10, 24)
for bar, val in zip(bars, mv_psnr):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{val:.1f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(y=mv_psnr[0], color='#2ca02c', linestyle='--', alpha=0.5)
ax.tick_params(axis='x', rotation=30)

# 3. Depth Error
ax = axes[0, 2]
bars = ax.bar(methods, depth_err, color=colors, edgecolor='black', linewidth=0.5)
ax.set_title('Depth Consistency Error (lower = better)', fontsize=12, fontweight='bold')
ax.set_ylabel('Relative Error')
ax.set_ylim(0, 3.2)
for bar, val in zip(bars, depth_err):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, f'{val:.2f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(y=depth_err[0], color='#2ca02c', linestyle='--', alpha=0.5)
ax.tick_params(axis='x', rotation=30)

# 4. LoFTR Inlier %
ax = axes[1, 0]
bars = ax.bar(methods, inlier_pct, color=colors, edgecolor='black', linewidth=0.5)
ax.set_title('LoFTR Inlier % (higher = better)', fontsize=12, fontweight='bold')
ax.set_ylabel('%')
ax.set_ylim(0, 60)
for bar, val in zip(bars, inlier_pct):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(y=inlier_pct[0], color='#2ca02c', linestyle='--', alpha=0.5)
ax.tick_params(axis='x', rotation=30)

# 5. Epipolar Error (median)
ax = axes[1, 1]
bars = ax.bar(methods, epi_med, color=colors, edgecolor='black', linewidth=0.5)
ax.set_title('Median Epipolar Error (lower = better)', fontsize=12, fontweight='bold')
ax.set_ylabel('pixels')
ax.set_ylim(0, 42)
for bar, val in zip(bars, epi_med):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(y=epi_med[0], color='#2ca02c', linestyle='--', alpha=0.5)
ax.tick_params(axis='x', rotation=30)

# 6. Radar chart
ax = axes[1, 2]
ax.set_visible(False)
ax_radar = fig.add_subplot(2, 3, 6, polar=True)

categories = ['MV-SSIM', 'MV-PSNR', 'Depth\nConsist.', 'Inlier%', 'Epi\nAccuracy']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

def normalize(vals, higher_better=True):
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return [0.5] * len(vals)
    if higher_better:
        return [(v - mn) / (mx - mn) for v in vals]
    else:
        return [(mx - v) / (mx - mn) for v in vals]

for idx, (method, color) in enumerate(zip(methods, colors)):
    if method in ['SparseGS', 'S3GS', 'AD-GS']:
        continue
    vals = [
        normalize(mv_ssim, True)[idx],
        normalize(mv_psnr, True)[idx],
        normalize(depth_err, False)[idx],
        normalize(inlier_pct, True)[idx],
        normalize(epi_med, False)[idx],
    ]
    vals += [vals[0]]
    ax_radar.plot(angles, vals, 'o-', linewidth=1.5, color=color, label=method, markersize=4)
    ax_radar.fill(angles, vals, alpha=0.08, color=color)

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(categories, fontsize=8)
ax_radar.set_ylim(0, 1.1)
ax_radar.set_title('Normalized Comparison\n(GT, Ours, Top-3 GS)', fontsize=11, fontweight='bold', pad=20)
ax_radar.legend(loc='lower right', bbox_to_anchor=(1.35, -0.1), fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('/home/user/Depth_Seg_eval/multiview_metrics_vis.png', dpi=150, bbox_inches='tight')
print('Saved: multiview_metrics_vis.png')
