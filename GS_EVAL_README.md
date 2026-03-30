# gs_eval.py - GS 方法统一评测脚本

对多个 3D Gaussian Splatting 方法的渲染结果进行统一评测，与 GT 车端图像对比。

## 支持的方法

| 方法 | 路径 | 目录结构 |
|---|---|---|
| DDGS | `/mnt/myn/project/DDGS/output` | `{clip}/{ts}/vehicle_renders/{cam}/render.png` |
| DropGaussian | `/mnt/myn/project/DropGaussian_release/output` | 同上 |
| Co-Adaptation | `/mnt/myn/project/Co-Adaptation-of-3DGS/output` | 同上 |
| AD-GS | `/mnt/myn/project/AD-GS/output` | 同上 |
| SparseGS | `/mnt/zyc_wzh/SparseGS/output/car_road` | `scene{NNN}_{dist}/vehicle_renders/{cam}/render.png` |
| S3Gaussian | `/mnt/zyc_wzh/S3Gaussian/work_dirs/roadside_colmap/renders` | `scene{NNN}_{dist}/{cam}.png` |

GT 数据：`/mnt/car_road_data_TianJin/{clip}/car/images/{cam}/addc_*_{timestamp}.jpg`

## 评测指标

| 任务 | 指标 | 说明 |
|---|---|---|
| `image_metrics` | PSNR, SSIM, LPIPS | 图像质量（需要GT） |
| `depth` | abs_rel, rmse, delta_1/2/3, pearson, spearman | 深度一致性 |
| `seg` | mIoU, pixel_acc, consistency | 语义分割一致性 |
| `nta` | NTA-IoU, precision, recall | 交通参与者检测一致性 (YOLO11) |
| `ntl` | NTL-IoU, NTL-F1, DA-IoU | 车道线检测一致性 (TwinLiteNet) |

## 基本用法

```bash
# 全部方法、全部指标
python gs_eval.py --task all

# 只跑图像质量指标
python gs_eval.py --task image_metrics

# 只跑 NTA + NTL
python gs_eval.py --task nta
python gs_eval.py --task ntl
```

## 筛选范围

```bash
# 指定方法
python gs_eval.py --task all --methods DDGS DropGaussian AD-GS

# 指定 clip
python gs_eval.py --task image_metrics --clips 003_car0325_road0327_t3 004_car0325_road0327_t4

# 指定相机
python gs_eval.py --task depth --cameras FL FN FW

# 指定距离
python gs_eval.py --task nta --distances near far

# 组合筛选
python gs_eval.py --task image_metrics --methods DDGS --cameras FL --distances near
```

## 其他参数

```bash
# 指定 GPU
python gs_eval.py --task all --device cuda:1

# 自定义输出目录
python gs_eval.py --task all --output-dir ./results/gs_eval_run2

# 自定义 GT 路径
python gs_eval.py --task all --gt-root /path/to/gt_data

# GT 时间戳匹配容差（默认 100ms）
python gs_eval.py --task all --max-time-diff-ms 200
```

## 输出结构

```
results/gs_eval/
  image_metrics_results.json   # 图像质量结果
  depth_results.json           # 深度一致性结果
  seg_results.json             # 分割一致性结果
  nta_results.json             # NTA-IoU 结果
  ntl_results.json             # NTL-IoU 结果
  full_results.json            # 全部结果汇总
```

每个 JSON 按方法组织，内含多维度汇总：

```json
{
  "DDGS": {
    "overall": {"psnr": 25.3, "ssim": 0.85, ...},
    "by_camera": {"FL": {...}, "FN": {...}, ...},
    "by_distance": {"near": {...}, "middle": {...}, "far": {...}},
    "by_clip": {"003_car0325_road0327_t3": {...}, ...}
  },
  "DropGaussian": { ... },
  ...
}
```

## 数据规模

- 18 个 clip x 3 个距离 x 7 个相机 = **378 帧/方法**
- 6 个方法共 **2268 帧**

## 可用的 clip 列表

```
003_car0325_road0327_t3    015_car0325_road0327_t18   035_car0402_road0402_t13
004_car0325_road0327_t4    020_car0325_road0327_t25   039_car0402_road0402_t17
009_car0325_road0327_t10   031_car0402_road0402_t9    050_car0402_road0402_t28
055_car0402_road0402_t33   059_car0402_road0402_t37   076_car0402_road0402_t54
056_car0402_road0402_t34   063_car0402_road0402_t41   082_car0402_road0402_t64
085_car0402_road0402_t67   086_car0402_road0402_t68   088_car0402_road0402_t70
```

## 相机说明

| 缩写 | 含义 |
|---|---|
| FL | Front-Left |
| FN | Front-Narrow (前方中央) |
| FR | Front-Right |
| FW | Front-Wide (前方广角) |
| RL | Rear-Left |
| RN | Rear-Narrow (后方中央) |
| RR | Rear-Right |
