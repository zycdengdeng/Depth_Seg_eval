# evalkit：生成图 vs 真值图统一评测

所有方法、所有任务共用同一套指标与流程。任何任务只要产出 **manifest**（gen 与 gt 一一对应的 CSV），即可直接评测。

## 快速开始（小批量：推理结果目录 vs 真值目录）

```bash
git clone <本仓库> && cd Depth_Seg_eval
git checkout claude/determined-pascal-6i0j27
pip install -r evalkit/requirements.txt

# 1. 检查环境、GPU、模型权重（首次有网络时会自动下载权重）
python -m evalkit check --device cuda:0

# 2. 推理结果与真值按相对路径配对（扩展名可不同，如 pred/a/001.png ↔ gt/a/001.jpg）
python -m evalkit pairs --gen MyMethod=/data/pred --gt /data/gt --out ./results/manifest.csv
#    多个方法：重复 --gen A=/data/predA --gen B=/data/predB
#    --clip-depth 1（默认）：第一级子目录视为一个场景，置信区间按场景重采样；
#    --clip-depth 0：每个样本独立（没有场景划分时使用）

# 3. 评测
python -m evalkit run --manifest ./results/manifest.csv --out ./results/eval --device cuda:0
#    默认评测分辨率 1280x720；其他分辨率用 --eval-size W H
#    样本少于 50 个时自动跳过 KID/FID（不可靠）
```

结果：`results/eval/table.md`（主表）、`summary.json`（均值 + 95% CI）、`per_sample.csv`（逐样本）。

## 环境与模型权重

需要整个仓库（evalkit 复用根目录的 `metrics.py`、`depth_eval.py`、`seg_eval.py`、`image_metrics_eval.py`、`utils.py`、`undistort.py`）。

| 权重 | 大小 | 缓存位置 |
|---|---|---|
| depth-anything/Depth-Anything-V2-Large-hf | ~1.3 GB | `~/.cache/huggingface/hub/` |
| facebook/mask2former-swin-large-cityscapes-semantic | ~0.9 GB | `~/.cache/huggingface/hub/` |
| AlexNet（LPIPS 主干，torchvision） | ~233 MB | `~/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth` |
| LPIPS 线性层 | 很小 | 随 `pip install lpips` 安装 |
| Inception（clean-fid，KID/FID） | ~92 MB | `/tmp/inception-2015-12-05.pt` |

**新服务器能联网**：不用拷任何东西，`check` 时自动下载。

**新服务器不能联网**：从已跑通的服务器拷贝缓存，再离线运行：

```bash
# 在旧服务器上打包
tar czf eval_weights.tgz -C ~ \
    .cache/huggingface/hub/models--depth-anything--Depth-Anything-V2-Large-hf \
    .cache/huggingface/hub/models--facebook--mask2former-swin-large-cityscapes-semantic \
    .cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth
cp /tmp/inception-2015-12-05.pt .   # 若存在

# 在新服务器上解压
tar xzf eval_weights.tgz -C ~
cp inception-2015-12-05.pt /tmp/
export HF_HUB_OFFLINE=1
python -m evalkit check --device cuda:0
```

想让两台服务器的结果完全可比，Python 包版本也要一致：在旧服务器 `pip freeze > env_freeze.txt`，新服务器 `pip install -r env_freeze.txt`。

## 完整流程（多方法、路侧→车端数据）

```bash
# 1. 测试集清单（以 Ours 测试集为准：每个测试 clip 29 帧 × 7 相机）
#    同时导出评测分辨率 1280x720 下、GT 去畸变后的相机内参，交给各方法按此渲染
python -m evalkit render-list --out ./results/render_list.json

# 2. 各方法按清单输出后，生成 manifest
#    GS 类方法目录约定：{root}/{clip}/{timestamp}/{camera}.png
python -m evalkit manifest --render-list ./results/render_list.json \
    --gs DDGS=/path/ddgs --gs SparseGS=/path/sparsegs \
    --ours Ours=/mnt/zihanw/tf2.5_verion2_test_evaluation \
    --out ./results/manifest.csv

# 3. 评测
python -m evalkit run --manifest ./results/manifest.csv --out ./results/evalkit \
    --gt-undistort tianjin --device cuda:0
```

输出：`per_sample.csv`（逐样本）、`summary.json`（均值、95% CI、分组）、`table.md` / `table.tex`。

## manifest 格式（其他任务自己生成即可）

| 列 | 说明 |
|---|---|
| sample_id | 样本标识，跨方法相同；同一 sample_id 必须对应同一张 GT（不一致会报错） |
| method | 方法名 |
| clip | 场景名，置信区间按 clip 重采样 |
| camera | 相机名（`--gt-undistort tianjin` 时用于去畸变，否则可空） |
| timestamp / group | 可空；group 非空时按组额外汇总（如 near/middle/far） |
| gen | 图像路径，或视频帧 `video.mp4#帧序号` |
| gt | 真值图像路径 |

默认只评测**所有方法共有的样本**，保证可比；`--allow-partial` 可取消（结果不可直接比较）。

## 预处理

- gen：resize 到评测分辨率（默认 1280x720，bicubic）
- GT：`--gt-undistort tianjin` 时按标定去畸变（alpha=0，FN 不处理），再 resize（bilinear）；`none` 时只 resize

## 指标

| 指标 | 定义 | 汇总 |
|---|---|---|
| PSNR↑ | uint8 RGB 全图 MSE，`10·log10(255²/MSE)` | 逐样本均值 |
| SSIM↑ | skimage，高斯窗 11、σ=1.5、RGB、data_range=255 | 逐样本均值 |
| LPIPS↓ | AlexNet v0.1，输入 [-1,1] | 逐样本均值 |
| KID↓ / FID↓ | clean-fid（clean 模式），每个方法与其样本对应的 GT 集合 | 集合级，无 CI |
| δ1↑ / AbsRel↓ | Depth-Anything-V2-Large 视差；GT 有效区域：视差 > P99/100（深度 ≤ 最近处×100）；视差空间最小二乘 scale+shift 对齐后取倒数为深度；δ1 = `max(d/d*, d*/d) < 1.25` 占比，AbsRel = `mean(|d−d*|/d*)`；不截尾 | 逐样本均值 |
| Seg-Cons↑ | Mask2Former Swin-L (Cityscapes) 分别分割，类别相同的像素占比（含地面类） | 逐样本均值 |
| Coarse-mIoU↑ | 19 类合并为 Cityscapes 7 个 category，全部样本混淆矩阵累加后算 IoU 再平均 | 数据集级 |

置信区间：以 clip 为单位 bootstrap 1000 次，取 2.5% / 97.5% 分位数（同一 clip 内帧高度相关，按帧重采样会低估不确定性）。表格中 `±` 为 CI 宽度的一半，精确上下界见 `summary.json`。

## 回归验证

```bash
python -m evalkit legacy-common --out ./results/manifest_common.csv
python -m evalkit run --manifest ./results/manifest_common.csv --out ./results/evalkit_common \
    --gt-undistort tianjin --metrics pixel depth seg
```

结果应与 `gs_eval.py --common` / `tf_eval.py --common` 一致（KID/FID 需要足够样本，28 对时不计算）。
