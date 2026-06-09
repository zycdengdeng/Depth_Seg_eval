# 下游感知评测 (downstream perception eval)

回应审稿"只在 3DGS 内评估、没证 data reuse 价值"：用本库对**我们的渲染(gen)**和**真值(gt)**
各跑预训练感知模型（深度/分割/SAM），比 `metric(gen, gt)`。GS-Net init 的一致性更高
= 更好的重建翻译成更好的下游感知。

- **gen** = 我们的 3DGS 渲染（baseline=SfM 点 / gsnet=GS-Net init）
- **gt**  = 真值图（两次 run 完全相同，只变 gen）
- **task** = depth / seg / sam（深度最该重点看；CARLA 合成域分割/SAM 作补充，nuScenes 三个都可信）

数据由 gsnet 侧的 `gsnet.prep_downstream` 整理成本库的图像帧格式
`_eval_frames/<group>/{gen,gt}/*.png`（`<group>` = 序列 id / clip 名，仅影响 per-group 报告）。

## 1. 4 个 config（已对应 4 次 run）

| config | 数据集 | init | `_eval_frames` root |
|---|---|---|---|
| `config_cse_baseline.yaml`  | CARLA CSE | baseline | `/mnt/zihanw/downstream_cse/baseline/_eval_frames` |
| `config_cse_gsnet.yaml`     | CARLA CSE | gsnet    | `/mnt/zihanw/downstream_cse/gsnet/_eval_frames` |
| `config_nusc_baseline.yaml` | nuScenes SSE | baseline(sfm) | `/mnt/zihanw/downstream_nusc/sfm/_eval_frames` |
| `config_nusc_gsnet.yaml`    | nuScenes SSE | gsnet    | `/mnt/zihanw/downstream_nusc/gsnet/_eval_frames` |

每个 run 的结果落在该 init 目录下的 `results/metrics/`，互不覆盖。
> 路径/分组/跳过 310 等如需调整，直接改对应 config 即可。

## 2. 跑评测（每个 config 跑 depth/seg/sam 三个 task）

`evaluate.py` 的 `--task` 一次只接一个；`all` 会多跑 image_metrics/fvd（这里不需要），所以分开跑三个：

```bash
cd Depth_Seg_eval

# 以 CARLA baseline 为例（其余 3 个 config 同理换 --config）
for T in depth seg sam; do
  python evaluate.py --config downstream/config_cse_baseline.yaml --task $T --no-vis
done
```

多 GPU 加速（按 group 分到不同卡，depth/seg/sam 都支持；FVD 不在我们清单里）：

```bash
# --parallel 但不带 --gpus：开跑前自动查 nvidia-smi，只用空闲的卡，避开合作者占用的
python evaluate.py --config downstream/config_cse_baseline.yaml --task depth --no-vis --parallel
```

**共用服务器自动选卡**：不指定 `--gpus` 时，串行会挑最空的一张卡、并行会用所有空闲卡
（默认要求单卡 ≥20GB 空闲且已用 ≤50%，占满的自动跳过）。可调：
- `--min-free-mem 30000` 提高空闲显存门槛
- `--max-gpus 4` 最多用 4 张
- `--gpus 0,1,2,3` 仍可手动指定（优先级最高）
- 先单独看一眼选卡情况：`python gpu_utils.py`

对 4 个 config 各跑一遍，得到 4 套 `results/metrics/{depth,seg,sam}_results.json`。

## 3. 出对比表（baseline vs gsnet）

```bash
python downstream/compare_runs.py \
  --pair "CARLA CSE" \
      /mnt/zihanw/downstream_cse/baseline/results/metrics \
      /mnt/zihanw/downstream_cse/gsnet/results/metrics \
  --pair "nuScenes SSE" \
      /mnt/zihanw/downstream_nusc/sfm/results/metrics \
      /mnt/zihanw/downstream_nusc/gsnet/results/metrics \
  --out downstream/comparison.md
```

- 并排打印 baseline / gsnet / Δ，✅/❌ 标 gsnet 是否更优，末尾给"论文主表"（abs_rel↓ / mIoU↑ / edge_f1↑）。
- 传 run root（`.../results`）也行，会自动找其下 `metrics/`。
- 加 `--per-group` 看每个序列/clip 的逐项对比。

## 4. 定性可视化（gt | baseline | gsnet 三联图）

`compare_runs.py` 出的是**数字**；要肉眼看"三种效果"，用 `visualize_compare.py`，
对采样帧每个模态各出一张 1x3 拼接图（列 = GT / Baseline / GS-Net）：

```bash
python downstream/visualize_compare.py \
  --baseline-root /mnt/zihanw/downstream_cse/baseline/_eval_frames \
  --gsnet-root    /mnt/zihanw/downstream_cse/gsnet/_eval_frames \
  --out           /mnt/zihanw/downstream_cse/compare_vis \
  --tasks rgb depth seg sam \
  --every 10
```

输出结构（每种模态独立成图）：
```
compare_vis/<group>/
├── rgb/<frame>.png     # gt | baseline | gsnet 原图
├── depth/<frame>.png   # 深度图（baseline/gsnet 已对齐 gt 尺度，三者同色阶）
├── seg/<frame>.png     # 语义分割彩色图（Cityscapes 调色板）
└── sam/<frame>.png     # SAM 边缘图
```
- gt 取自 baseline 那份 `_eval_frames`（两次 run 的 gt 相同）。
- `--every N` 控制采样密度（默认 10）；`--groups 110 210` 只看部分序列；`--gpu N` 指定卡（默认自动选空闲卡）。
- 这是给论文用的定性对比图：gsnet 的深度/分割若更贴近 gt 列，就直观印证了指标。

## 指标怎么理解（重要）
- **gt 是"参照标准/满分锚点"，不是被打分的对象。** 表里是两列数字：baseline-vs-gt、gsnet-vs-gt，
  比谁离 gt 更近。gt 自己跟自己比恒为满分（abs_rel=0 / mIoU=100%），只是上界。
- 这里 gen/gt 是**同视角像素对齐**（渲染 vs 真值），所以 depth 主指标 `abs_rel/rmse/delta_1`
  本就是严格逐像素算的（库里的 `align_spatial`/`tolerant_*` 只是额外参考列，不影响主指标）。
- 期望结果：gsnet 的 depth abs_rel/rmse 更低、seg mIoU/consistency 更高、SAM edge_f1 更高。
