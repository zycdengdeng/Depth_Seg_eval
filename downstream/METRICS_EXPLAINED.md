# CSE 下游感知指标说明（写论文用）

> 用途：把"CSE 渲染 → 下游感知"的指标放进论文，论证 **data reuse / GS-Net 重建对下游任务有效**。
> 本文逐个解释指标含义、公式、取值范围与方向、**标准出处**，并给出论文里的措辞建议。

---

## 0. 先讲清楚：我们到底在量什么（最重要，审稿人必问）

我们**没有人工标注的深度图/分割图**作真值。做法是：

1. 取一个**冻结的、现成的预训练感知模型**（深度用 Depth-Anything-V2，分割用 Mask2Former/Cityscapes）。
2. 把它分别作用在 **真实采集图(GT)** 和 **我们的渲染图(render)** 上。
3. 比较两者输出：`metric( perception(render), perception(GT) )`。

所以——

- **"参照(reference)" = 同一个感知模型在真实 GT 图上的输出**，不是人工标注。
- 指标度量的是 **"在渲染图上做感知，和在真实图上做感知，结果有多一致"**。
- **逻辑链**：如果渲染图保留了真实的几何与语义，那么下游感知模型在渲染图上的行为就应当和在真实图上一致 → 渲染图对下游任务"可用"。**gsnet 比 baseline 更一致 = 更好的重建确实翻译成更好的下游感知。**
- 两次 run（baseline / gsnet）的 **GT 完全相同**，只有 render 变。所以表里是 **两列数字**，比谁离"真实图上的感知结果"更近。

> 一句话定性：这是一种 **感知一致性 / 域差距代理 (perceptual-consistency / domain-gap proxy)** 评测，而非对人工标签的绝对精度评测。诚实地这么写，反而堵住审稿人的质疑。

---

## 1. 这些指标有标准出处吗？——有，全是社区通用协议

| 指标族 | 标准出处（可引用） |
|---|---|
| 深度 abs_rel / sq_rel / rmse / rmse_log / δ<1.25ⁿ | **Eigen et al., NeurIPS 2014**（单目深度评测协议），后被 KITTI/NYU、MonoDepth 等沿用为事实标准 |
| Pearson / Spearman 相关 | 标准统计量（线性相关 / 秩相关） |
| 分割 mIoU / pixel acc / fwIoU | **Long et al. (FCN), CVPR 2015**；mIoU 亦源自 **PASCAL VOC (Everingham et al.)** |
| Cityscapes 19 类 / 7 超类体系 | **Cordts et al. (Cityscapes), CVPR 2016** |
| 感知模型 | 深度 **Depth-Anything-V2 (Yang et al., 2024)**；分割 **Mask2Former (Cheng et al., CVPR 2022)** |

即：指标本身是教科书/顶会标准协议，**新意在于把它们用作 3DGS 渲染的下游感知评测**，而不是发明新指标。这点对论文有利——可比、可信、好复现。

---

## 2. 深度指标（Depth-Anything-V2，render vs GT 图）

记渲染图深度 `d`、真实图深度 `d*`（单目相对深度，已按中位数对齐尺度）。

| 指标 | 全称 | 方向 | 取值 |
|---|---|---|---|
| abs_rel | Absolute Relative Error | ↓ 越低越好 | [0, ∞) |
| rmse | Root Mean Squared Error | ↓ | [0, ∞)（任意单位）|
| delta_1 (δ<1.25) | Threshold Accuracy | ↑ 越高越好 | [0, 100]% |
| pearson | Pearson 相关系数 | ↑ | [-1, 1] |

**逐个说明：**

- **abs_rel（绝对相对误差）** = 均值 `|d − d*| / d*`。
  每个像素的相对误差再平均。对尺度不敏感、最常用、最该重点报。
  *0.0508 表示渲染深度与真实图深度平均相差约 5%。*

- **rmse（均方根误差）** = `sqrt( mean( (d − d*)² ) )`。
  逐像素绝对误差的二范数，对大误差更敏感。
  ⚠️ **注意单位**：这里深度是单目**相对深度（任意单位）**，所以 rmse≈13 是模型尺度下的数值，**不是米**。它只能在**同一套设置内做相对比较**（baseline vs gsnet），论文里**不要标成米/绝对深度**，否则会被挑错。abs_rel / δ / pearson 这类尺度不敏感的更适合放主表。

- **delta_1（阈值准确率 δ<1.25）** = 满足 `max(d/d*, d*/d) < 1.25` 的像素占比(%)。
  即"渲染深度与参照深度的比值落在 0.8~1.25 之间"的像素比例。越高说明大部分像素都对得很准。
  （另有 δ<1.25²=delta_2、δ<1.25³=delta_3，阈值放宽，值更高。）

- **pearson（皮尔逊相关）** = `d` 与 `d*` 跨像素的线性相关。
  衡量整幅深度图的结构/排布是否一致。这里两边都 >0.99，**已饱和**，区分度低——可作"两者高度相关"的佐证，但**不适合当主打差异指标**。
  （另有 spearman=秩相关，对单调非线性更鲁棒。）

> 主表建议：**abs_rel(↓) + δ<1.25(↑)** 两个就够代表深度，必要时加 rmse（但注明任意单位/相对）。pearson 当"已高度一致"的背景说明。

---

## 3. 分割指标（Mask2Former / Cityscapes，render vs GT 图）

记 `P = seg(render)`、`R = seg(GT图)`，把 `R` 当作参照。

| 指标 | 全称 | 方向 | 取值 |
|---|---|---|---|
| consistency | Pixel Agreement（像素一致率）| ↑ | [0, 100]% |
| mIoU (19类) | mean Intersection-over-Union | ↑ | [0, 100]% |
| mIoU-7 (7超类) | coarse mean IoU | ↑ | [0, 100]% |
| fwIoU | frequency-weighted IoU | ↑ | [0, 100]% |

**逐个说明：**

- **consistency（一致率）** = `P` 与 `R` 类别相同的像素占比(%)。
  最直观：渲染图和真实图被分到同一类的像素有多少。等价于把 `R` 当 GT 的 pixel accuracy。

- **mIoU（19 类平均交并比）** = 对每个类算 `IoU = TP / (TP+FP+FN)`（以 `R` 为 GT 定义 TP/FP/FN），再对**出现过的类**求平均。
  语义分割的**头号标准指标**。
  ⚠️ 为什么数值偏低(~46%)：mIoU 是**按类平均**，细小/稀有类（杆子、交通标志、行人）一致性差，会把均值拉低；而且这是"模型 vs 模型"的一致性、不是对人工标签。**这是正常现象，不代表分割很差**——可在文中点一句，免得读者误读。

- **mIoU-7（7 超类）** = 把 19 类按 Cityscapes 官方层级并成 7 个超类（flat/construction/object/nature/sky/human/vehicle）后再算 mIoU。
  粒度更粗 → 数值更高(~60%)、更稳，受细类噪声影响小。适合作为"语义结构层面"的稳健补充。

- **fwIoU（频率加权 IoU）** = `Σ_c (freq_c × IoU_c)`，按各类像素占比加权。
  被**大面积常见类**（路面、建筑、天空）主导 → 数值高(~83%)。反映"画面主体语义"的一致性。

> 主表建议：分割放 **mIoU(↑)** 一个即可（最标准）；正文里可补 consistency 或 fwIoU 说明"主体语义高度一致"。19 类 mIoU 偏低务必加一句解释。

---

## 4. （可选）SAM 结构一致性

| 指标 | 含义 | 方向 |
|---|---|---|
| edge_f1 | SAM 特征边缘图的边界 F1（带容差）| ↑ |
| edge_correlation | 边缘图逐像素相关 | ↑ |

度量物体边界/结构一致性（不涉及类别）。**但实测两边都 ~96–98%、已饱和，gsnet 甚至略低 0.8%**——差异在噪声级。
**建议：不要把 SAM 当下游有效性的主打证据**；要放就如实写"边界结构两者均高度一致(~97%)，无显著差异"。

---

## 5. 论文里怎么组织语言（可直接改用）

### 5.1 方法/评测设置段（英文模板）
> **Downstream perception consistency.** To assess whether our reconstruction yields renderings that are *useful for downstream perception*, we run frozen off-the-shelf perception models on both the real held-out images (reference) and our renderings, and measure their agreement. We use Depth-Anything-V2 for monocular depth and Mask2Former (Cityscapes) for semantic segmentation. For depth we report the standard error metrics of Eigen et al. (AbsRel, RMSE, δ<1.25) after median scale alignment; for segmentation we report mean IoU, frequency-weighted IoU and pixel agreement under the Cityscapes taxonomy. Higher agreement indicates that the rendering preserves the geometry and semantics a perception model relies on. We compare two initializations under identical references: SfM (baseline) and GS-Net (ours).

### 5.2 结果段（英文模板，数字替换成你筛的）
> As shown in Table X, GS-Net initialization consistently improves downstream perception consistency over the SfM baseline on CARLA CSE: depth AbsRel drops by 12.5% (0.058→0.051) and δ<1.25 rises to 94.7%, while segmentation mIoU improves by +1.0 (46.1→47.2). The gains are largest in geometry-sensitive depth metrics, consistent with GS-Net improving the underlying geometric reconstruction.

### 5.3 每个指标一句话（塞进表注或正文）
- AbsRel↓: mean relative depth error w.r.t. perception on the real image (Eigen et al.).
- RMSE↓: root-mean-square depth error (relative units; for within-setup comparison only).
- δ<1.25↑: fraction of pixels within a 1.25 ratio of the reference depth.
- mIoU↑: mean per-class IoU between segmentation of render vs real image (Cityscapes 19 classes).
- fwIoU↑: frequency-weighted IoU (dominated by common classes).
- pixel agreement↑: fraction of pixels assigned the same class.

### 5.4 表注模板
> Table X. Downstream perception consistency on CARLA CSE. We report agreement between a frozen perception model's predictions on real images vs. renderings; ↑/↓ indicate whether higher/lower is better. **Bold** = better. GT is the reference (trivially perfect) and thus omitted.

---

## 6. 写作注意（避免被审稿人挑）

1. **别声称绝对精度**：reference 是"模型在真实图上的预测"，不是人工标注。用 "consistency / agreement" 而非 "accuracy"。
2. **RMSE 不要写成米**：单目相对深度、任意单位，仅同设置内可比。
3. **19 类 mIoU 偏低要解释**：类平均 + 含稀有细类 + 模型间一致性，不是分割崩了。
4. **SAM 别硬吹**：已饱和、无显著差异，如实写或不放。
5. **深度是最强证据**：GS-Net 改的是几何，深度因果最直接、提升最大（per-scene 110 上 AbsRel 提升 ~30%），主叙事围绕深度展开最稳。
6. **per-scene 差异要诚实**：若只展示某场景，注明是代表性场景；不要从 gsnet 反而更差的场景（如 510）里挑单帧炫耀。

---

## 附：库里还能输出但本表没列的指标
深度还有 `sq_rel / rmse_log / delta_2 / delta_3 / spearman / ssim`；分割还有 `pixel_acc / 每类 class_iou`。
需要更全的对照或某个细类 IoU，可在结果 JSON 里取，或让我加进 `compare_runs.py` 的指标清单。
