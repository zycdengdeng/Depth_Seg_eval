# 生成图像质量评测工具

评测扩散模型生成图像与真值图像在**深度估计**和**语义分割**上的一致性。

## 评测原理

对于生成式模型（如Cosmos），我们通过以下方式评测生成图像的质量：

1. **深度一致性**：对gen和gt图像分别进行深度估计，比较 `depth(gen)` vs `depth(gt)`
   - 如果一致性高，说明生成图像的**几何结构**保持良好

2. **分割一致性**：对gen和gt图像分别进行语义分割，比较 `seg(gen)` vs `seg(gt)`
   - 如果一致性高，说明生成图像的**语义内容**保持良好

## 使用的模型

### 深度估计
- **Depth Anything V2** (推荐): 最先进的单目深度估计模型
- MiDaS: 经典深度估计模型（备选）

### 语义分割
- **Mask2Former** (推荐): Cityscapes预训练，适合自动驾驶场景
- SegFormer: 轻量级但效果好
- OneFormer: 支持多数据集

## 安装

```bash
# 创建虚拟环境（可选）
conda create -n eval python=3.10
conda activate eval

# 安装依赖
pip install -r requirements.txt

# 安装PyTorch (根据CUDA版本选择)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 使用方法

### 1. 修改配置文件

编辑 `config.yaml`，设置数据路径：

```yaml
data:
  root: "/path/to/your/_eval_frames"
  cameras:
    - "cross_left_120fov"
    - "cross_right_120fov"
    - "front_tele_30fov"
    - "front_wide_120fov"
  gen_folder: "gen"
  gt_folder: "gt"
```

### 2. 运行评测

```bash
# 运行全部评测
python evaluate.py --config config.yaml --task all

# 仅运行深度评测
python evaluate.py --task depth

# 仅运行分割评测
python evaluate.py --task seg

# 不保存可视化（加速）
python evaluate.py --task all --no-vis
```

### 3. 查看结果

结果保存在 `./results/metrics/` 目录下：
- `depth_results.json`: 深度评测详细结果
- `seg_results.json`: 分割评测详细结果
- `evaluation_report.txt`: 综合报告
- `full_results.json`: 完整结果

## 评测指标

### 深度指标

| 指标 | 说明 | 理想值 |
|------|------|--------|
| abs_rel | 绝对相对误差 | ↓ 越低越好 |
| rmse | 均方根误差 | ↓ 越低越好 |
| delta_1 | δ < 1.25 比例 | ↑ 越高越好 |
| delta_2 | δ < 1.25² 比例 | ↑ 越高越好 |
| delta_3 | δ < 1.25³ 比例 | ↑ 越高越好 |
| pearson | Pearson相关系数 | ↑ 越高越好 |
| spearman | Spearman相关系数 | ↑ 越高越好 |

### 分割指标

| 指标 | 说明 | 理想值 |
|------|------|--------|
| consistency | 分割结果一致率 | ↑ 越高越好 |
| miou | 平均交并比 | ↑ 越高越好 |
| pixel_acc | 像素准确率 | ↑ 越高越好 |
| fwiou | 频率加权IoU | ↑ 越高越好 |
| class_iou | 每类IoU | ↑ 越高越好 |

## 数据目录结构

```
_eval_frames/
├── cross_left_120fov/
│   ├── gen/          # 生成的图像
│   │   ├── 001.png
│   │   └── ...
│   └── gt/           # 真值图像
│       ├── 001.png
│       └── ...
├── cross_right_120fov/
│   ├── gen/
│   └── gt/
├── front_tele_30fov/
│   ├── gen/
│   └── gt/
└── front_wide_120fov/
    ├── gen/
    └── gt/
```

**注意**：gen和gt中的文件名必须对应。

## 输出示例

```
======================================================================
生成图像质量评测报告
生成时间: 2024-01-15 14:30:00
======================================================================

## 深度一致性评测
--------------------------------------------------
总体结果:
  abs_rel     : 0.0823 ± 0.0156
  rmse        : 2.4567 ± 0.3421
  delta_1     : 92.34% ± 2.15%
  pearson     : 0.9456 ± 0.0234

## 语义分割一致性评测
--------------------------------------------------
总体结果:
  consistency : 87.65% ± 3.21%
  miou        : 72.34% ± 4.56%
  pixel_acc   : 89.12% ± 2.34%
```

## 自定义评测

### 添加新的深度模型

在 `depth_eval.py` 中继承 `DepthEstimator` 类：

```python
class MyDepthEstimator(DepthEstimator):
    def _load_model(self):
        # 加载你的模型
        pass

    def predict(self, image: np.ndarray) -> np.ndarray:
        # 返回深度图
        pass
```

### 添加新的分割模型

在 `seg_eval.py` 中继承 `SemanticSegmentor` 类：

```python
class MySegmentor(SemanticSegmentor):
    def _load_model(self):
        # 加载你的模型
        pass

    def predict(self, image: np.ndarray) -> np.ndarray:
        # 返回分割图
        pass
```
