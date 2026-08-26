# ROI160数据集按来源过滤 - 使用指南

## 🎯 过滤策略

根据数据来源分析的结果，采用差异化处理：

| 数据来源 | 策略 | 理由 |
|---------|------|------|
| **ARD100原始** | ✅ **保留全部** | 已经很干净，<0.1%有大目标 |
| **Diverse多样性** | ⚠️ **过滤>80px** | 大目标主要在这里（~10-15%） |
| **其他文件** | ✅ **保留全部** | 安全起见 |

---

## 🚀 快速开始

### 步骤1：Dry-run查看统计

```bash
python tools/filter_by_source.py \
    --input data/ard100_roi160_merged \
    --output data/ard100_roi160_filtered \
    --dry-run
```

**预期输出示例**：
```
==================================================================
数据集过滤统计 - 按来源分组
==================================================================

📊 ARD100 数据:
  总图片数:    4000
  保留:        4000 (100.0%)
  过滤:           0 (0.0%)
  尺寸分布:
      0-  9px:  400 (10.0%) ████████
     10- 19px: 1600 (40.0%) ████████████████████████████████
     20- 29px: 1400 (35.0%) ████████████████████████
     30- 39px:  500 (12.5%) ██████████
     40- 49px:   80 ( 2.0%) █
     50- 59px:   18 ( 0.5%) 
     60- 69px:    2 ( 0.05%)

📊 DIVERSE 数据:
  总图片数:    1000
  保留:         900 (90.0%)
  过滤:         100 (10.0%)
  尺寸分布:
     10- 19px:  150 (15.0%) ████████
     20- 29px:  200 (20.0%) ███████████
     30- 39px:  200 (20.0%) ███████████
     40- 49px:  150 (15.0%) ████████
     50- 59px:  100 (10.0%) █████
     60- 69px:   80 ( 8.0%) ████
     70- 79px:   70 ( 7.0%) ███
     80- 89px:   30 ( 3.0%) █  ← 过滤线
     90- 99px:   15 ( 1.5%)
    100-109px:    5 ( 0.5%)

==================================================================
📊 总体统计:
  总图片数:    5000
  保留:        4900 (98.0%)
  过滤:         100 (2.0%)
==================================================================
```

### 步骤2：确认无误后执行过滤

```bash
python tools/filter_by_source.py \
    --input data/ard100_roi160_merged \
    --output data/ard100_roi160_filtered
```

### 步骤3：更新数据集配置

```bash
# 复制并修改配置文件
cp data/ard100_roi160_merged.yaml data/ard100_roi160_filtered.yaml

# 修改路径
vim data/ard100_roi160_filtered.yaml
```

修改为：
```yaml
train: data/ard100_roi160_filtered/train/images
val: data/ard100_roi160_filtered/val/images

nc: 1
names: ['uav']
```

---

## 📋 参数说明

### 必需参数

```bash
--input         # 输入数据集目录
--output        # 输出目录
```

### 可选参数

```bash
--roi-size      # ROI尺寸，默认160
--max-size      # Diverse数据的最大目标尺寸阈值，默认80
--ard100-pattern # ARD100文件名匹配模式，默认"ard100"
--diverse-pattern # Diverse文件名匹配模式，默认"diverse"
--dry-run       # 只统计不实际复制文件
```

---

## 🔍 文件名匹配规则

### 默认匹配模式

工具通过文件名识别数据来源：

```python
# ARD100文件：文件名包含"ard100"（不区分大小写）
ard100_0001.jpg  ✅
ARD100_test.jpg  ✅
video_ard100.jpg ✅

# Diverse文件：文件名包含"diverse"（不区分大小写）
diverse_001.jpg  ✅
DIVERSE_uav.jpg  ✅
drone_diverse.jpg ✅

# 其他文件：都不包含
other_image.jpg  → 保留全部
```

### 自定义匹配模式

如果您的文件名不符合默认规则，可以自定义：

```bash
python tools/filter_by_source.py \
    --input data/ard100_roi160_merged \
    --output data/ard100_roi160_filtered \
    --ard100-pattern "ard" \
    --diverse-pattern "div"
```

---

## 📊 预期结果

### 数据量变化

假设原始数据：
- ARD100：4000张
- Diverse：1000张
- 总计：5000张

过滤后：
- ARD100：4000张（100%保留）
- Diverse：~900张（90%保留，过滤~10%）
- **总计：~4900张（98%保留）**

### 目标尺寸分布变化

**过滤前**：
```
8-32px:   60%
32-48px:  20%
48-64px:  12%
64-80px:   6%
>80px:     2%  ← 需要过滤
```

**过滤后**：
```
8-32px:   62%
32-48px:  21%
48-64px:  13%
64-80px:   4%
>80px:     0%  ← 已过滤
```

---

## ✅ 验证过滤结果

### 检查文件数量

```bash
# 原始数据
echo "Original:"
find data/ard100_roi160_merged/train/images -name "*.jpg" | wc -l
find data/ard100_roi160_merged/val/images -name "*.jpg" | wc -l

# 过滤后数据
echo "Filtered:"
find data/ard100_roi160_filtered/train/images -name "*.jpg" | wc -l
find data/ard100_roi160_filtered/val/images -name "*.jpg" | wc -l
```

### 检查标签文件

```bash
# 确保images和labels数量一致
echo "Train:"
ls data/ard100_roi160_filtered/train/images/*.jpg | wc -l
ls data/ard100_roi160_filtered/train/labels/*.txt | wc -l

echo "Val:"
ls data/ard100_roi160_filtered/val/images/*.jpg | wc -l
ls data/ard100_roi160_filtered/val/labels/*.txt | wc -l
```

### 抽查被过滤的文件

```bash
# 查看被过滤的文件（从dry-run输出）
python tools/filter_by_source.py \
    --input data/ard100_roi160_merged \
    --output /tmp/test \
    --dry-run | grep "❌"
```

---

## 🎯 与训练集成

### 使用过滤后的数据集训练

```bash
source /home/adlink/chenx/rknn-env/bin/activate

torchrun --nproc_per_node=2 --master_port=29500 \
    tools/train.py \
    --conf configs/yolov6n_ard100_roi160_p2p3_nwd.py \
    --data data/ard100_roi160_filtered.yaml \
    --img-size 160 \
    --batch-size 128 \
    --epochs 400 \
    --device 0,1 \
    --workers 4 \
    --output-dir runs/train \
    --name yolov6n_roi160_p2p3_filtered
```

---

## 🆚 对比实验建议

为了验证过滤效果，建议进行对比实验：

### 实验A：未过滤数据集

```bash
torchrun --nproc_per_node=2 tools/train.py \
    --conf configs/yolov6n_ard100_roi160_p2p3_nwd.py \
    --data data/ard100_roi160_merged.yaml \
    --name yolov6n_roi160_unfiltered
```

### 实验B：过滤后数据集

```bash
torchrun --nproc_per_node=2 tools/train.py \
    --conf configs/yolov6n_ard100_roi160_p2p3_nwd.py \
    --data data/ard100_roi160_filtered.yaml \
    --name yolov6n_roi160_filtered
```

### 对比指标

- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall
- 推理速度
- 虚警率（实际测试视频）

---

## ⚠️ 注意事项

### 1. 备份原始数据

```bash
# 强烈建议先备份
cp -r data/ard100_roi160_merged data/ard100_roi160_merged.backup
```

### 2. 先Dry-run

**永远先运行dry-run**，确认统计信息合理后再执行实际过滤。

### 3. 检查文件名模式

确保文件名符合默认匹配规则：
- ARD100文件包含"ard100"
- Diverse文件包含"diverse"

如果不符合，使用`--ard100-pattern`和`--diverse-pattern`自定义。

### 4. 验证标签完整性

过滤后检查images和labels数量是否一致。

---

## 💡 常见问题

### Q1: 如果文件名不规范怎么办？

**A**: 有两种方法：

方法1：重命名文件
```bash
# 为ARD100文件添加前缀
cd data/ard100_roi160_merged/train/images
for f in original_*.jpg; do
    mv "$f" "ard100_$f"
done
```

方法2：使用自定义匹配模式
```bash
python tools/filter_by_source.py \
    --ard100-pattern "original" \
    --diverse-pattern "div"
```

### Q2: 如果过滤率异常怎么办？

**A**: 检查三个方面：
1. 文件名匹配是否正确（dry-run输出会显示分组）
2. 数据来源是否混淆
3. ROI切片策略是否有问题

### Q3: 能否调整Diverse的过滤阈值？

**A**: 可以，使用`--max-size`参数：

```bash
# 更宽松：100px
python tools/filter_by_source.py \
    --input data/ard100_roi160_merged \
    --output data/ard100_roi160_filtered_loose \
    --max-size 100

# 更严格：64px
python tools/filter_by_source.py \
    --input data/ard100_roi160_merged \
    --output data/ard100_roi160_filtered_strict \
    --max-size 64
```

---

## 📚 相关文档

- **过滤工具**: `tools/filter_by_source.py`
- **数据来源分析**: `docs/roi160_target_source_analysis.md`
- **过滤策略**: `docs/roi160_filtering_strategy_revised.md`
- **影响评估**: `docs/roi160_filtering_impact_estimate.md`

---

## ✨ 总结

这个工具实现了您要求的策略：

✅ **ARD100原始数据**：全部保留（已经很干净）
✅ **Diverse多样性数据**：过滤>80px（大目标主要在这）
✅ **Label文件**：自动同步处理
✅ **统计分析**：按来源分组显示
✅ **Dry-run模式**：安全预览

**预期结果**：保留98%数据，只过滤Diverse中的大目标（~2%）

---

**创建日期**: 2026-08-26  
**工具**: `tools/filter_by_source.py`
