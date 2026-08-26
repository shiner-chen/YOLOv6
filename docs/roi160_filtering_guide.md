# ROI160 数据集过滤指南

## ❓ 为什么需要过滤大目标？

### 问题分析

在ROI160数据集中，大目标图片会带来以下问题：

| 目标尺寸 | 面积占比 | 边长占比 | 问题 |
|---------|---------|---------|------|
| 80px | 25% | 50% | 背景上下文不足，易边缘截断 |
| 100px | 39% | 62% | 严重缺乏场景信息，目标几乎占满 |
| 120px+ | 56%+ | 75%+ | 完全失去检测意义 |

### 核心理由

1. **P2/P3检测头设计范围**
   - P2 (stride=4): 设计用于 4-20px
   - P3 (stride=8): 设计用于 16-40px
   - **80px+目标超出设计范围**

2. **背景上下文不足**
   - 80px目标占ROI的25%面积
   - 网络无法学习"目标在场景中的关系"
   - 分类能力下降

3. **边缘截断风险**
   - Motion检测的blob中心可能不精确
   - 80px目标容易被截断在边缘
   - 检测头感受野不足

4. **与motion-guided场景不符**
   - Motion检测提取的是**远距离微小目标**
   - 如果目标已经80px+，不需要ROI切片

---

## ✅ 推荐过滤策略

### 方案对比

| 方案 | 阈值 | 过滤比例 | 适用场景 |
|-----|------|---------|---------|
| **保守**(推荐) | >80px | 中等 | 追求高精度，边缘部署 |
| 宽松 | >100px | 较少 | 数据不足，追求召回 |
| 严格 | >64px | 较多 | 极致性能，对虚警零容忍 |

### 推荐阈值：80px

**理由**：
- ✅ 与P2/P3设计目标一致（4-40px）
- ✅ 面积占比<25%，保留充足上下文
- ✅ 边长占比=50%，边界安全阈值
- ✅ 符合motion-guided微小目标检测场景

---

## 🚀 使用方法

### 1. 先dry-run查看统计
```bash
python tools/filter_roi160_oversized.py \
    --data-dir data/ard100_roi160_merged \
    --output-dir data/ard100_roi160_filtered \
    --max-size 80 \
    --dry-run
```

输出示例：
```
总图片数:     5000
保留:         4500 (90.0%)
过滤:         500 (10.0%)

目标尺寸分布:
  0-  9px:  800 (16.0%) ████████
  10- 19px: 1500 (30.0%) ███████████████
  20- 29px: 1200 (24.0%) ████████████
  30- 39px:  600 (12.0%) ██████
  40- 49px:  400 (8.0%)  ████
  50- 59px:  200 (4.0%)  ██
  60- 69px:  150 (3.0%)  █
  70- 79px:  100 (2.0%)  █
  80- 89px:   30 (0.6%)  
  90- 99px:   15 (0.3%)  
  100+px:      5 (0.1%)  
```

### 2. 执行实际过滤
```bash
python tools/filter_roi160_oversized.py \
    --data-dir data/ard100_roi160_merged \
    --output-dir data/ard100_roi160_filtered \
    --max-size 80
```

### 3. 更新数据集配置
```yaml
# data/ard100_roi160_filtered.yaml
train: data/ard100_roi160_filtered/train/images
val: data/ard100_roi160_filtered/val/images

nc: 1
names: ['uav']
```

### 4. 使用过滤后的数据集训练
```bash
torchrun --nproc_per_node=2 --master_port=29500 \
    tools/train.py \
    --conf configs/yolov6n_ard100_roi160_p2p3_nwd.py \
    --data data/ard100_roi160_filtered.yaml \
    --img-size 160 --batch-size 128 --epochs 400
```

---

## 🔧 参数说明

### 基本参数
```bash
--data-dir       # 输入数据集目录（包含train/val文件夹）
--output-dir     # 输出目录
--roi-size       # ROI尺寸，默认160
--max-size       # 最大目标尺寸(像素)，默认80
--dry-run        # 只统计不实际复制文件
```

### 不同阈值的效果

**--max-size 64 (严格)**
```bash
python tools/filter_roi160_oversized.py \
    --data-dir data/ard100_roi160_merged \
    --output-dir data/ard100_roi160_filtered_strict \
    --max-size 64
```
- 过滤更多图片
- 数据更纯净，但样本量减少
- 适合数据充足且追求极致性能

**--max-size 80 (推荐)**
```bash
python tools/filter_roi160_oversized.py \
    --data-dir data/ard100_roi160_merged \
    --output-dir data/ard100_roi160_filtered \
    --max-size 80
```
- 平衡性能和数据量
- **最推荐的方案**

**--max-size 100 (宽松)**
```bash
python tools/filter_roi160_oversized.py \
    --data-dir data/ard100_roi160_merged \
    --output-dir data/ard100_roi160_filtered_loose \
    --max-size 100
```
- 保留更多样本
- 适合数据不足的情况
- 可能影响精度

---

## 📊 预期效果

### 过滤前后对比

| 指标 | 过滤前 | 过滤后(80px) | 改善 |
|-----|-------|------------|------|
| 平均目标尺寸 | 25px | 20px | 更聚焦微小目标 |
| 大目标比例(>80px) | 5-10% | 0% | 完全剔除 |
| 背景上下文 | 不足 | 充足 | 提升分类能力 |
| 边缘截断风险 | 高 | 低 | 提升召回率 |

### 训练效果预期

过滤后训练的模型：
- ✅ **虚警率降低**：大目标干扰减少
- ✅ **精度提升**：P2/P3专注设计范围
- ✅ **泛化能力强**：场景上下文充足
- ✅ **边缘case减少**：无截断目标

---

## 🆚 与ROI320对比

| 项目 | ROI320 | ROI160 |
|-----|--------|--------|
| 推荐阈值 | >80px | >80px |
| 面积占比 | 6.25% | 25% |
| 边长占比 | 25% | 50% |
| 过滤必要性 | 中等 | **高** |

**结论**：ROI160比ROI320**更需要**过滤大目标！

---

## ⚠️ 注意事项

### 1. 备份原始数据
```bash
# 建议先备份
cp -r data/ard100_roi160_merged data/ard100_roi160_merged.backup
```

### 2. 检查过滤结果
```bash
# 查看过滤的图片
ls -lh data/ard100_roi160_filtered/train/images | wc -l
ls -lh data/ard100_roi160_merged/train/images | wc -l
```

### 3. 对比训练效果
建议同时训练过滤前后的模型进行对比：
```bash
# 过滤前
--name yolov6n_roi160_unfiltered

# 过滤后
--name yolov6n_roi160_filtered_80px
```

### 4. 可视化检查
随机抽查一些被过滤的图片，确认确实是大目标：
```bash
# 查看被过滤的图片列表（dry-run输出）
python tools/filter_roi160_oversized.py \
    --data-dir data/ard100_roi160_merged \
    --output-dir /tmp/test \
    --max-size 80 \
    --dry-run | grep "❌"
```

---

## 📚 相关文档

- **理论分析**: `tools/analyze_roi_target_size.py`
- **实现文档**: `docs/yolov6n_roi160_p2p3_implementation.md`
- **快速开始**: `QUICKSTART.md`

---

## 💡 FAQ

### Q1: 为什么不是64px或100px？
**A**: 80px是综合平衡点：
- 64px：过于严格，可能损失有效样本
- 80px：面积25%，边长50%，符合检测头设计
- 100px：过于宽松，39%面积占比过大

### Q2: 如果数据量不够怎么办？
**A**: 可以放宽到100px，但建议：
1. 先用80px训练一个模型
2. 再用100px训练一个模型
3. 对比性能后决定

### Q3: motion-guided场景下会有很多大目标吗？
**A**: 理论上不应该有：
- Motion检测提取的是远距离微小目标
- 如果有大量>80px目标，说明ROI切片策略有问题
- 建议检查motion检测的阈值设置

### Q4: 过滤会不会影响模型的泛化能力？
**A**: 不会，反而会提升：
- 大目标缺乏上下文，本身就是噪声
- 专注4-40px范围，模型更专业
- P2/P3检测头本来就不是为大目标设计的

---

**创建日期**: 2026-08-26  
**建议**: 使用80px阈值过滤ROI160数据集
