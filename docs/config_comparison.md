# 训练配置对比：ET-YOLO vs YOLOv6n P2/P3

## 配置文件对比

### 基础信息
| 配置项 | ET-YOLO ROI160 3-scale | YOLOv6n ROI160 P2/P3 |
|-------|----------------------|---------------------|
| 配置文件 | `et_yolov6n_ard100_roi160_nwd_3scale.py` | `yolov6n_ard100_roi160_p2p3_nwd.py` |
| 输入尺寸 | 160×160 | 160×160 |
| 检测头 | P2/P3/P4 (3头) | P2/P3 (2头) |
| 训练模式 | repvgg | repvgg |

---

## 模型架构对比

### Backbone
| 组件 | ET-YOLO | YOLOv6n P2/P3 |
|-----|---------|--------------|
| 类型 | `EfficientRepStar3Scale` | `EfficientRep` |
| Stage数 | 4-stage | **5-stage** (完整) |
| 输出通道 | [64, 128, 256, 512] | [64, 128, 256, 512, **1024**] |
| fuse_P2 | True | True |
| cspsppf | False (SPPF) | False (SPPF) |
| **特点** | CrossLayer多层融合 | **原始YOLOv6，重参数化优化** |

### Neck
| 组件 | ET-YOLO | YOLOv6n P2/P3 |
|-----|---------|--------------|
| 类型 | `CrossTwoLevelBiFPANNeck3ScaleV2` | `RepBiFPANNeckP2P3` |
| 融合方式 | CrossLayerBifusion | BiFusion |
| 输出数 | 3 (P2/P3/P4) | 2 (P2/P3) |
| **特点** | ET-YOLO创新结构 | **原始YOLOv6，硬件友好** |

### Head
| 组件 | ET-YOLO | YOLOv6n P2/P3 |
|-----|---------|--------------|
| 检测头数 | 3 | **2** |
| 特征图 | P2(40×40), P3(20×20), P4(10×10) | P2(40×40), P3(20×20) |
| Stride | [4, 8, 16] | [4, 8] |
| out_indices | [17, 20, 23] | [17, 20] |

---

## Anchors对比

### ET-YOLO ROI160 3-scale
```python
anchors_init=[
    [8,6,   12,9,   18,12],    # P2: small-medium
    [20,14, 28,20,  38,28],    # P3: medium (primary)
    [42,32, 56,42,  72,56]     # P4: large
]
```

### YOLOv6n ROI160 P2/P3
```python
anchors_init=[
    [6,4,   10,7,   15,10],    # P2: 4-20px (primary)
    [20,14, 28,20,  38,28]     # P3: 16-40px (secondary)
]
```

**差异说明**：
- YOLOv6n的P2 anchor更小（6,4起），更适合极小目标
- 删除了P4的大目标anchor（42-72像素范围）

---

## NWD Loss对比

| 参数 | ET-YOLO | YOLOv6n P2/P3 | 说明 |
|-----|---------|--------------|-----|
| nwd_ratio | 0.5 | **0.6** | YOLOv6增加NWD权重 |
| nwd_constant | 32.0 | **17.0** | 基于实际目标大小 |
| 目标假设 | avg 46×26 px | avg 20×15 px | 不同裁剪策略 |

**差异原因**：
- ET-YOLO：基于"tighter crop"，目标占ROI比例大
- YOLOv6n：基于motion-guided ROI，目标可能更小

---

## 训练参数对比

### Solver
| 参数 | ET-YOLO | YOLOv6n P2/P3 | 说明 |
|-----|---------|--------------|-----|
| lr0 | 0.005 | 0.005 | **相同** |
| lrf | 0.01 | 0.01 | **相同** |
| momentum | 0.937 | 0.937 | **相同** |
| weight_decay | 0.0005 | 0.0005 | **相同** |
| warmup_epochs | 3.0 | 3.0 | **相同** |

### Data Augmentation
| 参数 | ET-YOLO | YOLOv6n P2/P3 | 说明 |
|-----|---------|--------------|-----|
| hsv_h | 0.015 | 0.015 | **相同** |
| hsv_s | 0.7 | 0.7 | **相同** |
| hsv_v | 0.4 | 0.4 | **相同** |
| degrees | 0.0 | 0.0 | **相同** |
| translate | 0.1 | 0.1 | **相同** |
| scale | 0.5 | 0.5 | **相同** |
| mosaic | 1.0 | 1.0 | **相同** |
| mixup | 0.0 | 0.0 | **相同** |

**结论**：训练参数和数据增强**完全一致**，保证对比公平性。

---

## 计算量对比（估算）

| 模型 | Backbone | Neck | Head | 总FLOPs | 相对 |
|-----|---------|------|------|---------|-----|
| ET-YOLO 3-scale | 4-stage | CrossLayer | 3头 | ~0.6 G | 1.0× |
| YOLOv6n P2/P3 | 5-stage | BiFusion | **2头** | ~0.4-0.5 G | **0.75×** |

**分析**：
- YOLOv6n虽然backbone多1个stage，但在160×160下C5只有10×10，增加极少
- 删除P4检测头节省大量计算
- **总体FLOPs反而更低**

---

## 推理性能对比（预期）

| 指标 | ET-YOLO 3-scale | YOLOv6n P2/P3 | YOLOv6优势 |
|-----|----------------|--------------|-----------|
| 训练时间 | 基准 | 略快 | 少1个检测头 |
| 推理延迟 (RK3588) | ~3-4ms | **~2-3ms** | **RepVGG重参数化** |
| 召回率 | 高 | 高 | 相当 |
| 精度 (mAP) | 基准 | 预期相当 | 2头vs3头权衡 |
| 虚警率 | 低 | 低 | 都有深层语义 |
| **部署友好度** | 中等 | **极高** | **NPU优化** |

---

## 核心差异总结

### ET-YOLO优势
1. ✅ CrossLayerBifusion创新结构
2. ✅ 3个检测头覆盖更广
3. ✅ 可能对中等目标(30-60px)更好

### YOLOv6n P2/P3优势
1. ✅ **RepVGG重参数化：推理时融合为单3×3卷积**
2. ✅ **硬件友好：对RK3588/QCS6490等NPU极度优化**
3. ✅ **更低FLOPs：删除P4头，2头比3头更快**
4. ✅ **成熟架构：原始YOLOv6经过大规模验证**
5. ✅ **更适合motion-guided场景：专注4-32px微小目标**

---

## 选择建议

### 选择ET-YOLO如果：
- 需要检测跨度大的目标（4-60px）
- 对推理速度要求不极致
- 想探索CrossLayer融合的效果

### 选择YOLOv6n P2/P3如果：
- **需要极致的推理速度（边缘设备部署）**
- **目标集中在4-32px范围（motion-guided ROI）**
- **需要RepVGG重参数化的硬件优化**
- **追求最低的计算量和功耗**

---

## 实验建议

建议**同时训练两个模型**进行对比：

```bash
# ET-YOLO 3-scale
torchrun --nproc_per_node=2 tools/train.py \
    --conf configs/et_yolov6n_ard100_roi160_nwd_3scale.py \
    --data data/ard100_roi160_merged.yaml \
    --name et_yolov6n_roi160_3scale

# YOLOv6n P2/P3
torchrun --nproc_per_node=2 tools/train.py \
    --conf configs/yolov6n_ard100_roi160_p2p3_nwd.py \
    --data data/ard100_roi160_merged.yaml \
    --name yolov6n_roi160_p2p3
```

**对比维度**：
1. 训练收敛速度
2. 最终mAP/Recall/Precision
3. 推理延迟（训练后重参数化）
4. RKNN量化后性能
5. 实际场景效果

---

**创建日期**: 2026-08-26  
**用途**: 训练前架构选择参考
