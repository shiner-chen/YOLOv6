# YOLOv6n ROI160 P2/P3 2-Head Implementation

## 概述

基于原始YOLOv6架构实现的160×160 ROI微小目标检测网络，专门用于motion-guided检测场景。

**核心设计理念**：保留完整Backbone和Neck以提取深层语义特征，仅输出P2/P3两个检测头，实现"高语义分类能力 + 精准定位 + 低计算量"的平衡。

## 架构特点

### 1. 完整的Backbone (EfficientRep)
- **5个stage完整保留**: C1 → C2 → C3 → C4 → C5
- **输出尺度** (160×160输入):
  - C2: 80×80 (stride=2)
  - C3: 40×40 (stride=4) → 送入P2
  - C4: 20×20 (stride=8) → 送入P3
  - C5: 10×10 (stride=16) → 提供深层语义
- **fuse_P2=True**: 启用P2输出
- **RepVGG结构**: 训练时多分支，推理时重参数化为单3×3卷积

### 2. 完整的Neck (RepBiFPANNeckP2P3)
**新增的自定义Neck类**，保留完整的FPN/PAN特征融合路径：

```
Top-down FPN:
  P5 → P4 → P3 → P2
  (深层语义自顶向下融合)

Bottom-up PAN:
  P2 → P3
  (浅层细节自底向上refinement)
```

**关键设计**：
- 保留C5到P4的特征融合（BiFusion）
- 保留C4到P3的特征融合（BiFusion）
- 额外增加P3到P2的上采样分支
- P2和P3通过PAN进行二次融合

### 3. 精简的Head (EffiDeHead 2-layer)
- **只保留2个检测头**: P2 (stride=4) + P3 (stride=8)
- **删除**: P4/P5检测头 (不需要检测大目标)
- **特征图尺寸** (160×160输入):
  - P2: 40×40 → 检测4-20像素目标
  - P3: 20×20 → 检测16-40像素目标

### 4. Anchor设计
针对160×160 ROI中的微小目标优化：

```python
anchors_init=[
    [6,4,   10,8,   14,10],    # P2: 4-20px tiny targets
    [18,14, 26,20,  36,28]     # P3: 16-40px small targets
]
```

### 5. 损失函数
- **NWD (Normalized Wasserstein Distance)**: 专为微小目标设计
- **nwd_ratio=0.6**: 60% NWD + 40% SIoU
- **nwd_constant=14.0**: 针对10-20像素目标 (≈√(10×20))

## 模型性能

### 测试结果 (160×160输入)
```
✓ Model built successfully
✓ Forward pass successful

Total parameters: 3,742,892 (~3.7M)
Trainable parameters: 3,742,858
Model size: 14.28 MB

Output shape: [batch, 2000, 6]
  - 2000 = 40×40 + 20×20 = 1600 + 400 anchors
  - 6 = [x, y, w, h, conf, cls]
```

### 计算量对比
| 模型配置 | 输入尺寸 | 检测头 | FLOPs (估算) | 相对计算量 |
|---------|---------|--------|------------|-----------|
| YOLOv6n 标准 | 640×640 | P3/P4/P5 | ~4.5 GFLOPs | 1.0× |
| YOLOv6n ROI160 P2/P3 | 160×160 | P2/P3 | ~0.3-0.5 GFLOPs | **0.1×** |

**优势**：
- 计算量降低到原来的10%
- 保持160×160 ROI的原始分辨率（不下采样）
- 目标从全图的16×16像素保持为ROI中的16×16像素

## 实现文件

### 1. 配置文件
**`configs/yolov6n_roi160_p2p3.py`**
- Backbone: EfficientRep (完整5-stage)
- Neck: RepBiFPANNeckP2P3 (新增)
- Head: 2-layer (P2/P3)
- Training mode: repvgg (重参数化)

### 2. Neck实现
**`yolov6/models/reppan.py`** - 新增类：
```python
class RepBiFPANNeckP2P3(nn.Module):
    """RepBiFPANNeck with P2+P3 outputs only"""
```

特点：
- 输入: (P2, P3, P4, P5) from backbone
- 输出: [P2_out, P3_out] for detection heads
- 完整的FPN (P5→P4→P3→P2) + PAN (P2→P3) 路径

### 3. Head修改
**`yolov6/models/effidehead.py`** - 修改：
1. `Detect.__init__`: 添加 `num_layers==2` 的stride支持
2. `build_effidehead_layer`: 动态构建head layers（支持2/3/4层）

### 4. 测试脚本
**`test_p2p3_model.py`** - 验证模型构建和前向传播

## 使用方法

### 1. 训练
```bash
source /home/adlink/chenx/rknn-env/bin/activate

torchrun --nproc_per_node=2 --master_port=29500 \
    tools/train.py \
    --conf configs/yolov6n_roi160_p2p3.py \
    --data data/ard100_roi160.yaml \
    --img-size 160 \
    --batch-size 128 \
    --epochs 400 \
    --device 0,1 \
    --workers 4 \
    --output-dir runs/train \
    --name yolov6n_roi160_p2p3 \
    --eval-interval 5
```

### 2. 推理
```bash
python tools/infer.py \
    --weights runs/train/yolov6n_roi160_p2p3/weights/best_ckpt.pt \
    --source <image_or_video> \
    --img-size 160 \
    --conf-thres 0.4 \
    --iou-thres 0.5
```

### 3. 导出ONNX
```bash
python deploy/ONNX/export_onnx.py \
    --weights runs/train/yolov6n_roi160_p2p3/weights/best_ckpt.pt \
    --img-size 160 \
    --batch-size 1 \
    --simplify
```

### 4. 重参数化（推理加速）
```bash
python tools/reparameterize.py \
    --weights runs/train/yolov6n_roi160_p2p3/weights/best_ckpt.pt \
    --output runs/train/yolov6n_roi160_p2p3/weights/best_ckpt_reparam.pt
```

## Motion-Guided推理流程

完整的运行时pipeline：

```
1. Motion Detection (全图)
   └─> 输入: 1920×1080原始帧
   └─> 输出: Blob中心点 [(cx1,cy1), (cx2,cy2), ...]

2. ROI Extraction (带margin)
   └─> 以blob中心截取160×160 ROI
   └─> 留20-30像素margin防止边缘截断
   └─> 边界padding处理

3. Batch Inference
   └─> 固定batch (例如最多4个ROI)
   └─> 不足则zero-padding
   └─> YOLOv6n-P2P3推理: 160×160 → [2000, 6]

4. 后处理
   └─> NMS过滤
   └─> 坐标映射回全图: x_global = x_patch + roi_offset
   └─> 置信度过滤
```

## 数据准备

### 方案1: 从现有数据集裁剪ROI
```bash
python tools/prepare_ard100_roi160.py \
    --input data/ard100/images \
    --output data/ard100_roi160/images \
    --anno data/ard100/labels \
    --roi-size 160 \
    --margin 30
```

### 方案2: 使用motion先验生成ROI
```bash
python tools/generate_roi_from_motion.py \
    --video <path_to_video> \
    --output data/ard100_roi160_motion/ \
    --roi-size 160
```

## 设计优势总结

### ✅ 为什么保留完整Backbone/Neck？
**避免浅层特征的分类能力不足问题**：
- P2/P3是浅层特征，感受野小，语义信息弱
- 容易将树叶、飞鸟、云朵误识别为无人机
- 通过保留C4/C5并融合到P2/P3，注入深层语义特征
- **用高层语义做分类抑制，用浅层特征做精准定位**

### ✅ 为什么只删除检测头而不删除Neck？
- 删除P4/P5检测头：节省大量输出卷积和NMS计算
- 保留完整Neck：计算量很小（5×5特征图的卷积几乎可忽略）
- 在160×160输入下，C5特征图只有10×10，对其做卷积的FLOPs极低

### ✅ 为什么用原始YOLOv6而不是ET-YOLO？
- **RepVGG重参数化优势**: 推理时融合为单3×3卷积，硬件极度友好
- **NPU/GPU优化**: 重参后的结构对RK3588/QCS6490等边缘芯片性能最优
- **官方验证**: YOLOv6是经过大规模验证的成熟架构

### ✅ 160×160输入的优势
- **保持原始分辨率**: ROI切片不下采样，目标细节完整
- **特征图充足**: P2(40×40)为每个10px目标分配约4个grid
- **计算量极低**: 相比640×640下降90%

## 后续优化方向

### 1. 数据增强
- [ ] 针对motion切片的特殊增强策略
- [ ] 边缘padding的随机变换
- [ ] 多尺度ROI训练（160/192/224）

### 2. 模型压缩
- [ ] 进一步减少backbone宽度（width_multiple < 0.25）
- [ ] 知识蒸馏（用大模型指导）
- [ ] 量化训练（QAT for INT8）

### 3. 部署优化
- [ ] RKNN转换与优化
- [ ] TensorRT FP16推理
- [ ] 多ROI并行处理的dynamic batch优化

### 4. 系统集成
- [ ] Motion检测与YOLO的联合训练
- [ ] 端到端的ROI提取+检测pipeline
- [ ] 时序信息利用（跨帧关联）

## 相关文件清单

```
configs/
  └─ yolov6n_roi160_p2p3.py          # 模型配置

yolov6/models/
  ├─ reppan.py                        # 添加RepBiFPANNeckP2P3类
  ├─ effidehead.py                    # 修改支持2-layer head
  └─ yolo.py                          # (无需修改)

test_p2p3_model.py                    # 模型测试脚本

docs/
  └─ yolov6n_roi160_p2p3_implementation.md  # 本文档
```

## Git分支

- **当前分支**: `et-yolov6n-roi160-p2p3`
- **基于**: `et-yolov6s-nwd` (包含NWD loss实现)
- **主要改动**: 
  - 新增RepBiFPANNeckP2P3 Neck
  - 修改effidehead支持2-layer
  - 新增配置文件

## 性能预期

基于讨论和理论分析：

| 指标 | 640×640 全图 | 160×160 ROI P2/P3 |
|-----|-------------|------------------|
| 输入分辨率 | 640×640 | 160×160 |
| 目标物理尺寸 | 16×16px → 1.3×1.3px | 16×16px → 16×16px |
| FLOPs | ~4.5 G | ~0.3-0.5 G |
| 推理延迟 (RK3588) | ~50ms | **~2-3ms** |
| 召回率 (小目标) | 中等 | **极高** |
| 虚警率 | 中等 | 低 (深层语义抑制) |

## 参考资料

1. YOLOv6 官方仓库: https://github.com/meituan/YOLOv6
2. NWD Loss论文: Normalized Wasserstein Distance for Tiny Object Detection
3. 讨论记录: Motion-guided ROI detection架构设计

---

**创建日期**: 2026-08-26  
**作者**: xuan chen  
**分支**: et-yolov6n-roi160-p2p3
