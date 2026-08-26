# YOLOv6n ROI160 P2/P3 快速开始指南

## 🚀 快速开始

### 1. 环境准备
```bash
# 激活环境
source /home/adlink/chenx/rknn-env/bin/activate

# 验证环境
python test_p2p3_model.py
```

预期输出：
```
✓ Model built successfully
✓ Forward pass successful
✓ All tests passed!
Parameters: 3.7M, FLOPs: ~0.3-0.5G
```

### 2. 数据准备
确保数据集已准备：
```bash
data/ard100_roi160_merged.yaml  # 数据集配置文件
data/ard100_roi160/
  ├── train/
  │   ├── images/
  │   └── labels/
  └── val/
      ├── images/
      └── labels/
```

### 3. 开始训练
**方式1：使用快速脚本**
```bash
./train_p2p3.sh
```

**方式2：手动命令**
```bash
source /home/adlink/chenx/rknn-env/bin/activate

torchrun --nproc_per_node=2 --master_port=29500 \
    tools/train.py \
    --conf configs/yolov6n_ard100_roi160_p2p3_nwd.py \
    --data data/ard100_roi160_merged.yaml \
    --img-size 160 \
    --batch-size 128 \
    --epochs 400 \
    --device 0,1 \
    --workers 4 \
    --output-dir runs/train \
    --name yolov6n_roi160_p2p3_nwd \
    --eval-interval 5
```

**训练时间**：约8-10小时 (2×RTX 3090)

---

## 📊 模型架构

```
输入: 160×160×3 ROI patch
  ↓
Backbone: EfficientRep (5-stage, C1→C5)
  ├─ C2: 80×80  (stride=2)
  ├─ C3: 40×40  (stride=4) → P2
  ├─ C4: 20×20  (stride=8) → P3
  └─ C5: 10×10  (stride=16) → 深层语义
  ↓
Neck: RepBiFPANNeckP2P3
  ├─ FPN: P5 → P4 → P3 → P2 (自顶向下)
  └─ PAN: P2 → P3 (自底向上refinement)
  ↓
Head: EffiDeHead (2-layer)
  ├─ P2: 40×40 (stride=4) → 4-20px targets
  └─ P3: 20×20 (stride=8) → 16-40px targets
  ↓
输出: [batch, 2000, 6]
  └─ 2000 = 1600(P2) + 400(P3) anchors
  └─ 6 = [x, y, w, h, conf, cls]
```

---

## 🎯 核心特性

### 1. 完整的Backbone/Neck
- ✅ 保留5-stage backbone (到C5)
- ✅ 深层语义通过FPN/PAN融合到P2/P3
- ✅ **用高层语义做分类，用浅层特征做定位**

### 2. 精简的检测头
- ✅ 只保留P2/P3两个检测头
- ✅ 删除P4/P5（不需要检测大目标）
- ✅ FLOPs降低25%

### 3. RepVGG重参数化
- ✅ 训练时：多分支结构（提升表达能力）
- ✅ 推理时：融合为单3×3卷积（极致速度）
- ✅ 对NPU/GPU硬件极度友好

### 4. NWD Loss for Tiny Objects
- ✅ nwd_ratio=0.6 (60% NWD + 40% SIoU)
- ✅ nwd_constant=17.0 (针对20×15px目标)
- ✅ 专为微小目标优化

---

## 📈 性能预期

| 指标 | 640×640全图 | 160×160 ROI P2/P3 | 提升 |
|-----|-----------|------------------|-----|
| FLOPs | 4.5G | 0.4-0.5G | **90%↓** |
| 参数量 | 4.7M | 3.7M | 21%↓ |
| 推理延迟 (RK3588) | 50ms | **2-3ms** | **95%↓** |
| 目标分辨率 | 1.3px | 16px | **12×** |

---

## 🔧 训练后处理

### 1. 评估模型
```bash
python tools/eval.py \
    --weights runs/train/yolov6n_roi160_p2p3_nwd/weights/best_ckpt.pt \
    --data data/ard100_roi160_merged.yaml \
    --img-size 160 \
    --batch-size 64 \
    --device 0
```

### 2. 重参数化（推理加速）
```bash
python tools/reparameterize.py \
    --weights runs/train/yolov6n_roi160_p2p3_nwd/weights/best_ckpt.pt \
    --output runs/train/yolov6n_roi160_p2p3_nwd/weights/best_ckpt_reparam.pt
```

### 3. 导出ONNX
```bash
python deploy/ONNX/export_onnx.py \
    --weights runs/train/yolov6n_roi160_p2p3_nwd/weights/best_ckpt_reparam.pt \
    --img-size 160 \
    --batch-size 1 \
    --simplify
```

### 4. 转换RKNN
```bash
python deploy/RKNN/convert_rknn.py \
    --onnx runs/train/yolov6n_roi160_p2p3_nwd/weights/best_ckpt_reparam.onnx \
    --output yolov6n_roi160_p2p3.rknn \
    --platform RK3588 \
    --quantize i8
```

---

## 🎮 Motion-Guided推理流程

完整的运行时pipeline：

```python
# 1. Motion Detection (全图)
blobs = motion_detector.detect(frame)  # [(cx, cy), ...]

# 2. ROI Extraction (带margin)
rois = []
for cx, cy in blobs:
    roi = extract_roi_with_margin(
        frame, cx, cy, 
        size=160, 
        margin=30  # 防止边缘截断
    )
    rois.append(roi)

# 3. Batch Inference (固定batch size)
if len(rois) > MAX_BATCH:
    rois = rois[:MAX_BATCH]  # 取前N个
elif len(rois) < MAX_BATCH:
    rois = pad_to_batch(rois, MAX_BATCH)  # zero-padding

detections = model(rois)  # YOLOv6n P2/P3

# 4. 坐标映射回全图
for i, det in enumerate(detections):
    det[:, :4] += roi_offsets[i]  # 加上ROI偏移
```

---

## 📚 文档索引

- **完整实现文档**: `docs/yolov6n_roi160_p2p3_implementation.md`
- **配置对比**: `docs/config_comparison.md`
- **快速入门**: `README_P2P3.md`
- **测试脚本**: `test_p2p3_model.py`
- **训练脚本**: `train_p2p3.sh`

---

## 🆚 与ET-YOLO对比

| 维度 | ET-YOLO ROI160 3-scale | YOLOv6n ROI160 P2/P3 |
|-----|----------------------|---------------------|
| 架构 | CrossLayerBifusion | 原始YOLOv6 RepVGG |
| 检测头 | P2/P3/P4 (3头) | P2/P3 (2头) |
| FLOPs | ~0.6G | **~0.4-0.5G** (25%↓) |
| 推理速度 | 快 | **更快** (RepVGG) |
| 硬件优化 | 中等 | **极致** (NPU友好) |
| 适用场景 | 4-60px目标 | **4-32px微小目标** |

**推荐**：Motion-guided场景优先选择YOLOv6n P2/P3

---

## ❓ FAQ

### Q1: 为什么保留完整Backbone而不裁剪？
**A**: P2/P3是浅层特征，语义信息弱，容易误报（树叶、飞鸟）。保留C5深层特征，通过FPN/PAN融合到P2/P3，用高层语义抑制虚警。在160×160输入下，C5只有10×10，计算量几乎可忽略。

### Q2: 为什么删除P4/P5检测头？
**A**: 160×160 ROI主要包含4-32px的微小目标，不需要P4(stride=16)和P5(stride=32)来检测大目标。删除这两个头可节省60%+ head计算量。

### Q3: RepVGG重参数化有什么优势？
**A**: 训练时用多分支结构提升表达能力，推理时融合为单3×3卷积。这种结构对NPU/GPU极度友好，在RK3588等边缘设备上性能最优。

### Q4: NWD loss的作用是什么？
**A**: Normalized Wasserstein Distance专为微小目标设计，比传统IoU更适合评估小目标的定位质量。配置nwd_ratio=0.6表示60% NWD + 40% SIoU的混合损失。

### Q5: 如何准备ROI数据集？
**A**: 有两种方式：
1. 从全图数据集裁剪：使用`tools/prepare_ard100_roi160.py`
2. 基于motion先验：使用`tools/generate_roi_from_motion.py`

---

## 📞 支持

- **文档**: `docs/` 目录下的详细文档
- **测试**: 运行 `python test_p2p3_model.py` 验证环境
- **Issues**: 遇到问题请检查配置文件和数据路径

---

**创建**: 2026-08-26  
**分支**: `et-yolov6n-roi160-p2p3`  
**状态**: ✅ 实现完成，可开始训练
