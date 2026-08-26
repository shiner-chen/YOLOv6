# YOLOv6n ROI160 RKNN 部署指南

本目录包含将 YOLOv6n ROI160 模型部署到 RKNN (RK3588) 平台的完整工具链。

## 文件说明

### 核心脚本

| 文件 | 功能 | 用途 |
|------|------|------|
| `export_roi160_split_onnx.py` | ONNX 导出 | 将 PyTorch 权重导出为 split 模式 ONNX |
| `convert_roi160_rknn_bs4.py` | RKNN 转换 | 将 ONNX 转换为 INT8 量化的 RKNN 模型 |
| `test_rknn_corrected.py` | 推理测试 | 在 RK3588 设备上运行推理和性能测试 |
| `verify_repvgg_optimization.py` | 优化验证 | 验证 RepVGG 重参数化是否生效 |

## 使用流程

### 步骤 1: ONNX 导出

在**开发机**上运行（需要 RKNN 虚拟环境）：

```bash
cd /home/chenx/workdir/et-yolov6n
source /home/chenx/rknn-env/bin/activate

python3 deploy/export_roi160_split_onnx.py
```

**输入**：
- 权重文件：`/data/workdir/et-yolov6/yolov6n_nwd_p2-3_roi160.pt`

**输出**：
- ONNX 文件：`/data/workdir/et-yolov6/yolov6n_roi160_p2p3_split.onnx` (13.62 MB)
- 输出格式：4 个张量 (reg_s0, cls_s0, reg_s1, cls_s1)

**关键特性**：
- ✅ RepVGG 重参数化（35 个块融合）
- ✅ Split 输出模式（pre-sigmoid logits）
- ✅ 双尺度检测 (P2/P3, stride 4/8)

### 步骤 2: RKNN 转换

继续在**开发机**上运行：

```bash
python3 deploy/convert_roi160_rknn_bs4.py
```

**输入**：
- ONNX 文件：`yolov6n_roi160_p2p3_split.onnx`
- 校验数据集：`rknn_calibration_roi640_list.txt` (500 张图像)

**输出**：
- RKNN 文件：`yolov6n_roi160_p2p3_bs4_int8.rknn` (3.87 MB)

**配置**：
- 量化方式：INT8 PTQ
- Batch Size：4
- 目标平台：RK3588
- 优化级别：3

**转换时间**：约 2-3 分钟

### 步骤 3: 部署到设备

将模型和测试脚本传输到 **RK3588 设备**：

```bash
# 从开发机传输文件
scp yolov6n_roi160_p2p3_bs4_int8.rknn firefly@192.168.1.34:/home/firefly/workspace/et-yolov6/
scp deploy/test_rknn_corrected.py firefly@192.168.1.34:/home/firefly/workspace/et-yolov6/
```

### 步骤 4: 在设备上测试

SSH 登录到设备并运行测试：

```bash
ssh firefly@192.168.1.34
cd /home/firefly/workspace/et-yolov6
source ~/rknn-venv/bin/activate

python3 test_rknn_corrected.py
```

**测试内容**：
- 推理性能测试
- 端到端延迟测试
- 检测结果可视化

**预期性能**：
- NPU 推理：12.36 ms
- 后处理：1.51 ms
- 端到端：13.87 ms (72.1 FPS)

## 验证优化

验证 RepVGG 重参数化是否正确应用：

```bash
cd /home/chenx/workdir/et-yolov6n
source /home/chenx/rknn-env/bin/activate

python3 deploy/verify_repvgg_optimization.py
```

**检查项**：
- ✅ RepVGG 重参数化（35/35 块融合）
- ✅ ONNX 结构优化（58 个 Conv，0 个 Add）
- ✅ RKNN 模型大小（3.87 MB）

## 关键技术点

### 1. RepVGG 重参数化

训练时的多分支结构融合为推理时的单分支：

```python
for m in model.modules():
    if isinstance(m, RepVGGBlock):
        m.switch_to_deploy()  # 融合为单个 3×3 卷积
```

**效果**：计算量减少 30-40%，预期加速 1.5-2.0x

### 2. Split 输出模式

输出 pre-sigmoid logits，避免 INT8 量化对 0-1 范围的压缩：

```python
# 输出：reg_s0, cls_s0, reg_s1, cls_s1
# cls_sX 是 pre-sigmoid logits，需要在 CPU 端做 sigmoid
```

**效果**：精度损失从 7.3% 降至 1.5%（提升 5.8%）

### 3. 正确的 dist2bbox 解码

YOLOv6 输出是 distance (ltrb) 格式，解码时**必须乘以 stride**：

```python
def dist2bbox_corrected(distance, anchor_points, stride):
    # 关键：distance 必须乘以 stride！
    lt = distance[:, :2] * stride
    rb = distance[:, 2:] * stride
    
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    
    # 转为中心点+宽高
    ...
```

**问题排查**：
- ❌ 不乘 stride → 检测框太小
- ❌ 使用 exp() → 检测框太大
- ✅ distance × stride → 检测框正确

### 4. 单目标优化

单类别检测场景，直接选最高置信度框，无需 NMS：

```python
max_idx = scores.argmax()
max_conf = scores[max_idx]

if max_conf > threshold:
    return boxes[max_idx]  # 只返回 1 个框
```

**效果**：后处理提速 6.7x (10.19ms → 1.51ms)

## 性能对比

| 版本 | 后处理 | 端到端 | FPS | 检测框 |
|------|--------|--------|-----|--------|
| 原版 (NMS) | 10.19 ms | 22.41 ms | 44.6 | ❌ 错误 |
| 优化版 | 1.51 ms | 13.87 ms | 72.1 | ✅ 正确 |
| **提升** | **6.7x** | **1.62x** | **1.62x** | **修正** |

## 模型规格

### 输入

```
格式: uint8 RGB
尺寸: [4, 3, 160, 160]
预处理: (pixel - 0) / 255 → [0, 1]
```

### 输出

```
reg_s0: [4, 4, 40, 40]   # P2 回归 (stride 4)
cls_s0: [4, 1, 40, 40]   # P2 分类 logits (pre-sigmoid)
reg_s1: [4, 4, 20, 20]   # P3 回归 (stride 8)
cls_s1: [4, 1, 20, 20]   # P3 分类 logits (pre-sigmoid)
```

### 模型信息

- **参数量**：~4.7M
- **FP32 大小**：7.83 MB
- **INT8 大小**：3.87 MB
- **压缩比**：2.0x

## 故障排查

### 问题 1: 检测框过大

**原因**：使用了错误的 decode 方法（exp() 导致爆炸）

**解决**：使用 `dist2bbox` 格式，distance × stride

### 问题 2: 检测框过小

**原因**：漏掉了 stride 乘法

**解决**：`lt = distance[:, :2] * stride`

### 问题 3: 性能不达预期

**分析**：
- NPU 推理 12.36ms 是合理的（小模型内存密集）
- 原版后处理 10.19ms 是瓶颈（NMS 占 7.5ms）

**优化**：
- ✅ 单目标优化 → 1.51ms（提速 6.7x）
- 可选：C++ 实现 → 0.4-0.6ms（再提速 3x）

### 问题 4: 推理失败

**检查**：
1. 虚拟环境是否激活：`source ~/rknn-venv/bin/activate`
2. 模型文件路径是否正确
3. 输入图像格式是否为 uint8 RGB

## 进一步优化

### 1. C++ 后处理实现

当前 Python 实现：1.51 ms

预期 C++ + NEON：
- 后处理：0.4-0.6 ms
- 端到端：12.86 ms (**78 FPS**)

### 2. Batch 推理

结合 batch=4 和单目标优化：
- 吞吐量：210 FPS
- 单帧成本：4.75 ms

### 3. 多核 NPU 测试

当前使用 3 核，可测试单核性能：
```python
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
```

## 参考文档

详细文档请查看 `docs/` 目录：

- `ROI160_RKNN_转换报告.md` - 完整技术文档
- `RK3588_性能评估报告.md` - 性能测试详情
- `单目标优化报告.md` - 后处理优化分析
- `最终解决报告.md` - 问题排查记录
- `项目完整总结.md` - 项目总览

## 联系方式

- 项目路径：`/home/chenx/workdir/et-yolov6n/deploy/`
- 测试设备：`firefly@192.168.1.34`

---

**最后更新**: 2026-08-26  
**版本**: v1.0  
**状态**: ✅ 完全验证通过
