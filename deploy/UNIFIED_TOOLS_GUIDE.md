# YOLOv6 RKNN 统一转换工具使用指南

## 概述

本工具集提供了统一的YOLOv6模型RKNN转换流程，支持：
- ✅ 自动检测检测头数量（P2-P3双尺度 / P2-P5四尺度等）
- ✅ 自动RepVGG重参数化优化
- ✅ 任意batch size支持（1, 2, 4, 8等）
- ✅ Split output模式（置信度单独量化）
- ✅ 完整的性能评估（推理+后处理）

## 工具组成

### 1. export_yolov6_rknn.py - ONNX导出工具
将PyTorch权重导出为ONNX格式，自动进行RepVGG融合和检测头配置。

### 2. convert_yolov6_rknn.py - RKNN转换工具
将ONNX模型转换为RKNN INT8量化模型，支持任意batch size。

### 3. benchmark_yolov6_rknn.py - 性能评估工具
在RK3588设备上测试RKNN模型性能，自动适配检测头配置。

---

## 使用示例

### 示例1：ROI160 双尺度检测头 (P2+P3)

**步骤1：导出ONNX**
```bash
cd /home/chenx/workdir/et-yolov6n
/home/chenx/rknn-env/bin/python3 /data/workdir/et-yolov6/export_yolov6_rknn.py \
    --weights /data/workdir/et-yolov6/yolov6n_nwd_p2-3_roi160.pt \
    --img-size 160 \
    --output /data/workdir/et-yolov6/yolov6n_roi160_split.onnx
```

**输出示例：**
```
✓ 融合了 35 个 RepVGG 块
✓ 检测头配置:
    类别数 (nc): 1
    检测头数 (nl): 2
    步长 (stride): [4, 8]
    检测头类型: P2+P3 (双尺度)
✓ 导出成功! (13.62 MB)
```

**步骤2：转换RKNN (batch=4)**
```bash
/home/chenx/rknn-env/bin/python3 /data/workdir/et-yolov6/convert_yolov6_rknn.py \
    --onnx /data/workdir/et-yolov6/yolov6n_roi160_split.onnx \
    --output /data/workdir/et-yolov6/yolov6n_roi160_bs4_int8.rknn \
    --batch-size 4 \
    --dataset /data/workdir/et-yolov6/rknn_calibration_roi640_list.txt \
    --img-size 160
```

**步骤3：性能测试（在RK3588设备上）**
```bash
# 上传文件到设备
sshpass -p firefly scp yolov6n_roi160_bs4_int8.rknn benchmark_yolov6_rknn.py \
    firefly@192.168.1.34:/home/firefly/workspace/test/

# 运行测试
sshpass -p firefly ssh firefly@192.168.1.34 \
    "cd /home/firefly/workspace/test && \
    /home/firefly/rknn-venv/bin/python3 benchmark_yolov6_rknn.py \
        --model yolov6n_roi160_bs4_int8.rknn \
        --img-size 160 \
        --batch-size 4"
```

---

### 示例2：ROI320 四尺度检测头 (P2+P3+P4+P5)

**步骤1：导出ONNX**
```bash
cd /home/chenx/workdir/et-yolov6n
/home/chenx/rknn-env/bin/python3 /data/workdir/et-yolov6/export_yolov6_rknn.py \
    --weights /data/workdir/et-yolov6/yolov6n+p2+nwd_roi320.pt \
    --img-size 320 \
    --output /data/workdir/et-yolov6/yolov6n_roi320_split.onnx
```

**输出示例：**
```
✓ 融合了 43 个 RepVGG 块
✓ 检测头配置:
    类别数 (nc): 1
    检测头数 (nl): 4
    步长 (stride): [4, 8, 16, 32]
    检测头类型: P2+P3+P4+P5 (四尺度)
✓ 导出成功! (18.59 MB)
```

**步骤2：转换多个batch size**
```bash
# Batch size = 1 (低延迟)
/home/chenx/rknn-env/bin/python3 /data/workdir/et-yolov6/convert_yolov6_rknn.py \
    --onnx /data/workdir/et-yolov6/yolov6n_roi320_split.onnx \
    --output /data/workdir/et-yolov6/yolov6n_roi320_bs1_int8.rknn \
    --batch-size 1 \
    --dataset /data/workdir/et-yolov6/rknn_calibration_roi320_list.txt \
    --img-size 320

# Batch size = 4 (高吞吐)
/home/chenx/rknn-env/bin/python3 /data/workdir/et-yolov6/convert_yolov6_rknn.py \
    --onnx /data/workdir/et-yolov6/yolov6n_roi320_split.onnx \
    --output /data/workdir/et-yolov6/yolov6n_roi320_bs4_int8.rknn \
    --batch-size 4 \
    --dataset /data/workdir/et-yolov6/rknn_calibration_roi320_list.txt \
    --img-size 320
```

**步骤3：性能对比测试**
```bash
# 上传文件
sshpass -p firefly scp yolov6n_roi320_bs*.rknn benchmark_yolov6_rknn.py \
    firefly@192.168.1.34:/home/firefly/workspace/test/

# 对比测试两个模型
sshpass -p firefly ssh firefly@192.168.1.34 \
    "cd /home/firefly/workspace/test && \
    /home/firefly/rknn-venv/bin/python3 benchmark_yolov6_rknn.py \
        --model yolov6n_roi320_bs1_int8.rknn yolov6n_roi320_bs4_int8.rknn \
        --img-size 320 320 \
        --batch-size 1 4"
```

**性能对比输出：**
```
性能对比总结
================================================================================
模型 1: yolov6n_roi320_bs1_int8.rknn
  输入尺寸:     320x320
  检测尺度:     4个
  Batch Size:   1
  NPU推理:      18.37 ms
  吞吐量:       54.4 FPS
  后处理:       2.36 ms
  端到端:       20.74 ms

模型 2: yolov6n_roi320_bs4_int8.rknn
  输入尺寸:     320x320
  检测尺度:     4个
  Batch Size:   4
  NPU推理:      36.86 ms
  吞吐量:       108.5 FPS (单图9.22ms)
================================================================================
```

---

## 参数说明

### export_yolov6_rknn.py

| 参数 | 必需 | 说明 | 示例 |
|------|------|------|------|
| `--weights` | 是 | PyTorch权重文件路径 | `yolov6n_roi320.pt` |
| `--img-size` | 是 | 输入图像尺寸 | `160`, `320`, `640` |
| `--output` | 是 | 输出ONNX文件路径 | `yolov6n_roi320.onnx` |
| `--opset` | 否 | ONNX opset版本 (默认13) | `11`, `13` |
| `--device` | 否 | 运行设备 (默认cpu) | `cpu`, `cuda` |

### convert_yolov6_rknn.py

| 参数 | 必需 | 说明 | 示例 |
|------|------|------|------|
| `--onnx` | 是 | 输入ONNX模型路径 | `yolov6n_roi320.onnx` |
| `--output` | 是 | 输出RKNN模型路径 | `yolov6n_roi320_bs4.rknn` |
| `--batch-size` | 是 | Batch size | `1`, `2`, `4`, `8` |
| `--dataset` | 是 | 量化校验数据集列表 | `calibration_list.txt` |
| `--img-size` | 是 | 输入图像尺寸 | `320` |
| `--platform` | 否 | 目标平台 (默认rk3588) | `rk3588`, `rk3576` |
| `--quantize-algorithm` | 否 | 量化算法 (默认normal) | `normal`, `mmse` |
| `--optimization-level` | 否 | 优化级别 (默认3) | `0`, `1`, `2`, `3` |

### benchmark_yolov6_rknn.py

| 参数 | 必需 | 说明 | 示例 |
|------|------|------|------|
| `--model` | 是 | RKNN模型路径（可多个） | `model1.rknn model2.rknn` |
| `--img-size` | 是 | 输入尺寸（与模型对应） | `320 320` |
| `--batch-size` | 是 | Batch size（与模型对应） | `1 4` |
| `--warmup` | 否 | 预热次数 (默认10) | `10`, `20` |
| `--test-runs` | 否 | 测试次数 (默认100) | `50`, `100` |
| `--core-mask` | 否 | NPU核心 (默认0_1_2) | `0`, `0_1`, `0_1_2` |
| `--conf-thresh` | 否 | 置信度阈值 (默认0.25) | `0.25`, `0.5` |

---

## 关键特性

### 1. 自动RepVGG重参数化 ✅
- 训练时：3分支结构（dense 3×3 + 1×1 + identity）
- 推理时：自动融合为单个3×3卷积
- 性能提升：约1.5-2倍

### 2. 自动检测头适配 ✅
- 支持任意数量的检测尺度
- 自动推断stride配置
- 输出信息清晰展示

### 3. Split Output模式 ✅
- 分类输出：pre-sigmoid logits
- CPU端sigmoid：避免INT8量化精度损失
- 精度提升：约5.8%

### 4. 单目标优化后处理 ✅
- 直接argmax选择最高置信度
- 避免NMS开销
- 性能提升：约9.9倍

---

## 验证清单

使用统一工具时，请确认：
- ✅ RepVGG块融合数量正确（ROI160=35, ROI320=43）
- ✅ 检测头数量正确识别（P2+P3=2, P2-P5=4）
- ✅ stride配置正确（[4,8] 或 [4,8,16,32]）
- ✅ 量化数据集路径正确
- ✅ 模型文件大小合理（ROI160≈3.9MB, ROI320≈5.3MB）

---

## 故障排查

### 问题1：RepVGG块数量为0
```
⚠ 警告: 未找到RepVGG块
```
**原因**：模型可能未使用RepVGG结构  
**解决**：检查模型架构，确认是RepVGG类型

### 问题2：检测头数量不匹配
```
✗ 未找到Detect检测头
```
**原因**：模型结构不兼容  
**解决**：确认模型是YOLOv6架构

### 问题3：量化失败
```
✗ build失败: -1
```
**原因**：校验数据集路径错误或格式不正确  
**解决**：检查dataset文件存在且格式正确

### 问题4：设备推理失败
```
ModuleNotFoundError: No module named 'rknnlite'
```
**原因**：未使用正确的Python环境  
**解决**：使用 `/home/firefly/rknn-venv/bin/python3`

---

## 性能基准

### ROI160 (P2+P3, 2尺度)
- 模型大小：3.9 MB
- Batch=1: 13.87 ms (72.1 FPS)
- Batch=4: 吞吐量提升约1.5x

### ROI320 (P2-P5, 4尺度)  
- 模型大小：5.3 MB
- Batch=1: 18.37 ms (54.4 FPS)
- Batch=4: 108.5 FPS (吞吐量提升2.0x)

---

## 总结

统一工具的优势：
1. **简化流程**：3个脚本完成全部工作
2. **自动化**：自动检测配置，减少人工错误
3. **可扩展**：支持任意ROI尺寸和检测头配置
4. **一致性**：统一的参数命名和输出格式
5. **可维护**：修改一处，所有配置受益

推荐工作流：
```
PyTorch模型 → export_yolov6_rknn.py → ONNX模型
            ↓
ONNX模型 → convert_yolov6_rknn.py → RKNN模型(多个batch size)
            ↓
RKNN模型 → benchmark_yolov6_rknn.py → 性能报告
```
