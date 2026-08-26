#!/usr/bin/env python3
"""
Convert YOLOv6n ROI160 (P2-P3) ONNX → RKNN INT8 模型

模型配置：
- 输入尺寸: 160x160
- Batch Size: 4
- 量化方式: INT8 PTQ（使用校验数据集）
- 输出模式: split（pre-sigmoid logits）

关键优化：
1. **置信度单独量化方案**：
   - 分类输出 (cls_s0, cls_s1) 保持为 pre-sigmoid logits
   - Sigmoid 在 CPU 端完成，避免 INT8 量化压缩 0-1 范围导致的精度损失
   - 这是参考 RKNN Model Zoo YOLOv6n 的标准做法

2. **RepVGG 重参数化**：
   - YOLOv6n 模型已在导出时进行 RepVGG 融合
   - 训练时的多分支结构已融合为单个 3x3 卷积，推理加速

3. **量化算法**：
   - 使用 'normal' 算法（标准 PTQ）
   - asymmetric_quantized-8（非对称 INT8 量化）
   - optimization_level=3（最高优化级别）

Usage:
    /home/chenx/rknn-env/bin/python3 convert_roi160_rknn_bs4.py
"""
import sys
from rknn.api import RKNN

# ==================== 配置参数 ====================
ONNX_MODEL   = '/data/workdir/et-yolov6/yolov6n_roi160_p2p3_split.onnx'
RKNN_MODEL   = '/data/workdir/et-yolov6/yolov6n_roi160_p2p3_bs4_int8.rknn'
DATASET      = '/data/workdir/et-yolov6/rknn_calibration_roi640_list.txt'
PLATFORM     = 'rk3588'
BATCH_SIZE   = 4

# 输入预处理参数
# YOLOv6 输入为 uint8 RGB [0, 255]
# RKNN 归一化: x_float = (pixel - mean) / std
# mean=0, std=255 → 映射到 [0, 1]
MEAN_VALUES  = [[0, 0, 0]]
STD_VALUES   = [[255, 255, 255]]

# ==================== 初始化 RKNN ====================
print('=' * 80)
print(f'RKNN 转换配置')
print('=' * 80)
print(f'  ONNX 模型:     {ONNX_MODEL}')
print(f'  RKNN 输出:     {RKNN_MODEL}')
print(f'  校验数据集:    {DATASET}')
print(f'  目标平台:      {PLATFORM}')
print(f'  Batch Size:    {BATCH_SIZE}')
print(f'  量化方式:      INT8 PTQ')
print(f'  置信度处理:    Pre-sigmoid logits (CPU端sigmoid)')
print('=' * 80)

rknn = RKNN(verbose=True)

# ==================== 1. Config ====================
print('\n[1/4] 配置 RKNN...')
ret = rknn.config(
    mean_values=MEAN_VALUES,
    std_values=STD_VALUES,
    target_platform=PLATFORM,
    quantized_algorithm='normal',              # 标准 PTQ 量化
    quantized_dtype='asymmetric_quantized-8',  # 非对称 INT8
    optimization_level=3,                       # 最高优化级别
)
if ret != 0:
    print(f'✗ config 失败: {ret}')
    sys.exit(1)
print('✓ 配置完成')

# ==================== 2. Load ONNX ====================
print('\n[2/4] 加载 ONNX 模型...')
ret = rknn.load_onnx(model=ONNX_MODEL)
if ret != 0:
    print(f'✗ load_onnx 失败: {ret}')
    sys.exit(1)
print('✓ ONNX 加载完成')

# ==================== 3. Build (INT8 Quantization) ====================
print('\n[3/4] 量化构建 (INT8 PTQ)...')
print(f'  使用校验数据集进行量化校准')
print(f'  这将需要几分钟时间...')

ret = rknn.build(
    do_quantization=True,
    dataset=DATASET,
    rknn_batch_size=BATCH_SIZE,  # 设置 batch size = 4
)
if ret != 0:
    print(f'✗ build 失败: {ret}')
    sys.exit(1)
print('✓ 量化构建完成')

# ==================== 4. Export RKNN ====================
print('\n[4/4] 导出 RKNN 模型...')
ret = rknn.export_rknn(RKNN_MODEL)
if ret != 0:
    print(f'✗ export_rknn 失败: {ret}')
    sys.exit(1)

rknn.release()

# ==================== 完成 ====================
import os
file_size = os.path.getsize(RKNN_MODEL) / (1024 * 1024)

print('\n' + '=' * 80)
print('✓ 转换完成!')
print('=' * 80)
print(f'  输出文件: {RKNN_MODEL}')
print(f'  文件大小: {file_size:.2f} MB')
print(f'  Batch Size: {BATCH_SIZE}')
print(f'  输入: images [{BATCH_SIZE}, 3, 160, 160] uint8 RGB')
print(f'  输出: 4 个张量 (每个尺度 2 个)')
print(f'    - reg_s0 [{BATCH_SIZE}, 4, 40, 40]   P2 回归 (INT8)')
print(f'    - cls_s0 [{BATCH_SIZE}, 1, 40, 40]   P2 分类 logits (INT8, 需CPU端sigmoid)')
print(f'    - reg_s1 [{BATCH_SIZE}, 4, 20, 20]   P3 回归 (INT8)')
print(f'    - cls_s1 [{BATCH_SIZE}, 1, 20, 20]   P3 分类 logits (INT8, 需CPU端sigmoid)')
print('=' * 80)
print('\n关于置信度量化的说明:')
print('  ✓ 使用 split 输出模式，分类输出为 pre-sigmoid logits')
print('  ✓ 虽然 logits 也被 INT8 量化，但其动态范围更大（通常 -10 ~ +10）')
print('  ✓ INT8 量化后精度损失远小于直接量化 sigmoid 后的 0-1 范围')
print('  ✓ CPU 端进行 sigmoid 和 anchor decode，保持最终精度')
print('  ✓ 这是 RKNN Model Zoo 推荐的标准做法')
print('=' * 80)
