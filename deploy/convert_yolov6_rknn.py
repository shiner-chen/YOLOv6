#!/usr/bin/env python3
"""
YOLOv6 统一RKNN转换脚本 - 支持任意batch size和检测头配置

功能特性：
- 自动检测ONNX模型的检测头数量
- 支持任意batch size (1, 2, 4, 8等)
- INT8 PTQ量化
- 自动配置输入预处理
- 支持多种量化数据集

Usage:
    # ROI160 batch=4
    python convert_yolov6_rknn.py \
        --onnx yolov6n_roi160_split.onnx \
        --output yolov6n_roi160_bs4_int8.rknn \
        --batch-size 4 \
        --dataset rknn_calibration_roi640_list.txt \
        --img-size 160

    # ROI320 batch=1
    python convert_yolov6_rknn.py \
        --onnx yolov6n_roi320_split.onnx \
        --output yolov6n_roi320_bs1_int8.rknn \
        --batch-size 1 \
        --dataset rknn_calibration_roi320_list.txt \
        --img-size 320
"""
import argparse
import sys
import os
import onnx
from pathlib import Path
from rknn.api import RKNN


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv6 RKNN转换工具')
    parser.add_argument('--onnx', type=str, required=True,
                        help='输入ONNX模型路径')
    parser.add_argument('--output', type=str, required=True,
                        help='输出RKNN模型路径')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (默认: 1)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='量化校验数据集列表文件')
    parser.add_argument('--img-size', type=int, required=True,
                        help='输入图像尺寸 (用于显示输出形状)')
    parser.add_argument('--platform', type=str, default='rk3588',
                        choices=['rk3588', 'rk3576', 'rk3562'],
                        help='目标平台 (默认: rk3588)')
    parser.add_argument('--quantize-algorithm', type=str, default='normal',
                        choices=['normal', 'mmse'],
                        help='量化算法 (默认: normal)')
    parser.add_argument('--optimization-level', type=int, default=3,
                        choices=[0, 1, 2, 3],
                        help='优化级别 0-3 (默认: 3=最高)')
    return parser.parse_args()


def get_output_info(onnx_path, img_size, batch_size):
    """
    从ONNX模型中提取输出信息

    Returns:
        list of dict: [{name, stride, grid_h, grid_w, type}, ...]
    """
    model = onnx.load(onnx_path)
    outputs = []

    output_tensors = model.graph.output
    num_scales = len(output_tensors) // 2  # 每个尺度2个输出(reg+cls)

    print(f'\n检测头配置:')
    print(f'  检测尺度数: {num_scales}')

    for i in range(num_scales):
        reg_name = output_tensors[i*2].name
        cls_name = output_tensors[i*2+1].name

        # 从cls输出推断stride
        cls_shape = output_tensors[i*2+1].type.tensor_type.shape
        grid_h = cls_shape.dim[2].dim_value
        grid_w = cls_shape.dim[3].dim_value
        stride = img_size // grid_h

        outputs.append({
            'scale': i,
            'reg_name': reg_name,
            'cls_name': cls_name,
            'stride': stride,
            'grid_h': grid_h,
            'grid_w': grid_w,
        })

        print(f'  尺度 {i}: stride={stride}, grid={grid_h}x{grid_w}')
        print(f'    {reg_name}: [{batch_size}, 4, {grid_h}, {grid_w}]')
        print(f'    {cls_name}: [{batch_size}, 1, {grid_h}, {grid_w}]')

    return outputs


def main():
    args = parse_args()

    # 验证输入文件
    if not Path(args.onnx).exists():
        print(f'✗ ONNX文件不存在: {args.onnx}')
        sys.exit(1)

    if not Path(args.dataset).exists():
        print(f'✗ 数据集文件不存在: {args.dataset}')
        sys.exit(1)

    print('=' * 80)
    print('YOLOv6 RKNN 转换工具')
    print('=' * 80)
    print(f'  ONNX模型:    {args.onnx}')
    print(f'  输出文件:    {args.output}')
    print(f'  Batch Size:  {args.batch_size}')
    print(f'  校验数据集:  {args.dataset}')
    print(f'  目标平台:    {args.platform}')
    print(f'  量化算法:    {args.quantize_algorithm}')
    print(f'  优化级别:    {args.optimization_level}')
    print('=' * 80)

    # 解析ONNX模型输出信息
    output_info = get_output_info(args.onnx, args.img_size, args.batch_size)

    # 输入预处理参数 (YOLOv6标准: RGB, [0,255] → [0,1])
    MEAN_VALUES = [[0, 0, 0]]
    STD_VALUES = [[255, 255, 255]]

    # 初始化RKNN
    rknn = RKNN(verbose=True)

    # ==================== 1. Config ====================
    print('\n[1/4] 配置RKNN...')
    ret = rknn.config(
        mean_values=MEAN_VALUES,
        std_values=STD_VALUES,
        target_platform=args.platform,
        quantized_algorithm=args.quantize_algorithm,
        quantized_dtype='asymmetric_quantized-8',  # INT8非对称量化
        optimization_level=args.optimization_level,
    )
    if ret != 0:
        print(f'✗ config失败: {ret}')
        sys.exit(1)
    print('✓ 配置完成')

    # ==================== 2. Load ONNX ====================
    print('\n[2/4] 加载ONNX模型...')
    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        print(f'✗ load_onnx失败: {ret}')
        sys.exit(1)
    print('✓ ONNX加载完成')

    # ==================== 3. Build (INT8 Quantization) ====================
    print('\n[3/4] 量化构建 (INT8 PTQ)...')
    print(f'  使用校验数据集进行量化校准')
    print(f'  这将需要几分钟时间...')

    ret = rknn.build(
        do_quantization=True,
        dataset=args.dataset,
        rknn_batch_size=args.batch_size,
    )
    if ret != 0:
        print(f'✗ build失败: {ret}')
        sys.exit(1)
    print('✓ 量化构建完成')

    # ==================== 4. Export RKNN ====================
    print('\n[4/4] 导出RKNN模型...')
    ret = rknn.export_rknn(args.output)
    if ret != 0:
        print(f'✗ export_rknn失败: {ret}')
        sys.exit(1)

    rknn.release()

    # ==================== 完成 ====================
    file_size = os.path.getsize(args.output) / (1024 * 1024)

    print('\n' + '=' * 80)
    print('✓ 转换完成!')
    print('=' * 80)
    print(f'  输出文件:    {args.output}')
    print(f'  文件大小:    {file_size:.2f} MB')
    print(f'  Batch Size:  {args.batch_size}')
    print(f'  输入形状:    [{args.batch_size}, 3, {args.img_size}, {args.img_size}] uint8 RGB')
    print(f'  输出数量:    {len(output_info) * 2} 个张量 ({len(output_info)}个尺度)')
    print()

    # 显示每个输出的详细信息
    for info in output_info:
        i = info['scale']
        stride = info['stride']
        grid_h = info['grid_h']
        grid_w = info['grid_w']
        print(f'  尺度{i} (stride={stride}):')
        print(f'    {info["reg_name"]}: [{args.batch_size}, 4, {grid_h}, {grid_w}] - bbox回归 (INT8)')
        print(f'    {info["cls_name"]}: [{args.batch_size}, 1, {grid_h}, {grid_w}] - 分类logits (INT8, 需CPU端sigmoid)')

    print('=' * 80)
    print('\n量化说明:')
    print('  ✓ Split输出模式: 分类logits为pre-sigmoid，避免0-1范围压缩')
    print('  ✓ CPU端后处理: sigmoid + anchor decode保持精度')
    print('  ✓ INT8量化: logits动态范围大，量化精度损失小')
    print('=' * 80)


if __name__ == '__main__':
    main()
