#!/usr/bin/env python3
"""
YOLOv6 统一RKNN性能测试脚本 - 支持任意检测头配置

功能特性：
- 自动检测模型的检测头数量和配置
- 支持任意batch size
- 单目标优化后处理（argmax选择）
- 完整的性能统计（推理+后处理）

Usage:
    # ROI160 batch=4
    python benchmark_yolov6_rknn.py \
        --model yolov6n_roi160_bs4_int8.rknn \
        --img-size 160 \
        --batch-size 4

    # ROI320 batch=1
    python benchmark_yolov6_rknn.py \
        --model yolov6n_roi320_bs1_int8.rknn \
        --img-size 320 \
        --batch-size 1

    # 多个模型对比
    python benchmark_yolov6_rknn.py \
        --model yolov6n_roi320_bs1_int8.rknn yolov6n_roi320_bs4_int8.rknn \
        --img-size 320 320 \
        --batch-size 1 4
"""
import argparse
import numpy as np
import time
from pathlib import Path
from rknnlite.api import RKNNLite


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv6 RKNN性能测试工具')
    parser.add_argument('--model', type=str, nargs='+', required=True,
                        help='RKNN模型路径（可指定多个）')
    parser.add_argument('--img-size', type=int, nargs='+', required=True,
                        help='输入图像尺寸（与模型对应）')
    parser.add_argument('--batch-size', type=int, nargs='+', required=True,
                        help='Batch size（与模型对应）')
    parser.add_argument('--warmup', type=int, default=10,
                        help='预热次数 (默认: 10)')
    parser.add_argument('--test-runs', type=int, default=100,
                        help='测试次数 (默认: 100)')
    parser.add_argument('--core-mask', type=str, default='0_1_2',
                        choices=['0', '1', '2', '0_1', '0_1_2'],
                        help='NPU核心掩码 (默认: 0_1_2=三核)')
    parser.add_argument('--conf-thresh', type=float, default=0.25,
                        help='置信度阈值 (默认: 0.25)')
    return parser.parse_args()


def get_core_mask(core_str):
    """转换core mask字符串为RKNNLite常量"""
    core_map = {
        '0': RKNNLite.NPU_CORE_0,
        '1': RKNNLite.NPU_CORE_1,
        '2': RKNNLite.NPU_CORE_2,
        '0_1': RKNNLite.NPU_CORE_0_1,
        '0_1_2': RKNNLite.NPU_CORE_0_1_2,
    }
    return core_map[core_str]


def sigmoid(x):
    """Sigmoid激活函数"""
    return 1.0 / (1.0 + np.exp(-x))


def generate_anchors(stride, grid_h, grid_w):
    """生成anchor points"""
    shifts_x = np.arange(0, grid_w) * stride
    shifts_y = np.arange(0, grid_h) * stride
    shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
    anchor_points = np.stack([shift_x.ravel(), shift_y.ravel()], axis=1) + stride // 2
    return anchor_points.astype(np.float32)


def dist2bbox_corrected(distance, anchor_points, stride):
    """
    距离解码为bbox（关键：distance必须乘以stride）

    Args:
        distance: [N, 4] 数组，[left, top, right, bottom]
        anchor_points: [N, 2] anchor中心点坐标
        stride: 当前尺度的步长

    Returns:
        bbox: [N, 4] 数组，[cx, cy, w, h]
    """
    lt = distance[:, :2] * stride
    rb = distance[:, 2:] * stride

    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb

    cx = (x1y1[:, 0] + x2y2[:, 0]) / 2
    cy = (x1y1[:, 1] + x2y2[:, 1]) / 2
    w = x2y2[:, 0] - x1y1[:, 0]
    h = x2y2[:, 1] - x1y1[:, 1]

    return np.stack([cx, cy, w, h], axis=1)


def postprocess_single_target(outputs, img_size, conf_thresh=0.25):
    """
    单目标后处理：直接选择置信度最高的检测

    自动检测输出数量并适配任意检测头配置

    Args:
        outputs: RKNN输出列表
        img_size: 输入图像尺寸
        conf_thresh: 置信度阈值

    Returns:
        best_box: [cx, cy, w, h, conf] 或 None
    """
    num_scales = len(outputs) // 2

    # 自动推断每个尺度的stride
    strides = []
    for i in range(num_scales):
        cls_output = outputs[i*2 + 1]
        grid_h = cls_output.shape[2]
        stride = img_size // grid_h
        strides.append(stride)

    all_boxes = []
    all_scores = []

    # 遍历所有尺度
    for i in range(num_scales):
        reg_output = outputs[i*2]      # reg_sX
        cls_output = outputs[i*2 + 1]  # cls_sX

        stride = strides[i]
        _, _, grid_h, grid_w = cls_output.shape

        # 反量化 + sigmoid
        cls_scores = sigmoid(cls_output.astype(np.float32))
        reg_dists = reg_output.astype(np.float32)

        # reshape
        scores = cls_scores[0, 0, :, :].reshape(-1)
        distances = reg_dists[0].transpose(1, 2, 0).reshape(-1, 4)

        # 生成anchor points
        anchor_points = generate_anchors(stride, grid_h, grid_w)

        # 解码bbox
        boxes = dist2bbox_corrected(distances, anchor_points, stride)

        all_boxes.append(boxes)
        all_scores.append(scores)

    # 合并所有尺度
    all_boxes = np.concatenate(all_boxes, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)

    # 找到最高置信度
    max_idx = all_scores.argmax()
    max_score = all_scores[max_idx]

    if max_score < conf_thresh:
        return None

    best_box = all_boxes[max_idx]
    return np.append(best_box, max_score)


def benchmark_model(model_path, img_size, batch_size, args):
    """性能测试单个模型"""
    print(f'\n{"="*80}')
    print(f'测试模型: {model_path}')
    print(f'{"="*80}')
    print(f'  输入尺寸:    {img_size}x{img_size}')
    print(f'  Batch Size:  {batch_size}')
    print(f'  NPU核心:     {args.core_mask}')
    print(f'{"="*80}')

    # 验证文件存在
    if not Path(model_path).exists():
        print(f'✗ 模型文件不存在: {model_path}')
        return None

    # 初始化RKNN
    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print(f'✗ 加载模型失败: {ret}')
        return None

    core_mask = get_core_mask(args.core_mask)
    ret = rknn.init_runtime(core_mask=core_mask)
    if ret != 0:
        print(f'✗ 初始化runtime失败: {ret}')
        rknn.release()
        return None

    print('✓ 模型加载成功')

    # 自动检测输出数量
    dummy_input = [np.random.randint(0, 256, (batch_size, img_size, img_size, 3), dtype=np.uint8)]
    outputs = rknn.inference(inputs=dummy_input)
    num_scales = len(outputs) // 2

    print(f'\n模型配置:')
    print(f'  检测尺度数: {num_scales}')
    for i in range(num_scales):
        reg_shape = list(outputs[i*2].shape)
        cls_shape = list(outputs[i*2+1].shape)
        stride = img_size // cls_shape[2]
        print(f'  尺度{i} (stride={stride}): reg={reg_shape}, cls={cls_shape}')

    # Warmup
    print(f'\n预热 {args.warmup} 次...')
    for _ in range(args.warmup):
        rknn.inference(inputs=dummy_input)

    # 性能测试 - NPU推理
    print(f'推理 {args.test_runs} 次...')
    times = []
    for _ in range(args.test_runs):
        start = time.perf_counter()
        outputs = rknn.inference(inputs=dummy_input)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    times = np.array(times)
    avg_time = times.mean()
    std_time = times.std()
    min_time = times.min()
    max_time = times.max()
    throughput = 1000.0 / avg_time * batch_size

    print(f'\n{"="*80}')
    print(f'NPU推理性能 (batch_size={batch_size}):')
    print(f'{"="*80}')
    print(f'  平均延迟:  {avg_time:.2f} ms')
    print(f'  标准差:    {std_time:.2f} ms')
    print(f'  最小延迟:  {min_time:.2f} ms')
    print(f'  最大延迟:  {max_time:.2f} ms')
    print(f'  吞吐量:    {throughput:.1f} FPS')
    if batch_size > 1:
        print(f'  单图延迟:  {avg_time/batch_size:.2f} ms/image')
    print(f'{"="*80}')

    # 测试后处理性能 (仅batch=1)
    postprocess_time = None
    if batch_size == 1:
        print(f'\n测试后处理性能 (单目标优化)...')
        outputs = rknn.inference(inputs=dummy_input)

        postprocess_times = []
        for _ in range(args.test_runs):
            start = time.perf_counter()
            result = postprocess_single_target(outputs, img_size=img_size, conf_thresh=args.conf_thresh)
            end = time.perf_counter()
            postprocess_times.append((end - start) * 1000)

        postprocess_times = np.array(postprocess_times)
        postprocess_time = postprocess_times.mean()

        end_to_end = avg_time + postprocess_time

        print(f'  后处理延迟:  {postprocess_time:.2f} ms')
        print(f'  端到端延迟:  {end_to_end:.2f} ms ({1000/end_to_end:.1f} FPS)')
        print(f'{"="*80}')

    rknn.release()

    return {
        'model': model_path,
        'img_size': img_size,
        'batch_size': batch_size,
        'num_scales': num_scales,
        'avg_time': avg_time,
        'std_time': std_time,
        'throughput': throughput,
        'postprocess_time': postprocess_time,
    }


def main():
    args = parse_args()

    # 验证参数数量匹配
    if len(args.model) != len(args.img_size) or len(args.model) != len(args.batch_size):
        print('✗ --model, --img-size, --batch-size 的数量必须相同')
        return

    print('YOLOv6 RKNN 性能测试工具')
    print(f'预热次数: {args.warmup}')
    print(f'测试次数: {args.test_runs}')
    print(f'NPU核心: {args.core_mask}')

    results = []

    # 测试所有模型
    for model, img_size, batch_size in zip(args.model, args.img_size, args.batch_size):
        result = benchmark_model(model, img_size, batch_size, args)
        if result:
            results.append(result)

    # 对比总结
    if len(results) > 1:
        print(f'\n{"="*80}')
        print('性能对比总结')
        print(f'{"="*80}')
        for i, r in enumerate(results):
            model_name = Path(r['model']).name
            print(f'\n模型 {i+1}: {model_name}')
            print(f'  输入尺寸:     {r["img_size"]}x{r["img_size"]}')
            print(f'  检测尺度:     {r["num_scales"]}个')
            print(f'  Batch Size:   {r["batch_size"]}')
            print(f'  NPU推理:      {r["avg_time"]:.2f} ms')
            print(f'  吞吐量:       {r["throughput"]:.1f} FPS')
            if r['postprocess_time']:
                print(f'  后处理:       {r["postprocess_time"]:.2f} ms')
                print(f'  端到端:       {r["avg_time"] + r["postprocess_time"]:.2f} ms')
        print(f'{"="*80}')
    elif len(results) == 1:
        r = results[0]
        print(f'\n{"="*80}')
        print('测试完成')
        print(f'{"="*80}')
        print(f'  模型:         {Path(r["model"]).name}')
        print(f'  输入尺寸:     {r["img_size"]}x{r["img_size"]}')
        print(f'  检测尺度:     {r["num_scales"]}个')
        print(f'  Batch Size:   {r["batch_size"]}')
        print(f'  NPU推理:      {r["avg_time"]:.2f} ms ({r["throughput"]:.1f} FPS)')
        if r['postprocess_time']:
            print(f'  端到端:       {r["avg_time"] + r["postprocess_time"]:.2f} ms')
        print(f'{"="*80}')


if __name__ == '__main__':
    main()
