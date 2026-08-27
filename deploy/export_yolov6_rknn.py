#!/usr/bin/env python3
"""
YOLOv6 统一ONNX导出脚本 - 支持任意检测头配置

功能特性：
- 自动检测模型检测头数量（P2-P3 / P2-P5 等）
- 自动RepVGG重参数化优化
- Split output模式（pre-sigmoid logits）
- 支持任意输入尺寸
- 单类别检测优化

Usage:
    # ROI160 (P2-P3, 2个检测头)
    python export_yolov6_rknn.py \
        --weights yolov6n_nwd_p2-3_roi160.pt \
        --img-size 160 \
        --output yolov6n_roi160_split.onnx

    # ROI320 (P2-P5, 4个检测头)
    python export_yolov6_rknn.py \
        --weights yolov6n+p2+nwd_roi320.pt \
        --img-size 320 \
        --output yolov6n_roi320_split.onnx
"""
import argparse
import sys
import torch
import torch.nn as nn
import onnx
from io import BytesIO
from pathlib import Path

# 项目路径配置
ROOT = Path('/home/chenx/workdir/et-yolov6n')
sys.path.insert(0, str(ROOT))

from yolov6.layers.common import RepVGGBlock, ConvModule, SiLU
from yolov6.models.effidehead import Detect as DetectStd
from yolov6.utils.checkpoint import load_checkpoint

try:
    from yolov6.models.heads.effidehead_o2o import Detect as DetectO2O
    _DETECT_TYPES = (DetectStd, DetectO2O)
except ImportError:
    _DETECT_TYPES = (DetectStd,)


class SplitHeadWrapper(nn.Module):
    """
    包装 Detect head，输出每个尺度的独立 pre-sigmoid logits

    自动适配任意数量的检测头（nl=2,3,4,5等）
    """
    def __init__(self, detect_head):
        super().__init__()
        self.detect = detect_head
        self.nc = detect_head.nc
        self.nl = detect_head.nl  # 检测头数量
        self.use_dfl = detect_head.use_dfl

    def forward(self, x):
        """
        Args:
            x: list of feature maps from backbone+neck
        Returns:
            tuple of (reg_s0, cls_s0, reg_s1, cls_s1, ...)
            - reg_sX: [B, 4, H, W] 回归输出（raw，未经 anchor decode）
            - cls_sX: [B, nc, H, W] 分类 logits（pre-sigmoid）
        """
        outputs = []

        for i in range(self.nl):
            feat = self.detect.stems[i](x[i])

            # 分类分支
            cls_feat = self.detect.cls_convs[i](feat)
            cls_output = self.detect.cls_preds[i](cls_feat)  # [B, nc, H, W] pre-sigmoid

            # 回归分支
            reg_feat = self.detect.reg_convs[i](feat)
            reg_output = self.detect.reg_preds[i](reg_feat)  # [B, 4*reg_max or 4, H, W]

            if self.use_dfl:
                # DFL 模式：需要 softmax + proj_conv
                b, _, h, w = reg_output.shape
                reg_output = reg_output.reshape([b, 4, self.detect.reg_max + 1, h, w])
                reg_output = reg_output.permute(0, 2, 1, 3, 4)  # [B, reg_max+1, 4, H, W]
                reg_output = self.detect.proj_conv(torch.softmax(reg_output, dim=1))  # [B, 1, 4, H, W]
                reg_output = reg_output.squeeze(1)  # [B, 4, H, W]
            # else: 直接回归模式，reg_output 已经是 [B, 4, H, W]

            outputs.extend([reg_output, cls_output])

        return tuple(outputs)


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv6 ONNX导出工具')
    parser.add_argument('--weights', type=str, required=True,
                        help='PyTorch权重文件路径')
    parser.add_argument('--img-size', type=int, required=True,
                        help='输入图像尺寸 (例如: 160, 320, 640)')
    parser.add_argument('--output', type=str, required=True,
                        help='输出ONNX文件路径')
    parser.add_argument('--opset', type=int, default=13,
                        help='ONNX opset版本 (默认: 13)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='运行设备 (默认: cpu)')
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)

    print('=' * 80)
    print('YOLOv6 ONNX 导出工具')
    print('=' * 80)
    print(f'  权重文件: {args.weights}')
    print(f'  输入尺寸: {args.img_size}x{args.img_size}')
    print(f'  输出路径: {args.output}')
    print(f'  ONNX版本: opset {args.opset}')
    print('=' * 80)

    # 加载模型
    print(f'\n[1/5] 加载模型...')
    if not Path(args.weights).exists():
        print(f'✗ 权重文件不存在: {args.weights}')
        sys.exit(1)

    model = load_checkpoint(args.weights, map_location=device, inplace=True, fuse=True)
    print('✓ 模型加载完成')

    # RepVGG 重参数化（关键优化）
    print(f'\n[2/5] RepVGG 重参数化...')
    repvgg_count = 0
    for m in model.modules():
        if isinstance(m, RepVGGBlock):
            m.switch_to_deploy()
            repvgg_count += 1
        elif isinstance(m, nn.Upsample) and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None

    print(f'✓ 融合了 {repvgg_count} 个 RepVGG 块')

    if repvgg_count == 0:
        print('⚠ 警告: 未找到RepVGG块，模型可能未使用RepVGG结构')

    # 替换 SiLU 为导出友好版本
    print(f'\n[3/5] 优化激活函数...')
    for m in model.modules():
        if isinstance(m, ConvModule) and hasattr(m, 'act') and isinstance(m.act, nn.SiLU):
            m.act = SiLU()
    print('✓ SiLU激活函数已优化')

    # 查找并包装 Detect head
    print(f'\n[4/5] 配置检测头...')
    detect_head = None
    for m in model.modules():
        if isinstance(m, _DETECT_TYPES):
            detect_head = m
            break

    if detect_head is None:
        print('✗ 未找到Detect检测头')
        sys.exit(1)

    # 显示检测头配置
    print(f'✓ 检测头配置:')
    print(f'    类别数 (nc): {detect_head.nc}')
    print(f'    检测头数 (nl): {detect_head.nl}')
    print(f'    步长 (stride): {detect_head.stride.tolist()}')
    print(f'    DFL模式: {detect_head.use_dfl}')

    # 自动推断检测头类型
    if detect_head.nl == 2:
        head_type = 'P2+P3 (双尺度)'
    elif detect_head.nl == 3:
        head_type = 'P3+P4+P5 (三尺度)'
    elif detect_head.nl == 4:
        head_type = 'P2+P3+P4+P5 (四尺度)'
    elif detect_head.nl == 5:
        head_type = 'P2+P3+P4+P5+P6 (五尺度)'
    else:
        head_type = f'{detect_head.nl}个尺度'

    print(f'    检测头类型: {head_type}')

    # 包装模型：backbone+neck → split head
    class ExportModel(nn.Module):
        def __init__(self, base_model, detect_head):
            super().__init__()
            self.backbone = base_model.backbone
            self.neck = base_model.neck
            self.split_head = SplitHeadWrapper(detect_head)

        def forward(self, x):
            feat = self.backbone(x)
            feat = self.neck(feat)
            return self.split_head(feat)

    export_model = ExportModel(model, detect_head)
    export_model.eval()

    # 测试前向传播
    print(f'\n[5/5] 导出ONNX...')
    dummy = torch.zeros(1, 3, args.img_size, args.img_size, device=device)
    with torch.no_grad():
        outputs = export_model(dummy)

    # 构建输出名称
    output_names = []
    print(f'\n输出张量 (共 {len(outputs)} 个):')
    for i in range(len(outputs) // 2):
        reg_name = f'reg_s{i}'
        cls_name = f'cls_s{i}'
        output_names.extend([reg_name, cls_name])

        reg_shape = list(outputs[i*2].shape)
        cls_shape = list(outputs[i*2+1].shape)
        stride = detect_head.stride[i].item()

        print(f'  尺度 {i} (stride={stride}):')
        print(f'    {reg_name}: {reg_shape} - bbox回归')
        print(f'    {cls_name}: {cls_shape} - 分类logits (pre-sigmoid)')

    # 导出 ONNX
    print(f'\n正在导出到 {args.output}...')
    with BytesIO() as buf:
        torch.onnx.export(
            export_model, dummy, buf,
            verbose=False,
            opset_version=args.opset,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            input_names=['images'],
            output_names=output_names,
            dynamo=False,
        )
        buf.seek(0)
        onnx_model = onnx.load(buf)

    # 验证ONNX模型
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, args.output)

    file_size = Path(args.output).stat().st_size / 1e6

    # 输出总结
    print('\n' + '=' * 80)
    print('✓ 导出成功!')
    print('=' * 80)
    print(f'  输出文件: {args.output}')
    print(f'  文件大小: {file_size:.2f} MB')
    print(f'  输入形状: [1, 3, {args.img_size}, {args.img_size}]')
    print(f'  输出数量: {len(output_names)} 个张量')
    print(f'  检测头数: {detect_head.nl} 个尺度 ({head_type})')
    print(f'  RepVGG优化: ✓ ({repvgg_count} 个块已融合)')
    print(f'  Split模式: ✓ (pre-sigmoid logits)')
    print('=' * 80)


if __name__ == '__main__':
    main()
