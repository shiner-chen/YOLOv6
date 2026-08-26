#!/usr/bin/env python3
"""
Export YOLOv6n ROI160 (P2-P3) → split ONNX with per-scale pre-sigmoid outputs.

ROI160 模型特点：
- 双尺度 P2/P3 (stride 4, 8)
- 输入尺寸 160x160
- 单类别检测 (nc=1)
- use_dfl=False (直接回归)

输出格式（4个输出，每个尺度2个）：
  reg_s0  [1, 4, 40, 40]   P2 bbox regression (stride 4)
  cls_s0  [1, 1, 40, 40]   P2 class logits (pre-sigmoid)
  reg_s1  [1, 4, 20, 20]   P3 bbox regression (stride 8)
  cls_s1  [1, 1, 20, 20]   P3 class logits (pre-sigmoid)

Sigmoid 和 anchor decode 在 CPU 端完成，避免 INT8 量化对置信度的精度损失。

Usage:
    cd /home/chenx/workdir/et-yolov6n
    /home/chenx/rknn-env/bin/python3 /data/workdir/et-yolov6/export_roi160_split_onnx.py
"""
import argparse
import sys
import torch
import torch.nn as nn
import onnx
from io import BytesIO
from pathlib import Path

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
    """包装 Detect head，输出每个尺度的独立 pre-sigmoid logits"""
    def __init__(self, detect_head):
        super().__init__()
        self.detect = detect_head
        self.nc = detect_head.nc
        self.nl = detect_head.nl
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


def main():
    weights = '/data/workdir/et-yolov6/yolov6n_nwd_p2-3_roi160.pt'
    img_size = 160
    out_path = '/data/workdir/et-yolov6/yolov6n_roi160_p2p3_split.onnx'
    device = torch.device('cpu')

    print(f'Loading model from {weights}')
    model = load_checkpoint(weights, map_location=device, inplace=True, fuse=True)

    # 1. RepVGG 重参数化（加速推理）
    print('Fusing RepVGG blocks...')
    for m in model.modules():
        if isinstance(m, RepVGGBlock):
            m.switch_to_deploy()
        elif isinstance(m, nn.Upsample) and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None

    # 2. 替换 SiLU 为导出友好版本
    for m in model.modules():
        if isinstance(m, ConvModule) and hasattr(m, 'act') and isinstance(m.act, nn.SiLU):
            m.act = SiLU()

    # 3. 找到 Detect head 并用 SplitHeadWrapper 包装
    detect_head = None
    for m in model.modules():
        if isinstance(m, _DETECT_TYPES):
            detect_head = m
            print(f'Found Detect head: nc={m.nc}, nl={m.nl}, stride={m.stride.tolist()}, use_dfl={m.use_dfl}')
            break

    if detect_head is None:
        raise RuntimeError('No Detect head found in model')

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

    # 4. 测试前向传播
    dummy = torch.zeros(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        outputs = export_model(dummy)

    output_names = []
    print('\n输出张量:')
    for i in range(len(outputs) // 2):
        reg_name = f'reg_s{i}'
        cls_name = f'cls_s{i}'
        output_names.extend([reg_name, cls_name])
        print(f'  {reg_name}: {list(outputs[i*2].shape)} (bbox regression)')
        print(f'  {cls_name}: {list(outputs[i*2+1].shape)} (class logits, pre-sigmoid)')

    # 5. 导出 ONNX
    print(f'\nExporting to {out_path}...')
    with BytesIO() as buf:
        torch.onnx.export(
            export_model, dummy, buf,
            verbose=False,
            opset_version=13,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            input_names=['images'],
            output_names=output_names,
            dynamo=False,
        )
        buf.seek(0)
        onnx_model = onnx.load(buf)

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, out_path)
    print(f'\n✓ 导出成功: {out_path}  ({Path(out_path).stat().st_size/1e6:.2f} MB)')
    print(f'  输入: images [1, 3, {img_size}, {img_size}]')
    print(f'  输出: {len(output_names)} 个张量 (每个尺度 2 个: reg + cls)')


if __name__ == '__main__':
    main()
