# yolov6/models/heads/effidehead_o2o.py
"""Dual-branch O2M/O2O detection head with shared stem/cls_conv/reg_conv.

Training:  returns (feats, cls_o2m, reg_o2m, cls_o2o, reg_o2o).
Inference: only the O2O branch runs; returns (B, N_all, 4+1+nc).
"""
import math

import torch
import torch.nn as nn

from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.layers.common import ConvBNSiLU
from yolov6.utils.general import dist2bbox


class Detect(nn.Module):
    export = False

    def __init__(self, num_classes=80, num_layers=4, inplace=True,
                 head_layers=None, use_dfl=False, reg_max=0):
        super().__init__()
        assert head_layers is not None
        assert use_dfl is False and reg_max == 0, \
            "effidehead_o2o: DFL must be off (use_dfl=False, reg_max=0)"

        self.nc = num_classes
        self.no = num_classes + 5
        self.nl = num_layers
        self.prior_prob = 1e-2
        self.inplace = inplace

        # P2/P3/P4/P5 strides
        self.stride = torch.tensor([4, 8, 16, 32])
        self.grid = [torch.empty(0) for _ in range(num_layers)]
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0

        # 7 layers per scale:
        # [0] stem  [1] cls_conv  [2] reg_conv
        # [3] cls_pred_o2m  [4] reg_pred_o2m
        # [5] cls_pred_o2o  [6] reg_pred_o2o
        self.stems         = nn.ModuleList()
        self.cls_convs     = nn.ModuleList()
        self.reg_convs     = nn.ModuleList()
        self.cls_preds_o2m = nn.ModuleList()
        self.reg_preds_o2m = nn.ModuleList()
        self.cls_preds_o2o = nn.ModuleList()
        self.reg_preds_o2o = nn.ModuleList()

        for i in range(num_layers):
            idx = i * 7
            self.stems.append(head_layers[idx])
            self.cls_convs.append(head_layers[idx + 1])
            self.reg_convs.append(head_layers[idx + 2])
            self.cls_preds_o2m.append(head_layers[idx + 3])
            self.reg_preds_o2m.append(head_layers[idx + 4])
            self.cls_preds_o2o.append(head_layers[idx + 5])
            self.reg_preds_o2o.append(head_layers[idx + 6])

    def initialize_biases(self):
        """Initialize prediction head biases to prior_prob."""
        bias_val = -math.log((1 - self.prior_prob) / self.prior_prob)
        for conv in (*self.cls_preds_o2m, *self.cls_preds_o2o):
            conv.bias.data.fill_(bias_val)
            conv.weight.data.zero_()
        for conv in (*self.reg_preds_o2m, *self.reg_preds_o2o):
            conv.bias.data.fill_(1.0)
            conv.weight.data.zero_()

    def forward(self, x):
        """
        Args:
            x: list of feature maps [P2, P3, P4, P5]

        Training returns:
            (feats, cls_o2m, reg_o2m, cls_o2o, reg_o2o)
            shapes: cls (B, N_all, nc), reg (B, N_all, 4)

        Inference returns:
            Tensor (B, N_all, 4+1+nc) — O2O predictions only
        """
        if self.training:
            return self._forward_train(x)
        return self._forward_eval(x)

    def _forward_train(self, x):
        cls_o2m_list, reg_o2m_list = [], []
        cls_o2o_list, reg_o2o_list = [], []

        for i in range(self.nl):
            feat = self.stems[i](x[i])
            cls_feat = self.cls_convs[i](feat)
            reg_feat = self.reg_convs[i](feat)

            # O2M branch
            cls_o2m = torch.sigmoid(self.cls_preds_o2m[i](cls_feat))
            reg_o2m = self.reg_preds_o2m[i](reg_feat)
            cls_o2m_list.append(cls_o2m.flatten(2).permute(0, 2, 1))
            reg_o2m_list.append(reg_o2m.flatten(2).permute(0, 2, 1))

            # O2O branch
            cls_o2o = torch.sigmoid(self.cls_preds_o2o[i](cls_feat))
            reg_o2o = self.reg_preds_o2o[i](reg_feat)
            cls_o2o_list.append(cls_o2o.flatten(2).permute(0, 2, 1))
            reg_o2o_list.append(reg_o2o.flatten(2).permute(0, 2, 1))

        return (
            x,
            torch.cat(cls_o2m_list, dim=1),
            torch.cat(reg_o2m_list, dim=1),
            torch.cat(cls_o2o_list, dim=1),
            torch.cat(reg_o2o_list, dim=1),
        )

    def _forward_eval(self, x):
        cls_list, reg_list = [], []
        device = x[0].device

        for i in range(self.nl):
            b, _, h, w = x[i].shape
            feat = self.stems[i](x[i])
            cls_feat = self.cls_convs[i](feat)
            reg_feat = self.reg_convs[i](feat)

            # O2O branch only
            cls_out = torch.sigmoid(self.cls_preds_o2o[i](cls_feat))
            reg_out = self.reg_preds_o2o[i](reg_feat)

            cls_list.append(cls_out.reshape(b, self.nc, h * w))
            reg_list.append(reg_out.reshape(b, 4, h * w))

        cls_score = torch.cat(cls_list, dim=-1).permute(0, 2, 1)   # (B, N, nc)
        reg_dist  = torch.cat(reg_list, dim=-1).permute(0, 2, 1)   # (B, N, 4)

        anchor_points, stride_tensor = generate_anchors(
            x, self.stride, self.grid_cell_size, self.grid_cell_offset,
            device=device, is_eval=True, mode='af')

        pred_bboxes = dist2bbox(reg_dist, anchor_points, box_format='xywh')
        pred_bboxes = pred_bboxes * stride_tensor

        return torch.cat([
            pred_bboxes,
            torch.ones((pred_bboxes.shape[0], pred_bboxes.shape[1], 1),
                       device=device, dtype=pred_bboxes.dtype),
            cls_score,
        ], dim=-1)


def build_effidehead_layer(channels_list, num_classes, num_layers=4):
    """Build 7*num_layers modules for the O2O detection head.

    Layer order per scale (7 layers):
      [0] stem        1×1 ConvBNSiLU
      [1] cls_conv    3×3 ConvBNSiLU
      [2] reg_conv    3×3 ConvBNSiLU
      [3] cls_pred_o2m  1×1 Conv2d → num_classes
      [4] reg_pred_o2m  1×1 Conv2d → 4
      [5] cls_pred_o2o  1×1 Conv2d → num_classes
      [6] reg_pred_o2o  1×1 Conv2d → 4

    Args:
        channels_list: full channel list from backbone/neck (min length 13 for 4-scale).
        num_classes: number of foreground classes.
        num_layers: number of detection scales (default 4 for P2+P3+P4+P5).

    Returns:
        nn.Sequential of 7*num_layers modules.
    """
    # chx[i] selects the channel dim from channels_list for scale i
    # P2(idx=12), P3(idx=6), P4(idx=8), P5(idx=10) — matches et_yolov6n config
    chx = [12, 6, 8, 10]

    layers = []
    for i in range(num_layers):
        ch = channels_list[chx[i]]
        layers.extend([
            ConvBNSiLU(ch, ch, kernel_size=1, stride=1),   # stem
            ConvBNSiLU(ch, ch, kernel_size=3, stride=1),   # cls_conv
            ConvBNSiLU(ch, ch, kernel_size=3, stride=1),   # reg_conv
            nn.Conv2d(ch, num_classes, kernel_size=1),      # cls_pred_o2m
            nn.Conv2d(ch, 4,           kernel_size=1),      # reg_pred_o2m
            nn.Conv2d(ch, num_classes, kernel_size=1),      # cls_pred_o2o
            nn.Conv2d(ch, 4,           kernel_size=1),      # reg_pred_o2o
        ])

    return nn.Sequential(*layers)
