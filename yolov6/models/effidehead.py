import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from yolov6.layers.common import *
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox


class Detect(nn.Module):
    export = False
    '''Efficient Decoupled Head
    With hardware-aware degisn, the decoupled head is optimized with
    hybridchannels methods.
    '''
    def __init__(self, num_classes=80, num_layers=3, inplace=True, head_layers=None, use_dfl=True, reg_max=16, p2_head=False):  # detection layer
        super().__init__()
        assert head_layers is not None
        self.nc = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = num_layers  # number of detection layers
        self.grid = [torch.zeros(1)] * num_layers
        self.prior_prob = 1e-2
        self.inplace = inplace
        if num_layers == 2:
            stride = [4, 8]           # P2/P3 only (ROI160 2-head)
        elif num_layers == 3:
            stride = [8, 16, 32]
        elif p2_head:
            stride = [4, 8, 16, 32]   # ET-YOLOv6n P2/P3/P4/P5
        else:
            stride = [8, 16, 32, 64]  # P6 variant
        self.stride = torch.tensor(stride)
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0

        # Init decouple head
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        # Efficient decoupled head layers
        for i in range(num_layers):
            idx = i*5
            self.stems.append(head_layers[idx])
            self.cls_convs.append(head_layers[idx+1])
            self.reg_convs.append(head_layers[idx+2])
            self.cls_preds.append(head_layers[idx+3])
            self.reg_preds.append(head_layers[idx+4])

    def initialize_biases(self):

        for conv in self.cls_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        for conv in self.reg_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(1.0)
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.proj_conv.weight = nn.Parameter(self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(),
                                                   requires_grad=False)

    def forward(self, x):
        if self.training:
            cls_score_list = []
            reg_distri_list = []

            for i in range(self.nl):
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)

                cls_output = torch.sigmoid(cls_output)
                cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))
                reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))

            cls_score_list = torch.cat(cls_score_list, axis=1)
            reg_distri_list = torch.cat(reg_distri_list, axis=1)

            return x, cls_score_list, reg_distri_list
        else:
            cls_score_list = []
            reg_dist_list = []

            for i in range(self.nl):
                b, _, h, w = x[i].shape
                l = h * w
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)

                if self.use_dfl:
                    reg_output = reg_output.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
                    reg_output = self.proj_conv(F.softmax(reg_output, dim=1))

                cls_output = torch.sigmoid(cls_output)

                if self.export:
                    cls_score_list.append(cls_output)
                    reg_dist_list.append(reg_output)
                else:
                    cls_score_list.append(cls_output.reshape([b, self.nc, l]))
                    reg_dist_list.append(reg_output.reshape([b, 4, l]))

            if self.export:
                return tuple(torch.cat([cls, reg], 1) for cls, reg in zip(cls_score_list, reg_dist_list))

            cls_score_list = torch.cat(cls_score_list, axis=-1).permute(0, 2, 1)
            reg_dist_list = torch.cat(reg_dist_list, axis=-1).permute(0, 2, 1)


            anchor_points, stride_tensor = generate_anchors(
                x, self.stride, self.grid_cell_size, self.grid_cell_offset, device=x[0].device, is_eval=True, mode='af')

            pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format='xywh')
            pred_bboxes *= stride_tensor
            return torch.cat(
                [
                    pred_bboxes,
                    torch.ones((b, pred_bboxes.shape[1], 1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),
                    cls_score_list
                ],
                axis=-1)


def build_effidehead_layer(channels_list, num_anchors, num_classes, reg_max=16, num_layers=3, p2_head=False, in_channels=None):

    if in_channels is not None:
        # Use explicit in_channels when provided (for custom neck architectures)
        chx = None
    elif num_layers == 2:
        chx = [7, 8]          # YOLOv6n P2/P3 2-head: P2_out(ch[7]=128), P3_out(ch[8]=256)
    elif num_layers == 3:
        chx = [6, 8, 10]
    elif p2_head:
        chx = [12, 6, 8, 10]  # ET-YOLOv6n (CrossTwoLevelBiFPANNeck): P2(/4), P3(/8), P4(/16), P5(/32)
    else:
        chx = [8, 9, 10, 11]  # P6 variant

    head_layers = nn.Sequential()

    # Build head layers dynamically based on num_layers
    for i in range(num_layers):
        # Get input channel for this layer
        if in_channels is not None:
            in_ch = in_channels[i]
        else:
            in_ch = channels_list[chx[i]]

        # stem
        head_layers.add_module(f'stem{i}',
            ConvBNSiLU(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=1,
                stride=1
            )
        )
        # cls_conv
        head_layers.add_module(f'cls_conv{i}',
            ConvBNSiLU(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=3,
                stride=1
            )
        )
        # reg_conv
        head_layers.add_module(f'reg_conv{i}',
            ConvBNSiLU(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=3,
                stride=1
            )
        )
        # cls_pred
        head_layers.add_module(f'cls_pred{i}',
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=num_classes * num_anchors,
                kernel_size=1
            )
        )
        # reg_pred
        head_layers.add_module(f'reg_pred{i}',
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=4 * (reg_max + num_anchors),
                kernel_size=1
            )
        )

    if num_layers == 4:
        # Get input channel for 4th layer (P5)
        if in_channels is not None:
            in_ch_p5 = in_channels[3]
        else:
            in_ch_p5 = channels_list[chx[3]]

        head_layers.add_module('stem3',
            # stem3
            ConvBNSiLU(
                in_channels=in_ch_p5,
                out_channels=in_ch_p5,
                kernel_size=1,
                stride=1
            )
        )
        head_layers.add_module('cls_conv3',
            # cls_conv3
            ConvBNSiLU(
                in_channels=in_ch_p5,
                out_channels=in_ch_p5,
                kernel_size=3,
                stride=1
            )
        )
        head_layers.add_module('reg_conv3',
            # reg_conv3
            ConvBNSiLU(
                in_channels=in_ch_p5,
                out_channels=in_ch_p5,
                kernel_size=3,
                stride=1
            )
        )
        head_layers.add_module('cls_pred3',
            # cls_pred3
            nn.Conv2d(
                in_channels=in_ch_p5,
                out_channels=num_classes * num_anchors,
                kernel_size=1
            )
         )
        head_layers.add_module('reg_pred3',
            # reg_pred3
            nn.Conv2d(
                in_channels=in_ch_p5,
                out_channels=4 * (reg_max + num_anchors),
                kernel_size=1
            )
        )

    return head_layers
