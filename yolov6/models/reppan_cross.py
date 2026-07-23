"""CrossTwoLevelBiFPANNeck — ET-YOLOv6n neck producing 4 output scales (P2–P5)."""
import torch
from torch import nn
from yolov6.layers.common import RepVGGBlock, RepBlock, ConvBNReLU, Transpose
from yolov6.models.et_modules import CrossLayerBifusion


class CrossTwoLevelBiFPANNeck(nn.Module):
    """BiFPN neck with CrossLayerBifusion and an extra P2 output branch.

    Channel layout (indices into the shared channels_list, width=0.25 defaults):
        Backbone inputs:
            ch[1] = 32   (P2, /4)
            ch[2] = 64   (P3, /8)
            ch[3] = 128  (P4, /16)
            ch[4] = 256  (P5, /32)
        Neck internals / outputs:
            ch[5]  = 64  (P5 reduced; also P5-out anchor)
            ch[6]  = 32  (P4 reduced; P3-out)
            ch[7]  = 32  (P3→P4 downsample)
            ch[8]  = 64  (P4-out)
            ch[9]  = 64  (P4-out→P5 downsample)
            ch[10] = 128 (P5-out)
            ch[11] = 32  (P3 upsample for P2 branch)
            ch[12] = 32  (P2-out)

    head chx: [12, 6, 8, 10] → P2, P3, P4, P5 outputs
    """

    def __init__(self, channels_list=None, num_repeats=None, block=RepVGGBlock):
        super().__init__()
        assert channels_list is not None
        assert num_repeats is not None

        # ------------------------------------------------------------------
        # Top-down FPN
        # ------------------------------------------------------------------

        # P5 → P4 level
        self.reduce_layer0 = ConvBNReLU(channels_list[4], channels_list[5], 1, 1)
        self.Bifusion0 = CrossLayerBifusion(
            in_channels=[channels_list[3], channels_list[2]],   # P4, P3
            out_channels=channels_list[5],
        )
        self.Rep_p4 = RepBlock(channels_list[5], channels_list[5],
                               n=num_repeats[5], block=block)

        # P4 → P3 level
        self.reduce_layer1 = ConvBNReLU(channels_list[5], channels_list[6], 1, 1)
        self.Bifusion1 = CrossLayerBifusion(
            in_channels=[channels_list[2], channels_list[1]],   # P3, P2
            out_channels=channels_list[6],
        )
        self.Rep_p3 = RepBlock(channels_list[6], channels_list[6],
                               n=num_repeats[6], block=block)

        # P3 → P2 (extra top-down branch for small objects)
        self.upsample2   = Transpose(channels_list[6], channels_list[11])
        self.Rep_p2 = RepBlock(
            in_channels=channels_list[11] + channels_list[1],  # upsample + P2 backbone
            out_channels=channels_list[12],
            n=num_repeats[9],
            block=block,
        )

        # ------------------------------------------------------------------
        # Bottom-up PAN
        # ------------------------------------------------------------------

        # P3 → P4
        self.downsample2 = ConvBNReLU(channels_list[6], channels_list[7], 3, 2)
        self.Rep_n3 = RepBlock(
            in_channels=channels_list[6] + channels_list[7],   # pan_out_p3 + fpn_out1
            out_channels=channels_list[8],
            n=num_repeats[7],
            block=block,
        )

        # P4 → P5
        self.downsample1 = ConvBNReLU(channels_list[8], channels_list[9], 3, 2)
        self.Rep_n4 = RepBlock(
            in_channels=channels_list[5] + channels_list[9],   # fpn_out0 + down
            out_channels=channels_list[10],
            n=num_repeats[8],
            block=block,
        )

    def forward(self, input):
        # backbone tuple: (P2, P3, P4, P5) — same order as RepBiFPANNeck
        (x3, x2, x1, x0) = input   # x3=P2:32, x2=P3:64, x1=P4:128, x0=P5:256

        # ---- top-down ----
        fpn_out0        = self.reduce_layer0(x0)
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
        f_out0          = self.Rep_p4(f_concat_layer0)           # P4-level feat

        fpn_out1        = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])
        pan_out_p3      = self.Rep_p3(f_concat_layer1)           # P3 out (/8)

        # extra P2 branch
        up_p3           = self.upsample2(pan_out_p3)
        pan_out_p2      = self.Rep_p2(torch.cat([up_p3, x3], 1)) # P2 out (/4)

        # ---- bottom-up ----
        down1           = self.downsample2(pan_out_p3)
        pan_out_p4      = self.Rep_n3(torch.cat([down1, fpn_out1], 1))  # P4 out (/16)

        down0           = self.downsample1(pan_out_p4)
        pan_out_p5      = self.Rep_n4(torch.cat([down0, fpn_out0], 1))  # P5 out (/32)

        # return finest-first: [P2, P3, P4, P5]
        return [pan_out_p2, pan_out_p3, pan_out_p4, pan_out_p5]
