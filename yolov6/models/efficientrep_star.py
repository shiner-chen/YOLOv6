"""EfficientRepStar backbone — identical to EfficientRep but RepBlock → C2fStar."""
from torch import nn
from yolov6.layers.common import RepVGGBlock, SimCSPSPPF, ConvBNSiLU, SPPF, SimSPPF, CSPSPPF
from yolov6.models.et_modules import C2fStar


class EfficientRepStar(nn.Module):
    """EfficientRep backbone with C2fStar replacing RepBlock.

    Drop-in replacement for EfficientRep. Accepts the same constructor
    arguments so build_network() in yolo.py can instantiate it directly.
    Always returns 4 feature maps (P2..P5) because fuse_P2 is implicitly True.
    """

    def __init__(
        self,
        in_channels=3,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock,
        fuse_P2=True,       # ET-YOLOv6n always outputs P2
        cspsppf=True,
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        # Stem: stride 2 → /2
        self.stem = block(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2,
        )

        # /4  — P2
        self.ERBlock_2 = nn.Sequential(
            block(channels_list[0], channels_list[1], kernel_size=3, stride=2),
            C2fStar(channels_list[1], channels_list[1], n=num_repeats[1]),
        )

        # /8  — P3
        self.ERBlock_3 = nn.Sequential(
            block(channels_list[1], channels_list[2], kernel_size=3, stride=2),
            C2fStar(channels_list[2], channels_list[2], n=num_repeats[2]),
        )

        # /16 — P4
        self.ERBlock_4 = nn.Sequential(
            block(channels_list[2], channels_list[3], kernel_size=3, stride=2),
            C2fStar(channels_list[3], channels_list[3], n=num_repeats[3]),
        )

        # /32 — P5  (with SPPF)
        channel_merge = CSPSPPF if cspsppf else SimSPPF
        self.ERBlock_5 = nn.Sequential(
            block(channels_list[3], channels_list[4], kernel_size=3, stride=2),
            C2fStar(channels_list[4], channels_list[4], n=num_repeats[4]),
            channel_merge(channels_list[4], channels_list[4], kernel_size=5),
        )

    def forward(self, x):
        x = self.stem(x)
        p2 = self.ERBlock_2(x)   # /4
        p3 = self.ERBlock_3(p2)  # /8
        p4 = self.ERBlock_4(p3)  # /16
        p5 = self.ERBlock_5(p4)  # /32
        return (p2, p3, p4, p5)
