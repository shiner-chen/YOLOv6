"""ET-YOLOv6n new modules: StarBlock, C2fStar, CLAM, CrossLayerBifusion.

Paper: preprints202501.1348 — "ET-YOLOv6n: An Efficient and Tiny Object Detection Model"
"""
import torch
import torch.nn as nn
from yolov6.layers.common import ConvBNReLU, Transpose


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DWConvBN(nn.Module):
    """Depthwise Conv + BN (no activation)."""

    def __init__(self, channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(channels, channels, kernel_size, stride=stride,
                            padding=padding, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(self.dw(x))


class PWConvBN(nn.Module):
    """Pointwise Conv + BN (no activation)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pw = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.pw(x))


# ---------------------------------------------------------------------------
# StarBlock
# ---------------------------------------------------------------------------

class StarBlock(nn.Module):
    """Star (element-wise multiply) block from ET-YOLOv6n paper.

    Structure (for stride=1, in==out):
        Input
          └─ DWConv(3×3, BN)
               ├─ PWConv(BN)  ──────────────────────── ⊗ ── PWConv(BN) ── DWConv(3×3,BN) ─┐
               └─ PWConv(BN) ── ReLU6  ────────────────┘                                   + residual
                                                                                           Output
    When in_channels != out_channels the residual skip is dropped.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.use_res = (in_channels == out_channels)

        self.dw1 = DWConvBN(in_channels)
        self.pw_a = PWConvBN(in_channels, out_channels)          # linear branch
        self.pw_b = PWConvBN(in_channels, out_channels)          # activated branch
        self.act  = nn.ReLU6(inplace=True)
        self.pw_out = PWConvBN(out_channels, out_channels)
        self.dw2    = DWConvBN(out_channels)

    def forward(self, x):
        identity = x
        feat = self.dw1(x)
        a = self.pw_a(feat)
        b = self.act(self.pw_b(feat))
        star = a * b                          # Hadamard product — the "Star" op
        out = self.dw2(self.pw_out(star))
        if self.use_res:
            out = out + identity
        return out


# ---------------------------------------------------------------------------
# C2fStar — replaces RepBlock in the backbone
# ---------------------------------------------------------------------------

class C2fStar(nn.Module):
    """C2f with StarBlock inner repeats.

    Mimics the C2f pattern (split → repeated blocks → concat → project):
        Input ──────────────────────────────────────────────────────┐
          └─ split(half, half)                                       │
               ├─ branch_a: identity                                 │
               └─ branch_b: StarBlock × n, keep each output         │
                                                                     │
        Concat([branch_a, starout_0, ..., starout_n]) ──────────────┘
          └─ PWConv 1×1 (BN, no act) → out_channels
    """

    def __init__(self, in_channels, out_channels, n=1):
        super().__init__()
        assert in_channels % 2 == 0, "C2fStar requires even in_channels"
        half = in_channels // 2

        self.n = n
        self.blocks = nn.ModuleList(
            [StarBlock(half, half) for _ in range(n)]
        )
        # concat: branch_a (half) + n × half = half*(n+1)
        self.proj = PWConvBN(half * (n + 1), out_channels)

    def forward(self, x):
        a, b = x.chunk(2, dim=1)      # split along channel dim
        outs = [a]
        for blk in self.blocks:
            b = blk(b)
            outs.append(b)
        return self.proj(torch.cat(outs, dim=1))


# ---------------------------------------------------------------------------
# CrossLayerAttentionModule (CLAM)
# ---------------------------------------------------------------------------

class CLAM(nn.Module):
    """Channel attention using average + max pool, applied per feature map.

    Squeeze-and-Excitation style — no cross-layer tensor dependency,
    so any feature map can be enhanced independently before BiFusion.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        mx  = self.fc(self.max_pool(x))
        scale = self.sigmoid(avg + mx)
        return x * scale


# ---------------------------------------------------------------------------
# CrossLayerBifusion — replaces BiFusion in the neck
# ---------------------------------------------------------------------------

class CrossLayerBifusion(nn.Module):
    """3-input BiFusion augmented with per-branch CLAM channel attention.

    API is identical to BiFusion(in_channels=[skip1_ch, skip2_ch], out_channels):
        forward(x) where x = [feat_current, feat_skip1, feat_skip2]
          feat_current: feature from the layer above (will be upsampled)
          feat_skip1:   lateral connection (same or close scale)
          feat_skip2:   feature from two levels below (will be downsampled)

    Replaces BiFusion in RepBiFPANNeck without changing the caller's API.
    """

    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (list[int]): [skip1_channels, skip2_channels]
            out_channels (int): output channels (== feat_current channels after reduce_layer)
        """
        super().__init__()
        skip1_ch, skip2_ch = in_channels

        # Align skip branches to out_channels
        self.cv1 = ConvBNReLU(skip1_ch, out_channels, kernel_size=1, stride=1)
        self.cv2 = ConvBNReLU(skip2_ch, out_channels, kernel_size=1, stride=1)

        # Upsample current feature (already out_channels wide)
        self.upsample  = Transpose(out_channels, out_channels)
        # Downsample skip2 after alignment
        self.downsample = ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=2)

        # Per-branch channel attention
        self.attn_cur   = CLAM(out_channels)
        self.attn_skip1 = CLAM(out_channels)
        self.attn_skip2 = CLAM(out_channels)

        # Fuse 3 branches
        self.cv3 = ConvBNReLU(out_channels * 3, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        # x = [feat_current (already reduced), feat_skip1, feat_skip2]
        feat_cur, feat_skip1, feat_skip2 = x

        up   = self.attn_cur(self.upsample(feat_cur))
        s1   = self.attn_skip1(self.cv1(feat_skip1))
        s2   = self.attn_skip2(self.downsample(self.cv2(feat_skip2)))

        return self.cv3(torch.cat([up, s1, s2], dim=1))
