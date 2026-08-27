import torch
from torch import nn
from yolov6.layers.common import RepBlock, RepVGGBlock, BottleRep, BepC3, ConvBNReLU, Transpose, BiFusion, \
                                MBLABlock, ConvBNHS, CSPBlock, DPBlock

# _QUANT=False
class RepPANNeck(nn.Module):
    """RepPANNeck Module
    EfficientRep is the default backbone of this model.
    RepPANNeck has the balance of feature fusion ability and hardware efficiency.
    """

    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        self.Rep_p4 = RepBlock(
            in_channels=channels_list[3] + channels_list[5],
            out_channels=channels_list[5],
            n=num_repeats[5],
            block=block
        )

        self.Rep_p3 = RepBlock(
            in_channels=channels_list[2] + channels_list[6],
            out_channels=channels_list[6],
            n=num_repeats[6],
            block=block
        )

        self.Rep_n3 = RepBlock(
            in_channels=channels_list[6] + channels_list[7],
            out_channels=channels_list[8],
            n=num_repeats[7],
            block=block
        )

        self.Rep_n4 = RepBlock(
            in_channels=channels_list[5] + channels_list[9],
            out_channels=channels_list[10],
            n=num_repeats[8],
            block=block
        )

        self.reduce_layer0 = ConvBNReLU(
            in_channels=channels_list[4],
            out_channels=channels_list[5],
            kernel_size=1,
            stride=1
        )

        self.upsample0 = Transpose(
            in_channels=channels_list[5],
            out_channels=channels_list[5],
        )

        self.reduce_layer1 = ConvBNReLU(
            in_channels=channels_list[5],
            out_channels=channels_list[6],
            kernel_size=1,
            stride=1
        )

        self.upsample1 = Transpose(
            in_channels=channels_list[6],
            out_channels=channels_list[6]
        )

        self.downsample2 = ConvBNReLU(
            in_channels=channels_list[6],
            out_channels=channels_list[7],
            kernel_size=3,
            stride=2
        )

        self.downsample1 = ConvBNReLU(
            in_channels=channels_list[8],
            out_channels=channels_list[9],
            kernel_size=3,
            stride=2
        )

    def upsample_enable_quant(self, num_bits, calib_method):
        print("Insert fakequant after upsample")
        # Insert fakequant after upsample op to build TensorRT engine
        from pytorch_quantization import nn as quant_nn
        from pytorch_quantization.tensor_quant import QuantDescriptor
        conv2d_input_default_desc = QuantDescriptor(num_bits=num_bits, calib_method=calib_method)
        self.upsample_feat0_quant = quant_nn.TensorQuantizer(conv2d_input_default_desc)
        self.upsample_feat1_quant = quant_nn.TensorQuantizer(conv2d_input_default_desc)
        # global _QUANT
        self._QUANT = True

    def forward(self, input):

        (x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        upsample_feat0 = self.upsample0(fpn_out0)
        if hasattr(self, '_QUANT') and self._QUANT is True:
            upsample_feat0 = self.upsample_feat0_quant(upsample_feat0)
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        upsample_feat1 = self.upsample1(fpn_out1)
        if hasattr(self, '_QUANT') and self._QUANT is True:
            upsample_feat1 = self.upsample_feat1_quant(upsample_feat1)
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)
        pan_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        outputs = [pan_out2, pan_out1, pan_out0]

        return outputs


class RepBiFPANNeck(nn.Module):
    """RepBiFPANNeck Module
    """
    # [64, 128, 256, 512, 1024]
    # [256, 128, 128, 256, 256, 512]

    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        self.reduce_layer0 = ConvBNReLU(
            in_channels=channels_list[4], # 1024
            out_channels=channels_list[5], # 256
            kernel_size=1,
            stride=1
        )

        self.Bifusion0 = BiFusion(
            in_channels=[channels_list[3], channels_list[2]], # 512, 256
            out_channels=channels_list[5], # 256
        )
        self.Rep_p4 = RepBlock(
            in_channels=channels_list[5], # 256
            out_channels=channels_list[5], # 256
            n=num_repeats[5],
            block=block
        )

        self.reduce_layer1 = ConvBNReLU(
            in_channels=channels_list[5], # 256
            out_channels=channels_list[6], # 128
            kernel_size=1,
            stride=1
        )

        self.Bifusion1 = BiFusion(
            in_channels=[channels_list[2], channels_list[1]], # 256, 128
            out_channels=channels_list[6], # 128
        )

        self.Rep_p3 = RepBlock(
            in_channels=channels_list[6], # 128
            out_channels=channels_list[6], # 128
            n=num_repeats[6],
            block=block
        )

        self.downsample2 = ConvBNReLU(
            in_channels=channels_list[6], # 128
            out_channels=channels_list[7], # 128
            kernel_size=3,
            stride=2
        )

        self.Rep_n3 = RepBlock(
            in_channels=channels_list[6] + channels_list[7], # 128 + 128
            out_channels=channels_list[8], # 256
            n=num_repeats[7],
            block=block
        )

        self.downsample1 = ConvBNReLU(
            in_channels=channels_list[8], # 256
            out_channels=channels_list[9], # 256
            kernel_size=3,
            stride=2
        )

        self.Rep_n4 = RepBlock(
            in_channels=channels_list[5] + channels_list[9], # 256 + 256
            out_channels=channels_list[10], # 512
            n=num_repeats[8],
            block=block
        )


    def forward(self, input):

        (x3, x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])
        pan_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        outputs = [pan_out2, pan_out1, pan_out0]

        return outputs


class RepPANNeck6(nn.Module):
    """RepPANNeck+P6 Module
    EfficientRep is the default backbone of this model.
    RepPANNeck has the balance of feature fusion ability and hardware efficiency.
    """
    # [64, 128, 256, 512, 768, 1024]
    # [512, 256, 128, 256, 512, 1024]
    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        self.reduce_layer0 = ConvBNReLU(
            in_channels=channels_list[5], # 1024
            out_channels=channels_list[6], # 512
            kernel_size=1,
            stride=1
        )

        self.upsample0 = Transpose(
            in_channels=channels_list[6],  # 512
            out_channels=channels_list[6], # 512
        )

        self.Rep_p5 = RepBlock(
            in_channels=channels_list[4] + channels_list[6], # 768 + 512
            out_channels=channels_list[6], # 512
            n=num_repeats[6],
            block=block
        )

        self.reduce_layer1 = ConvBNReLU(
            in_channels=channels_list[6],  # 512
            out_channels=channels_list[7], # 256
            kernel_size=1,
            stride=1
        )

        self.upsample1 = Transpose(
            in_channels=channels_list[7], # 256
            out_channels=channels_list[7] # 256
        )

        self.Rep_p4 = RepBlock(
            in_channels=channels_list[3] + channels_list[7], # 512 + 256
            out_channels=channels_list[7], # 256
            n=num_repeats[7],
            block=block
        )

        self.reduce_layer2 = ConvBNReLU(
            in_channels=channels_list[7],  # 256
            out_channels=channels_list[8], # 128
            kernel_size=1,
            stride=1
        )

        self.upsample2 = Transpose(
            in_channels=channels_list[8], # 128
            out_channels=channels_list[8] # 128
        )

        self.Rep_p3 = RepBlock(
            in_channels=channels_list[2] + channels_list[8], # 256 + 128
            out_channels=channels_list[8], # 128
            n=num_repeats[8],
            block=block
        )

        self.downsample2 = ConvBNReLU(
            in_channels=channels_list[8],  # 128
            out_channels=channels_list[8], # 128
            kernel_size=3,
            stride=2
        )

        self.Rep_n4 = RepBlock(
            in_channels=channels_list[8] + channels_list[8], # 128 + 128
            out_channels=channels_list[9], # 256
            n=num_repeats[9],
            block=block
        )

        self.downsample1 = ConvBNReLU(
            in_channels=channels_list[9],  # 256
            out_channels=channels_list[9], # 256
            kernel_size=3,
            stride=2
        )

        self.Rep_n5 = RepBlock(
            in_channels=channels_list[7] + channels_list[9], # 256 + 256
            out_channels=channels_list[10], # 512
            n=num_repeats[10],
            block=block
        )

        self.downsample0 = ConvBNReLU(
            in_channels=channels_list[10],  # 512
            out_channels=channels_list[10], # 512
            kernel_size=3,
            stride=2
        )

        self.Rep_n6 = RepBlock(
            in_channels=channels_list[6] + channels_list[10], # 512 + 512
            out_channels=channels_list[11], # 1024
            n=num_repeats[11],
            block=block
        )


    def forward(self, input):

        (x3, x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)
        f_out0 = self.Rep_p5(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        upsample_feat1 = self.upsample1(fpn_out1)
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)
        f_out1 = self.Rep_p4(f_concat_layer1)

        fpn_out2 = self.reduce_layer2(f_out1)
        upsample_feat2 = self.upsample2(fpn_out2)
        f_concat_layer2 = torch.cat([upsample_feat2, x3], 1)
        pan_out3 = self.Rep_p3(f_concat_layer2) # P3

        down_feat2 = self.downsample2(pan_out3)
        p_concat_layer2 = torch.cat([down_feat2, fpn_out2], 1)
        pan_out2 = self.Rep_n4(p_concat_layer2) # P4

        down_feat1 = self.downsample1(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n5(p_concat_layer1) # P5

        down_feat0 = self.downsample0(pan_out1)
        p_concat_layer0 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n6(p_concat_layer0) # P6

        outputs = [pan_out3, pan_out2, pan_out1, pan_out0]

        return outputs


class RepBiFPANNeck6(nn.Module):
    """RepBiFPANNeck_P6 Module
    """
    # [64, 128, 256, 512, 768, 1024]
    # [512, 256, 128, 256, 512, 1024]

    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        self.reduce_layer0 = ConvBNReLU(
            in_channels=channels_list[5], # 1024
            out_channels=channels_list[6], # 512
            kernel_size=1,
            stride=1
        )

        self.Bifusion0 = BiFusion(
            in_channels=[channels_list[4], channels_list[6]], # 768, 512
            out_channels=channels_list[6], # 512
        )

        self.Rep_p5 = RepBlock(
            in_channels=channels_list[6], # 512
            out_channels=channels_list[6], # 512
            n=num_repeats[6],
            block=block
        )

        self.reduce_layer1 = ConvBNReLU(
            in_channels=channels_list[6],  # 512
            out_channels=channels_list[7], # 256
            kernel_size=1,
            stride=1
        )

        self.Bifusion1 = BiFusion(
            in_channels=[channels_list[3], channels_list[7]], # 512, 256
            out_channels=channels_list[7], # 256
        )

        self.Rep_p4 = RepBlock(
            in_channels=channels_list[7], # 256
            out_channels=channels_list[7], # 256
            n=num_repeats[7],
            block=block
        )

        self.reduce_layer2 = ConvBNReLU(
            in_channels=channels_list[7],  # 256
            out_channels=channels_list[8], # 128
            kernel_size=1,
            stride=1
        )

        self.Bifusion2 = BiFusion(
            in_channels=[channels_list[2], channels_list[8]], # 256, 128
            out_channels=channels_list[8], # 128
        )

        self.Rep_p3 = RepBlock(
            in_channels=channels_list[8], # 128
            out_channels=channels_list[8], # 128
            n=num_repeats[8],
            block=block
        )

        self.downsample2 = ConvBNReLU(
            in_channels=channels_list[8],  # 128
            out_channels=channels_list[8], # 128
            kernel_size=3,
            stride=2
        )

        self.Rep_n4 = RepBlock(
            in_channels=channels_list[8] + channels_list[8], # 128 + 128
            out_channels=channels_list[9], # 256
            n=num_repeats[9],
            block=block
        )

        self.downsample1 = ConvBNReLU(
            in_channels=channels_list[9],  # 256
            out_channels=channels_list[9], # 256
            kernel_size=3,
            stride=2
        )

        self.Rep_n5 = RepBlock(
            in_channels=channels_list[7] + channels_list[9], # 256 + 256
            out_channels=channels_list[10], # 512
            n=num_repeats[10],
            block=block
        )

        self.downsample0 = ConvBNReLU(
            in_channels=channels_list[10],  # 512
            out_channels=channels_list[10], # 512
            kernel_size=3,
            stride=2
        )

        self.Rep_n6 = RepBlock(
            in_channels=channels_list[6] + channels_list[10], # 512 + 512
            out_channels=channels_list[11], # 1024
            n=num_repeats[11],
            block=block
        )


    def forward(self, input):

        (x4, x3, x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
        f_out0 = self.Rep_p5(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])
        f_out1 = self.Rep_p4(f_concat_layer1)

        fpn_out2 = self.reduce_layer2(f_out1)
        f_concat_layer2 = self.Bifusion2([fpn_out2, x3, x4])
        pan_out3 = self.Rep_p3(f_concat_layer2) # P3

        down_feat2 = self.downsample2(pan_out3)
        p_concat_layer2 = torch.cat([down_feat2, fpn_out2], 1)
        pan_out2 = self.Rep_n4(p_concat_layer2) # P4

        down_feat1 = self.downsample1(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n5(p_concat_layer1) # P5

        down_feat0 = self.downsample0(pan_out1)
        p_concat_layer0 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n6(p_concat_layer0) # P6

        outputs = [pan_out3, pan_out2, pan_out1, pan_out0]

        return outputs


class CSPRepPANNeck(nn.Module):
    """
    CSPRepPANNeck module.
    """

    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        block=BottleRep,
        csp_e=float(1)/2,
        stage_block_type="BepC3"
    ):
        super().__init__()

        if stage_block_type == "BepC3":
            stage_block = BepC3
        elif stage_block_type == "MBLABlock":
            stage_block = MBLABlock
        else:
            raise NotImplementedError

        assert channels_list is not None
        assert num_repeats is not None

        self.Rep_p4 = stage_block(
            in_channels=channels_list[3] + channels_list[5], # 512 + 256
            out_channels=channels_list[5], # 256
            n=num_repeats[5],
            e=csp_e,
            block=block
        )

        self.Rep_p3 = stage_block(
            in_channels=channels_list[2] + channels_list[6], # 256 + 128
            out_channels=channels_list[6], # 128
            n=num_repeats[6],
            e=csp_e,
            block=block
        )

        self.Rep_n3 = stage_block(
            in_channels=channels_list[6] + channels_list[7], # 128 + 128
            out_channels=channels_list[8], # 256
            n=num_repeats[7],
            e=csp_e,
            block=block
        )

        self.Rep_n4 = stage_block(
            in_channels=channels_list[5] + channels_list[9], # 256 + 256
            out_channels=channels_list[10], # 512
            n=num_repeats[8],
            e=csp_e,
            block=block
        )

        self.reduce_layer0 = ConvBNReLU(
            in_channels=channels_list[4], # 1024
            out_channels=channels_list[5], # 256
            kernel_size=1,
            stride=1
        )

        self.upsample0 = Transpose(
            in_channels=channels_list[5], # 256
            out_channels=channels_list[5], # 256
        )

        self.reduce_layer1 = ConvBNReLU(
            in_channels=channels_list[5], # 256
            out_channels=channels_list[6], # 128
            kernel_size=1,
            stride=1
        )

        self.upsample1 = Transpose(
            in_channels=channels_list[6], # 128
            out_channels=channels_list[6] # 128
        )

        self.downsample2 = ConvBNReLU(
            in_channels=channels_list[6], # 128
            out_channels=channels_list[7], # 128
            kernel_size=3,
            stride=2
        )

        self.downsample1 = ConvBNReLU(
            in_channels=channels_list[8], # 256
            out_channels=channels_list[9], # 256
            kernel_size=3,
            stride=2
        )

    def forward(self, input):

        (x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        upsample_feat1 = self.upsample1(fpn_out1)
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)
        pan_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        outputs = [pan_out2, pan_out1, pan_out0]

        return outputs


class CSPRepBiFPANNeck(nn.Module):
    """
    CSPRepBiFPANNeck module.
    """

    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        block=BottleRep,
        csp_e=float(1)/2,
        stage_block_type="BepC3"
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        if stage_block_type == "BepC3":
            stage_block = BepC3
        elif stage_block_type == "MBLABlock":
            stage_block = MBLABlock
        else:
            raise NotImplementedError

        self.reduce_layer0 = ConvBNReLU(
            in_channels=channels_list[4], # 1024
            out_channels=channels_list[5], # 256
            kernel_size=1,
            stride=1
        )

        self.Bifusion0 = BiFusion(
            in_channels=[channels_list[3], channels_list[2]], # 512, 256
            out_channels=channels_list[5], # 256
        )

        self.Rep_p4 = stage_block(
            in_channels=channels_list[5], # 256
            out_channels=channels_list[5], # 256
            n=num_repeats[5],
            e=csp_e,
            block=block
        )

        self.reduce_layer1 = ConvBNReLU(
            in_channels=channels_list[5], # 256
            out_channels=channels_list[6], # 128
            kernel_size=1,
            stride=1
        )

        self.Bifusion1 = BiFusion(
            in_channels=[channels_list[2], channels_list[1]], # 256, 128
            out_channels=channels_list[6], # 128
        )

        self.Rep_p3 = stage_block(
            in_channels=channels_list[6], # 128
            out_channels=channels_list[6], # 128
            n=num_repeats[6],
            e=csp_e,
            block=block
        )

        self.downsample2 = ConvBNReLU(
            in_channels=channels_list[6], # 128
            out_channels=channels_list[7], # 128
            kernel_size=3,
            stride=2
        )

        self.Rep_n3 = stage_block(
            in_channels=channels_list[6] + channels_list[7], # 128 + 128
            out_channels=channels_list[8], # 256
            n=num_repeats[7],
            e=csp_e,
            block=block
        )

        self.downsample1 = ConvBNReLU(
            in_channels=channels_list[8], # 256
            out_channels=channels_list[9], # 256
            kernel_size=3,
            stride=2
        )


        self.Rep_n4 = stage_block(
            in_channels=channels_list[5] + channels_list[9], # 256 + 256
            out_channels=channels_list[10], # 512
            n=num_repeats[8],
            e=csp_e,
            block=block
        )


    def forward(self, input):

        (x3, x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])
        pan_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        outputs = [pan_out2, pan_out1, pan_out0]

        return outputs


class CSPRepPANNeck_P6(nn.Module):
    """CSPRepPANNeck_P6 Module
    """
    # [64, 128, 256, 512, 768, 1024]
    # [512, 256, 128, 256, 512, 1024]
    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        block=BottleRep,
        csp_e=float(1)/2,
        stage_block_type="BepC3"
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        if stage_block_type == "BepC3":
            stage_block = BepC3
        elif stage_block_type == "MBLABlock":
            stage_block = MBLABlock
        else:
            raise NotImplementedError

        self.reduce_layer0 = ConvBNReLU(
            in_channels=channels_list[5], # 1024
            out_channels=channels_list[6], # 512
            kernel_size=1,
            stride=1
        )

        self.upsample0 = Transpose(
            in_channels=channels_list[6],  # 512
            out_channels=channels_list[6], # 512
        )

        self.Rep_p5 = stage_block(
            in_channels=channels_list[4] + channels_list[6], # 768 + 512
            out_channels=channels_list[6], # 512
            n=num_repeats[6],
            e=csp_e,
            block=block
        )

        self.reduce_layer1 = ConvBNReLU(
            in_channels=channels_list[6],  # 512
            out_channels=channels_list[7], # 256
            kernel_size=1,
            stride=1
        )

        self.upsample1 = Transpose(
            in_channels=channels_list[7], # 256
            out_channels=channels_list[7] # 256
        )

        self.Rep_p4 = stage_block(
            in_channels=channels_list[3] + channels_list[7], # 512 + 256
            out_channels=channels_list[7], # 256
            n=num_repeats[7],
            e=csp_e,
            block=block
        )

        self.reduce_layer2 = ConvBNReLU(
            in_channels=channels_list[7],  # 256
            out_channels=channels_list[8], # 128
            kernel_size=1,
            stride=1
        )

        self.upsample2 = Transpose(
            in_channels=channels_list[8], # 128
            out_channels=channels_list[8] # 128
        )

        self.Rep_p3 = stage_block(
            in_channels=channels_list[2] + channels_list[8], # 256 + 128
            out_channels=channels_list[8], # 128
            n=num_repeats[8],
            e=csp_e,
            block=block
        )

        self.downsample2 = ConvBNReLU(
            in_channels=channels_list[8],  # 128
            out_channels=channels_list[8], # 128
            kernel_size=3,
            stride=2
        )

        self.Rep_n4 = stage_block(
            in_channels=channels_list[8] + channels_list[8], # 128 + 128
            out_channels=channels_list[9], # 256
            n=num_repeats[9],
            e=csp_e,
            block=block
        )

        self.downsample1 = ConvBNReLU(
            in_channels=channels_list[9],  # 256
            out_channels=channels_list[9], # 256
            kernel_size=3,
            stride=2
        )

        self.Rep_n5 = stage_block(
            in_channels=channels_list[7] + channels_list[9], # 256 + 256
            out_channels=channels_list[10], # 512
            n=num_repeats[10],
            e=csp_e,
            block=block
        )

        self.downsample0 = ConvBNReLU(
            in_channels=channels_list[10],  # 512
            out_channels=channels_list[10], # 512
            kernel_size=3,
            stride=2
        )

        self.Rep_n6 = stage_block(
            in_channels=channels_list[6] + channels_list[10], # 512 + 512
            out_channels=channels_list[11], # 1024
            n=num_repeats[11],
            e=csp_e,
            block=block
        )


    def forward(self, input):

        (x3, x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)
        f_out0 = self.Rep_p5(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        upsample_feat1 = self.upsample1(fpn_out1)
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)
        f_out1 = self.Rep_p4(f_concat_layer1)

        fpn_out2 = self.reduce_layer2(f_out1)
        upsample_feat2 = self.upsample2(fpn_out2)
        f_concat_layer2 = torch.cat([upsample_feat2, x3], 1)
        pan_out3 = self.Rep_p3(f_concat_layer2) # P3

        down_feat2 = self.downsample2(pan_out3)
        p_concat_layer2 = torch.cat([down_feat2, fpn_out2], 1)
        pan_out2 = self.Rep_n4(p_concat_layer2) # P4

        down_feat1 = self.downsample1(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n5(p_concat_layer1) # P5

        down_feat0 = self.downsample0(pan_out1)
        p_concat_layer0 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n6(p_concat_layer0) # P6

        outputs = [pan_out3, pan_out2, pan_out1, pan_out0]

        return outputs


class CSPRepBiFPANNeck_P6(nn.Module):
    """CSPRepBiFPANNeck_P6 Module
    """
    # [64, 128, 256, 512, 768, 1024]
    # [512, 256, 128, 256, 512, 1024]
    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        block=BottleRep,
        csp_e=float(1)/2,
        stage_block_type="BepC3"
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        if stage_block_type == "BepC3":
            stage_block = BepC3
        elif stage_block_type == "MBLABlock":
            stage_block = MBLABlock
        else:
            raise NotImplementedError

        self.reduce_layer0 = ConvBNReLU(
            in_channels=channels_list[5], # 1024
            out_channels=channels_list[6], # 512
            kernel_size=1,
            stride=1
        )

        self.Bifusion0 = BiFusion(
            in_channels=[channels_list[4], channels_list[6]], # 768, 512
            out_channels=channels_list[6], # 512
        )

        self.Rep_p5 = stage_block(
            in_channels=channels_list[6], # 512
            out_channels=channels_list[6], # 512
            n=num_repeats[6],
            e=csp_e,
            block=block
        )

        self.reduce_layer1 = ConvBNReLU(
            in_channels=channels_list[6],  # 512
            out_channels=channels_list[7], # 256
            kernel_size=1,
            stride=1
        )

        self.Bifusion1 = BiFusion(
            in_channels=[channels_list[3], channels_list[7]], # 512, 256
            out_channels=channels_list[7], # 256
        )

        self.Rep_p4 = stage_block(
            in_channels=channels_list[7], # 256
            out_channels=channels_list[7], # 256
            n=num_repeats[7],
            e=csp_e,
            block=block
        )

        self.reduce_layer2 = ConvBNReLU(
            in_channels=channels_list[7],  # 256
            out_channels=channels_list[8], # 128
            kernel_size=1,
            stride=1
        )

        self.Bifusion2 = BiFusion(
            in_channels=[channels_list[2], channels_list[8]], # 256, 128
            out_channels=channels_list[8], # 128
        )

        self.Rep_p3 = stage_block(
            in_channels=channels_list[8], # 128
            out_channels=channels_list[8], # 128
            n=num_repeats[8],
            e=csp_e,
            block=block
        )

        self.downsample2 = ConvBNReLU(
            in_channels=channels_list[8],  # 128
            out_channels=channels_list[8], # 128
            kernel_size=3,
            stride=2
        )

        self.Rep_n4 = stage_block(
            in_channels=channels_list[8] + channels_list[8], # 128 + 128
            out_channels=channels_list[9], # 256
            n=num_repeats[9],
            e=csp_e,
            block=block
        )

        self.downsample1 = ConvBNReLU(
            in_channels=channels_list[9],  # 256
            out_channels=channels_list[9], # 256
            kernel_size=3,
            stride=2
        )

        self.Rep_n5 = stage_block(
            in_channels=channels_list[7] + channels_list[9], # 256 + 256
            out_channels=channels_list[10], # 512
            n=num_repeats[10],
            e=csp_e,
            block=block
        )

        self.downsample0 = ConvBNReLU(
            in_channels=channels_list[10],  # 512
            out_channels=channels_list[10], # 512
            kernel_size=3,
            stride=2
        )

        self.Rep_n6 = stage_block(
            in_channels=channels_list[6] + channels_list[10], # 512 + 512
            out_channels=channels_list[11], # 1024
            n=num_repeats[11],
            e=csp_e,
            block=block
        )


    def forward(self, input):

        (x4, x3, x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
        f_out0 = self.Rep_p5(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])
        f_out1 = self.Rep_p4(f_concat_layer1)

        fpn_out2 = self.reduce_layer2(f_out1)
        f_concat_layer2 = self.Bifusion2([fpn_out2, x3, x4])
        pan_out3 = self.Rep_p3(f_concat_layer2) # P3

        down_feat2 = self.downsample2(pan_out3)
        p_concat_layer2 = torch.cat([down_feat2, fpn_out2], 1)
        pan_out2 = self.Rep_n4(p_concat_layer2) # P4

        down_feat1 = self.downsample1(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n5(p_concat_layer1) # P5

        down_feat0 = self.downsample0(pan_out1)
        p_concat_layer0 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n6(p_concat_layer0) # P6

        outputs = [pan_out3, pan_out2, pan_out1, pan_out0]

        return outputs

class Lite_EffiNeck(nn.Module):

    def __init__(
        self,
        in_channels,
        unified_channels,
    ):
        super().__init__()
        self.reduce_layer0 = ConvBNHS(
            in_channels=in_channels[0],
            out_channels=unified_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.reduce_layer1 = ConvBNHS(
            in_channels=in_channels[1],
            out_channels=unified_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.reduce_layer2 = ConvBNHS(
            in_channels=in_channels[2],
            out_channels=unified_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.upsample0 = nn.Upsample(scale_factor=2, mode='nearest')

        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.Csp_p4 = CSPBlock(
            in_channels=unified_channels*2,
            out_channels=unified_channels,
            kernel_size=5
        )
        self.Csp_p3 = CSPBlock(
            in_channels=unified_channels*2,
            out_channels=unified_channels,
            kernel_size=5
        )
        self.Csp_n3 = CSPBlock(
            in_channels=unified_channels*2,
            out_channels=unified_channels,
            kernel_size=5
        )
        self.Csp_n4 = CSPBlock(
            in_channels=unified_channels*2,
            out_channels=unified_channels,
            kernel_size=5
        )
        self.downsample2 = DPBlock(
            in_channel=unified_channels,
            out_channel=unified_channels,
            kernel_size=5,
            stride=2
        )
        self.downsample1 = DPBlock(
            in_channel=unified_channels,
            out_channel=unified_channels,
            kernel_size=5,
            stride=2
        )
        self.p6_conv_1 = DPBlock(
            in_channel=unified_channels,
            out_channel=unified_channels,
            kernel_size=5,
            stride=2
        )
        self.p6_conv_2 = DPBlock(
            in_channel=unified_channels,
            out_channel=unified_channels,
            kernel_size=5,
            stride=2
        )

    def forward(self, input):

        (x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0) #c5
        x1 = self.reduce_layer1(x1)       #c4
        x2 = self.reduce_layer2(x2)       #c3

        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)
        f_out1 = self.Csp_p4(f_concat_layer0)

        upsample_feat1 = self.upsample1(f_out1)
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)
        pan_out3 = self.Csp_p3(f_concat_layer1) #p3

        down_feat1 = self.downsample2(pan_out3)
        p_concat_layer1 = torch.cat([down_feat1, f_out1], 1)
        pan_out2 = self.Csp_n3(p_concat_layer1)  #p4

        down_feat0 = self.downsample1(pan_out2)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out1 = self.Csp_n4(p_concat_layer2)  #p5

        top_features = self.p6_conv_1(fpn_out0)
        pan_out0 = top_features + self.p6_conv_2(pan_out1)  #p6


        outputs = [pan_out3, pan_out2, pan_out1, pan_out0]

        return outputs


class RepBiFPANNeckP2P3(nn.Module):
    """RepBiFPANNeck with P2+P3 outputs only

    For 160x160 ROI micro-object detection.
    Architecture: Keep FULL backbone/neck for semantic features, output P2+P3 only.

    Input from backbone (when fuse_P2=True):
        x3: P2 (C2, stride=4)
        x2: P3 (C3, stride=8)
        x1: P4 (C4, stride=16)
        x0: P5 (C5, stride=32)

    Output to detection heads:
        pan_out_p2: P2 output (stride=4, 40x40 @ 160x160 input)
        pan_out_p3: P3 output (stride=8, 20x20 @ 160x160 input)

    Channel layout (width_multiple=0.25 for nano):
        Backbone: [64, 128, 256, 512, 1024] -> [16, 32, 64, 128, 256]
        Neck:     [256, 128, 128, 256, 256, 512] -> [64, 32, 32, 64, 64, 128]
        indices:  [5,   6,   7,   8,   9,   10]
    """

    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        # Top-down FPN path (keep FULL to extract deep semantics)
        # P5 -> P4
        self.reduce_layer0 = ConvBNReLU(
            in_channels=channels_list[4],   # C5: 1024 -> 256 (nano)
            out_channels=channels_list[5],  # 256 -> 64 (nano)
            kernel_size=1,
            stride=1
        )

        self.Bifusion0 = BiFusion(
            in_channels=[channels_list[3], channels_list[2]],  # C4:512, C3:256
            out_channels=channels_list[5],  # 256 -> 64 (nano)
        )

        self.Rep_p4 = RepBlock(
            in_channels=channels_list[5],   # 256 -> 64 (nano)
            out_channels=channels_list[5],  # 256 -> 64 (nano)
            n=num_repeats[5],
            block=block
        )

        # P4 -> P3
        self.reduce_layer1 = ConvBNReLU(
            in_channels=channels_list[5],   # 256 -> 64 (nano)
            out_channels=channels_list[6],  # 128 -> 32 (nano)
            kernel_size=1,
            stride=1
        )

        self.Bifusion1 = BiFusion(
            in_channels=[channels_list[2], channels_list[1]],  # C3:256, C2:128
            out_channels=channels_list[6],  # 128 -> 32 (nano)
        )

        self.Rep_p3 = RepBlock(
            in_channels=channels_list[6],   # 128 -> 32 (nano)
            out_channels=channels_list[6],  # 128 -> 32 (nano)
            n=num_repeats[6],
            block=block
        )

        # P3 -> P2 (extra branch for tiny objects)
        self.upsample_p2 = Transpose(
            in_channels=channels_list[6],   # 128 -> 32 (nano)
            out_channels=channels_list[7],  # 128 -> 32 (nano)
        )

        self.Rep_p2 = RepBlock(
            in_channels=channels_list[1] + channels_list[7],  # C2 + upsample: 128+128 -> 32+32
            out_channels=channels_list[7],  # 128 -> 32 (nano) - P2 output
            n=num_repeats[6],  # reuse p3 repeat count
            block=block
        )

        # Bottom-up PAN path (for feature refinement)
        # P2 -> P3
        self.downsample_p3 = ConvBNReLU(
            in_channels=channels_list[7],   # P2: 128 -> 32 (nano)
            out_channels=channels_list[7],  # 128 -> 32 (nano)
            kernel_size=3,
            stride=2
        )

        self.Rep_n3 = RepBlock(
            in_channels=channels_list[6] + channels_list[7],  # fpn_p3 + down_p2: 128+128
            out_channels=channels_list[8],  # 256 -> 64 (nano) - P3 output
            n=num_repeats[7],
            block=block
        )

    def forward(self, input):
        """
        Args:
            input: tuple of (P2, P3, P4, P5) from backbone

        Returns:
            list: [P2_out, P3_out] for detection heads
        """
        (x3, x2, x1, x0) = input  # P2, P3, P4, P5

        # Top-down FPN
        fpn_out0 = self.reduce_layer0(x0)  # P5 reduce
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])  # Fuse P5->P4->P3
        f_out0 = self.Rep_p4(f_concat_layer0)  # P4 level features

        fpn_out1 = self.reduce_layer1(f_out0)  # P4 reduce
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])  # Fuse P4->P3->P2
        fpn_out_p3 = self.Rep_p3(f_concat_layer1)  # P3 level features

        # P3 -> P2
        up_feat_p2 = self.upsample_p2(fpn_out_p3)
        p2_concat = torch.cat([up_feat_p2, x3], 1)  # Concat with backbone P2
        fpn_out_p2 = self.Rep_p2(p2_concat)  # P2 level features

        # Bottom-up PAN (refine with deeper features)
        down_feat_p3 = self.downsample_p3(fpn_out_p2)  # P2 -> P3
        p3_concat = torch.cat([down_feat_p3, fpn_out_p3], 1)
        pan_out_p3 = self.Rep_n3(p3_concat)  # Final P3 output

        # Return P2 first (finest scale first)
        outputs = [fpn_out_p2, pan_out_p3]

        return outputs


class RepBiFPANNeckP2P3P4P5(nn.Module):
    """RepBiFPANNeck with P2/P3/P4/P5 outputs (4-scale detection)

    Original YOLOv6n RepBiFPANNeck architecture extended with P2 head support.
    For 320x320 input with full range detection (15-180px targets).

    Input from backbone (when fuse_P2=True):
        x3: P2 (C2, stride=4)
        x2: P3 (C3, stride=8)
        x1: P4 (C4, stride=16)
        x0: P5 (C5, stride=32)

    Output to detection heads:
        [P2_out, P3_out, P4_out, P5_out]
        - P2_out: stride=4 (80x80 @ 320x320 input) for 15-40px targets
        - P3_out: stride=8 (40x40 @ 320x320 input) for 40-90px targets
        - P4_out: stride=16 (20x20 @ 320x320 input) for 80-160px targets
        - P5_out: stride=32 (10x10 @ 320x320 input) for 140-220px targets

    Channel layout (width_multiple=0.25 for nano):
        Backbone: [64, 128, 256, 512, 1024] -> [16, 32, 64, 128, 256]
        Neck out_channels: [256, 128, 128, 256, 256, 512, 128, 128, 256, 256, 512]
        indices:           [5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15]
    """

    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        # Top-down FPN: P5 -> P4 -> P3 -> P2

        # P5 -> P4
        self.reduce_layer0 = ConvBNReLU(
            in_channels=channels_list[4],
            out_channels=channels_list[5],
            kernel_size=1,
            stride=1
        )

        self.Bifusion0 = BiFusion(
            in_channels=[channels_list[3], channels_list[2]],
            out_channels=channels_list[5],
        )

        self.Rep_p4 = RepBlock(
            in_channels=channels_list[5],
            out_channels=channels_list[5],
            n=num_repeats[5],
            block=block
        )

        # P4 -> P3
        self.reduce_layer1 = ConvBNReLU(
            in_channels=channels_list[5],
            out_channels=channels_list[6],
            kernel_size=1,
            stride=1
        )

        self.Bifusion1 = BiFusion(
            in_channels=[channels_list[2], channels_list[1]],
            out_channels=channels_list[6],
        )

        self.Rep_p3 = RepBlock(
            in_channels=channels_list[6],
            out_channels=channels_list[6],
            n=num_repeats[6],
            block=block
        )

        # P3 -> P2 (NEW: extend to P2)
        self.upsample_p2 = Transpose(
            in_channels=channels_list[6],
            out_channels=channels_list[11],
        )

        self.Rep_p2 = RepBlock(
            in_channels=channels_list[1] + channels_list[11],
            out_channels=channels_list[11],
            n=num_repeats[6],
            block=block
        )

        # Bottom-up PAN: P2 -> P3 -> P4 -> P5

        # P2 -> P3 (NEW)
        self.downsample_p3 = ConvBNReLU(
            in_channels=channels_list[11],
            out_channels=channels_list[12],
            kernel_size=3,
            stride=2
        )

        self.Rep_n3 = RepBlock(
            in_channels=channels_list[6] + channels_list[12],
            out_channels=channels_list[8],
            n=num_repeats[7],
            block=block
        )

        # P3 -> P4
        self.downsample_p4 = ConvBNReLU(
            in_channels=channels_list[8],
            out_channels=channels_list[9],
            kernel_size=3,
            stride=2
        )

        self.Rep_n4 = RepBlock(
            in_channels=channels_list[5] + channels_list[9],
            out_channels=channels_list[13],
            n=num_repeats[8],
            block=block
        )

        # P4 -> P5
        self.downsample_p5 = ConvBNReLU(
            in_channels=channels_list[13],
            out_channels=channels_list[14],
            kernel_size=3,
            stride=2
        )

        self.Rep_n5 = RepBlock(
            in_channels=channels_list[5] + channels_list[14],
            out_channels=channels_list[10],
            n=num_repeats[8],
            block=block
        )

    def forward(self, input):
        """
        Args:
            input: tuple of (P2, P3, P4, P5) from backbone

        Returns:
            list: [P2_out, P3_out, P4_out, P5_out] for 4-scale detection heads
        """
        (x3, x2, x1, x0) = input  # P2, P3, P4, P5

        # Top-down FPN
        fpn_out0 = self.reduce_layer0(x0)
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
        fpn_out_p4 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(fpn_out_p4)
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])
        fpn_out_p3 = self.Rep_p3(f_concat_layer1)

        up_feat_p2 = self.upsample_p2(fpn_out_p3)
        p2_concat = torch.cat([up_feat_p2, x3], 1)
        fpn_out_p2 = self.Rep_p2(p2_concat)

        # Bottom-up PAN
        down_feat_p3 = self.downsample_p3(fpn_out_p2)
        p3_concat = torch.cat([down_feat_p3, fpn_out_p3], 1)
        pan_out_p3 = self.Rep_n3(p3_concat)

        down_feat_p4 = self.downsample_p4(pan_out_p3)
        p4_concat = torch.cat([down_feat_p4, fpn_out_p4], 1)
        pan_out_p4 = self.Rep_n4(p4_concat)

        down_feat_p5 = self.downsample_p5(pan_out_p4)
        p5_concat = torch.cat([down_feat_p5, fpn_out0], 1)
        pan_out_p5 = self.Rep_n5(p5_concat)

        # Return 4 outputs: [P2, P3, P4, P5]
        outputs = [fpn_out_p2, pan_out_p3, pan_out_p4, pan_out_p5]

        return outputs
