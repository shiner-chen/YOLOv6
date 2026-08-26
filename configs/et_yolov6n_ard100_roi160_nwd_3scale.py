# ET-YOLOv6n + NWD loss — ARD100 ROI160 3-Scale
#
# Based on successful 320×320 3-scale architecture:
#   - CrossLayerBifusion preserved (ET-YOLO's core innovation)
#   - CLAM channel attention mechanism
#   - SPPF (not CSPSPPF) for efficiency
#   - 3-scale detection: P2/P3/P4
#
# Key differences from 320×320:
#   1. Input size: 320×320 → 160×160
#   2. Feature maps: P2(40×40), P3(20×20), P4(10×10)
#   3. Target size: ~46×26 pixels (LARGER than 320 ROI - tighter crop!)
#   4. Epochs: 400 (vs 300 for 320, smaller input needs more iterations)
#
# Training command:
#   torchrun --nproc_per_node=2 --master_port=29500 \
#       tools/train.py \
#       --conf configs/et_yolov6n_ard100_roi160_nwd_3scale.py \
#       --data data/ard100_roi160_merged.yaml \
#       --img-size 160 \
#       --batch-size 128 \
#       --epochs 400 \
#       --device 0,1 \
#       --workers 4 \
#       --output-dir runs/train \
#       --name et_yolov6n_roi160_nwd_3scale \
#       --eval-interval 5

model = dict(
    type='YOLOv6n',
    pretrained=None,
    depth_multiple=0.33,
    width_multiple=0.25,           # Nano: 0.25
    backbone=dict(
        type='EfficientRepStar3Scale',
        # 4 channels for 3-scale: [P2_in, P3_in, P4_in, P4_out]
        # width_multiple=0.25: [64, 128, 256, 512] → [16, 32, 64, 128]
        num_repeats=[1, 6, 12, 18],
        out_channels=[64, 128, 256, 512],
        fuse_P2=True,
        cspsppf=False,  # Use SPPF (not CSPSPPF) for efficiency
    ),
    neck=dict(
        type='CrossTwoLevelBiFPANNeck3ScaleV2',
        # 8 channels for neck operations:
        # [0]:P4_reduced, [1]:P3_bifusion_out, [2]:P3→P2_up, [3]:P2_out,
        # [4]:P2→P3_down, [5]:P3_pan_out, [6]:P3→P4_down, [7]:P4_pan_out
        # Base values: [256, 128, 128, 128, 128, 128, 256, 256]
        # After width_multiple=0.25: [64, 32, 32, 32, 32, 32, 64, 64]
        num_repeats=[12, 12, 12, 12, 12],  # Rep_p3, Rep_p2, Rep_n3, Rep_n4, (spare)
        out_channels=[256, 128, 128, 128, 128, 128, 256, 256],
    ),
    head=dict(
        type='EffiDeHead',
        num_layers=3,
        p2_head=True,
        begin_indices=24,
        anchors=3,
        # Anchors for 160×160 input, 3-scale detection
        # Targets are LARGER than 320 ROI (tighter crop: avg 46×26 px vs 19×13)
        # P2 (stride=4, 40×40): 10-25 pixels
        # P3 (stride=8, 20×20): 20-40 pixels — primary layer
        # P4 (stride=16, 10×10): 35-60 pixels — larger targets
        anchors_init=[[8,6,   12,9,   18,12 ],   # P2: small-medium
                      [20,14, 28,20,  38,28 ],    # P3: medium (primary)
                      [42,32, 56,42,  72,56 ]],   # P4: large
        out_indices=[17, 20, 23],        # 3 outputs for 3-scale
        strides=[4, 8, 16],              # 3-scale strides
        atss_warmup_epoch=0,
        iou_type='siou',
        use_dfl=False,
        reg_max=0,
        # === NWD settings ===
        # ROI160 has LARGER targets than ROI320 (tighter crop, not downscale)
        # Measured: avg 46×26 px, geometric mean ≈ 34.8
        nwd_ratio=0.5,        # blend: 0.5*SIoU + 0.5*NWD
        nwd_constant=32.0,    # ~sqrt(46*26) ≈ 34.8, use 32 (vs 320 ROI's 12.8)
        distill_weight={
            'class': 1.0,
            'dfl': 1.0,
        },
    )
)

solver = dict(
    optim='SGD',
    lr_scheduler='Cosine',
    lr0=0.005,              # Initial learning rate (same as 320)
    lrf=0.01,               # Final lr = 0.005 * 0.01 = 0.00005
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
)

data_aug = dict(
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,              # Scale jitter for 160×160
    shear=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
)
