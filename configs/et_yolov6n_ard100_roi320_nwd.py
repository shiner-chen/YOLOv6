# ET-YOLOv6n + NWD loss — ARD100 ROI320 (Nano + 320×320 + 4-scale P2/P3/P4/P5)
#
# Key differences from et_yolov6s_ard100_roi640_nwd.py:
#   1. Model size: Small → Nano (width_multiple: 0.50 → 0.25)
#   2. Input size: 640×640 → 320×320
#   3. Detection head: Keep 4-scale (P2/P3/P4/P5) for architecture consistency
#      Note: P5 feature map is 10×10 on 320 input, rarely contributes but kept for simplicity
#
# Rationale:
#   - 320×320 ROI directly cropped from 1080p preserves target size (~17.5px)
#   - Target representation on P2 (stride=4): 17.5/4 = 4.4 cells (same as 640 input!)
#   - 4-scale head: P2(80×80), P3(40×40), P4(20×20), P5(10×10)
#   - P5 layer contribution expected to be minimal but kept to avoid architecture modification
#   - NWD loss: smooth gradients for tiny objects (avg 17.5×11.6 px)
#   - Computation: 75% reduction vs 640×640 input
#
# Training command:
#   python tools/train.py \
#       --conf configs/et_yolov6n_ard100_roi320_nwd.py \
#       --data data/ard100_roi320.yaml \
#       --img-size 320 \
#       --batch-size 64 \
#       --epochs 300 \
#       --device 0,1 \
#       --output-dir runs/train \
#       --name et_yolov6n_roi320_nwd \
#       --eval-interval 5

model = dict(
    type='YOLOv6n',                # Nano model
    pretrained=None,
    depth_multiple=0.33,
    width_multiple=0.25,           # Nano: 0.25 (vs Small: 0.50)
    backbone=dict(
        type='EfficientRepStar',
        num_repeats=[1, 6, 12, 18, 6],      # KEEP FULL 5-layer (P1-P5)
        out_channels=[64, 128, 256, 512, 1024],
        fuse_P2=True,
        cspsppf=True,
    ),
    neck=dict(
        type='CrossTwoLevelBiFPANNeck',
        num_repeats=[12, 12, 12, 12, 12],   # KEEP FULL structure
        out_channels=[256, 128, 128, 256, 256, 512, 128, 128],
    ),
    head=dict(
        type='EffiDeHead',
        num_layers=4,                # Keep 4-scale head (P5 feature map is 10×10 on 320 input)
        p2_head=True,
        begin_indices=24,
        anchors=3,
        # Anchors for 320×320 input, 4-scale detection
        # Targets avg 17.5×11.6 px in ROI
        # P2 (stride=4, 80×80): 4-18 pixels — primary for 17.5px targets (4.4 cells)
        # P3 (stride=8, 40×40): 10-30 pixels — secondary (2.2 cells)
        # P4 (stride=16, 20×20): 20-50 pixels — larger/distant targets (1.1 cells)
        # P5 (stride=32, 10×10): 40+ pixels — very rare, but kept for architecture consistency
        anchors_init=[[6,6,   10,8,  14,12 ],   # P2: small (avg 17.5×11.6 fits here)
                      [15,12, 20,16, 28,22 ],    # P3: medium-small
                      [25,20, 35,30, 45,40 ],    # P4: medium
                      [40,35, 55,50, 75,70 ]],   # P5: large (10×10 feature map, rarely triggered)
        out_indices=[17, 20, 23, 26],# P2/P3/P4/P5 outputs (keep all 4)
        strides=[4, 8, 16, 32],      # 4-scale (same as standard)
        atss_warmup_epoch=0,
        iou_type='siou',
        use_dfl=False,
        reg_max=0,
        # === NWD settings (same as Small-640-NWD for fair comparison) ===
        nwd_ratio=0.5,        # blend: 0.5*SIoU + 0.5*NWD
        nwd_constant=12.8,    # ~ mean object size in px (17.5×11.6 → geometric mean ≈ 14)
        distill_weight={
            'class': 1.0,
            'dfl': 1.0,
        },
    )
)

solver = dict(
    optim='SGD',
    lr_scheduler='Cosine',
    lr0=0.005,                     # Same as Small-640-NWD
    lrf=0.01,                      # Final lr = 0.005 × 0.01 = 0.00005
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
)

training_mode = 'repvgg'

# Data augmentation for small objects — IDENTICAL to Small-640-NWD.
# Targets are ~17.5px in 320×320 ROI → 4.4px on P2 feature map (stride=4).
# ANY scale-down or mosaic would destroy these 4px features.
data_aug = dict(
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.0,                     # DISABLED: 4px target cannot afford any shrink
    shear=0.0,
    flipud=0.0,                    # No vertical flip for drones
    fliplr=0.5,                    # Horizontal flip OK
    mosaic=0.0,                    # DISABLED: mosaic+resize destroys 4px features
    mixup=0.0,                     # DISABLED: fatal for small objects
)
