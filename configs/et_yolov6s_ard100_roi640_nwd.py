# ET-YOLOv6s + NWD loss — ARD100 ROI640 (Baseline architecture + NWD)
#
# Modification vs baseline: ADD Normalized Wasserstein Distance to bbox loss.
#   bbox_loss = (1 - nwd_ratio) * SIoU_loss + nwd_ratio * NWD_loss
#
# Rationale: IoU-based losses are extremely sensitive to 1-2 px shifts on tiny
# targets (avg 17.5×11.6 px → 4.4×2.9 px on P2). NWD models boxes as 2-D
# Gaussians and degrades smoothly, giving stabler gradients for small objects.
# Reference: Wang et al., 2021, https://arxiv.org/abs/2110.13389
#
# ONLY difference from baseline: nwd_ratio / nwd_constant in head.
# Everything else (backbone, neck, anchors, solver, aug) is IDENTICAL to
# configs/et_yolov6s_ard100_roi640.py for a strict single-variable comparison.
#
# Training: FROM SCRATCH, 400 epochs, eval_interval=5 (same as baseline).

model = dict(
    type='ETYOLOv6s',
    pretrained=None,
    depth_multiple=0.33,
    width_multiple=0.50,
    backbone=dict(
        type='EfficientRepStar',
        num_repeats=[1, 6, 12, 18, 6],
        out_channels=[64, 128, 256, 512, 1024],
        fuse_P2=True,
        cspsppf=True,
    ),
    neck=dict(
        type='CrossTwoLevelBiFPANNeck',
        num_repeats=[12, 12, 12, 12, 12],
        out_channels=[256, 128, 128, 256, 256, 512, 128, 128],
    ),
    head=dict(
        type='EffiDeHead',
        num_layers=4,
        p2_head=True,
        begin_indices=24,
        anchors=3,
        anchors_init=[[6,6,   10,8,  14,12 ],
                      [15,12, 20,16, 28,22 ],
                      [25,20, 35,30, 45,40 ],
                      [40,35, 55,50, 75,70 ]],
        out_indices=[17, 20, 23, 26],
        strides=[4, 8, 16, 32],
        atss_warmup_epoch=0,
        iou_type='siou',
        use_dfl=False,
        reg_max=0,
        # === NWD settings (the ONLY change vs baseline) ===
        nwd_ratio=0.5,        # blend: 0.5*SIoU + 0.5*NWD (paper's default)
        nwd_constant=12.8,    # ~ mean object size in px (targets avg 17.5×11.6)
        distill_weight={
            'class': 1.0,
            'dfl': 1.0,
        },
    )
)

solver = dict(
    optim='SGD',
    lr_scheduler='Cosine',
    lr0=0.005,                     # Lower than 0.01 — ARD-MAV training collapsed with 0.01
    lrf=0.01,                      # Final lr = 0.005 × 0.01 = 0.00005 (fine-tuning phase)
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
)

training_mode = 'repvgg'

# Data augmentation for small objects — IDENTICAL to baseline.
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
