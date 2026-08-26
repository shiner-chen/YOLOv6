# ET-YOLOv6n Fine-tune Configuration for Multi-scale Detection
# Base: runs/train/et_yolov6n_roi320_nwd2/weights/best_ckpt.pt (86.31% mAP@0.5)
# Goal: Extend detection to medium targets (40-120px) while maintaining tiny target performance
#
# Dataset: ARD100 ROI320 (35,850, avg 15.3px) + Diverse Filtered (400, 40-120px only)
# Total: 36,250 training images (98.9% tiny + 1.1% medium)
#
# Strategy: Fine-tune with adjusted NWD constant and lower learning rate
# Key differences from scratch training (et_yolov6n_ard100_roi320_nwd.py):
#   1. Load pretrained weights via --pretrain flag
#   2. NWD constant: 12.8 → 18.0 (expand coverage from 10-30px to 10-120px)
#   3. Learning rate: 0.005 → 0.001 (fine-tuning rate, 1/5 of scratch)
#   4. Epochs: 300 → 100 (shorter fine-tune)
#   5. Anchors: adjusted for extended target range

model = dict(
    type='YOLOv6n',
    pretrained=None,
    depth_multiple=0.33,
    width_multiple=0.25,           # Nano: 0.25

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
        num_layers=4,                # 4-scale: P2/P3/P4/P5
        p2_head=True,
        begin_indices=24,
        anchors=3,

        # Anchors — keep IDENTICAL to original model for fine-tuning
        # Let the model learn to adapt anchors during fine-tuning
        # Original anchors optimized for 15-18px targets will naturally extend
        # to cover 40-120px range through learning
        anchors_init=[[6,6,   10,8,  14,12 ],    # P2: small (original)
                      [15,12, 20,16, 28,22 ],     # P3: medium-small
                      [25,20, 35,30, 45,40 ],     # P4: medium
                      [40,35, 55,50, 75,70 ]],    # P5: large

        out_indices=[17, 20, 23, 26],
        strides=[4, 8, 16, 32],
        atss_warmup_epoch=0,
        iou_type='siou',
        use_dfl=False,
        reg_max=0,

        # === NWD settings for multi-scale ===
        # Original: nwd_constant=12.8 for 10-30px tiny objects
        # Adjusted: nwd_constant=18.0 to cover 10-120px range
        # Rationale:
        #   - ARD100 (98.9%): 15.3px avg → still well covered by 18.0
        #   - Diverse (1.1%): 40-120px → now adequately covered
        #   - NWD gradient smoothness extends to medium targets
        nwd_ratio=0.5,        # blend: 0.5*SIoU + 0.5*NWD
        nwd_constant=18.0,    # 🔥 Increased from 12.8 to cover medium targets

        distill_weight={
            'class': 1.0,
            'dfl': 1.0,
        },
    )
)

# Fine-tune solver settings
solver = dict(
    optim='SGD',
    lr_scheduler='Cosine',
    lr0=0.001,                     # 🔥 Fine-tuning: 0.005 → 0.001 (1/5)
    lrf=0.01,                      # Final lr = 0.001 × 0.01 = 0.00001
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
)

training_mode = 'repvgg'

# Data augmentation — IDENTICAL to original ROI320 NWD training
# Targets are ~15.3px avg in 320×320 ROI → ~3.8px on P2 feature map (stride=4)
# Even the diverse dataset's 40-120px targets become 10-30px on feature maps
# ANY scale-down, mosaic, or mixup would destroy these tiny features
data_aug = dict(
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.0,              # DISABLED: 3.8px target cannot afford any shrink
    shear=0.0,
    flipud=0.0,             # No vertical flip for drones
    fliplr=0.5,             # Horizontal flip OK
    mosaic=0.0,             # DISABLED: mosaic+resize destroys tiny features
    mixup=0.0,              # DISABLED: fatal for small objects
)

# ========================
# Fine-tuning Guide
# ========================
#
# Command:
#   torchrun --nproc_per_node=2 --master_port=29502 \
#     tools/train.py \
#     --conf configs/et_yolov6n_ard100_roi320_nwd_ft_multiscale.py \
#     --data data/ard100_roi320_merged.yaml \
#     --img-size 320 \
#     --batch-size 64 \
#     --epochs 100 \
#     --device 0,1 \
#     --workers 4 \
#     --output-dir runs/train \
#     --name et_yolov6n_roi320_nwd_ft_multiscale \
#     --eval-interval 5 \
#     --pretrain runs/train/et_yolov6n_roi320_nwd2/weights/best_ckpt.pt
#
# Expected Results:
#   - mAP@0.5 on tiny targets (10-30px): ~85-87% (maintain current 86.31%)
#   - mAP@0.5 on medium targets (40-120px): significant improvement from baseline
#   - Overall mAP@0.5: ~87-89% (slight improvement due to better multi-scale coverage)
#
# If tiny target performance drops >2%:
#   Run Stage 2 fine-tune on pure ARD100 with NWD=12.8 for 30-50 epochs to recover
