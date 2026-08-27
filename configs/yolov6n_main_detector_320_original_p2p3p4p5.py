# YOLOv6n Main Detector - Original Architecture with P2 Head (320x320)
# Dataset: main_detector_320_v2 (48,857 images)
#   - ARD100_roi320: 45,688 (93.5%, 15-40px small targets)
#   - Anti-UAV_crop320: 1,959 (4.0%, 30-80px medium targets)
#   - Anti-UAV_crop640: 804 (1.6%, 25-90px medium-large targets)
#   - Anti-UAV_crop800: 406 (0.8%, 48-180px large targets)
#
# Architecture: Original YOLOv6n + P2 detection head
#   - Backbone: EfficientRep (original)
#   - Neck: RepBiFPANNeckP2P3P4P5 (original RepBiFPANNeck + P2 extension)
#   - Head: 4-scale EffiDeHead (P2/P3/P4/P5)
#
# Training: From scratch with NWD loss
# Coverage: 15-180px targets (200m-30m range)
#
# Training command:
#   torchrun --nproc_per_node=2 --master_port=29502 \
#     tools/train.py \
#     --conf configs/yolov6n_main_detector_320_original_p2p3p4p5.py \
#     --data /home/adlink/data/main_detector_320_v2/main_detector_320.yaml \
#     --img-size 320 \
#     --batch-size 64 \
#     --epochs 400 \
#     --device 0,1 \
#     --workers 4 \
#     --output-dir runs/train \
#     --name yolov6n_main_detector_320_original_p2p3p4p5 \
#     --eval-interval 5

training_mode = 'repvgg'

model = dict(
    type='YOLOv6n',
    pretrained=None,
    depth_multiple=0.33,
    width_multiple=0.25,           # Nano: 0.25

    backbone=dict(
        type='EfficientRep',       # *** Original backbone (not EfficientRepStar) ***
        num_repeats=[1, 6, 12, 18, 6],
        out_channels=[64, 128, 256, 512, 1024],
        fuse_P2=True,              # Enable P2 feature extraction
        cspsppf=False,             # Use SPPF (not CSPSPPF) for original architecture
    ),

    neck=dict(
        type='RepBiFPANNeckP2P3P4P5',  # *** NEW: Original architecture + P2 ***
        num_repeats=[12, 12, 12, 12],
        # out_channels: [P5_reduce, P4_fpn, P3_fpn, P4_pan, P5_pan, P5_out,
        #                P2_fpn, P2_pan, P3_pan, P4_pan_out, P5_pan_out]
        # 11 channels for 4-scale architecture with P2
        out_channels=[256, 128, 128, 256, 256, 512, 128, 128, 256, 256, 512],
    ),

    head=dict(
        type='EffiDeHead',
        num_layers=4,              # *** 4-scale: P2/P3/P4/P5 ***
        p2_head=True,              # *** Enable P2 head indexing ***
        begin_indices=24,
        anchors=3,

        # *** CRITICAL: Specify input channels for RepBiFPANNeckP2P3P4P5 outputs ***
        # RepBiFPANNeckP2P3P4P5 outputs: [P2=32ch, P3=64ch, P4=64ch, P5=128ch]
        # indices in channels_list: [11, 8, 13, 10]
        in_channels=[32, 64, 64, 128],  # [P2, P3, P4, P5]

        # Anchors for full range 15-180px in 320x320 input
        # Same as yolov6n_main_detector_320_v2.py for consistency
        anchors_init=[
            [10,10,  18,15,  28,22],      # P2: 15-40px small targets (ARD100)
            [35,28,  52,42,  75,60],      # P3: 40-90px medium targets
            [85,70,  120,100, 165,140],   # P4: 80-160px large targets
            [140,120, 180,160, 220,200]   # P5: 140-220px extra-large targets
        ],

        out_indices=[17, 20, 23, 26],  # *** 4 outputs: [P2, P3, P4, P5] ***
        strides=[4, 8, 16, 32],        # *** 4 strides ***
        atss_warmup_epoch=0,
        iou_type='siou',
        use_dfl=False,
        reg_max=0,

        # === NWD settings for multi-scale (15-180px) ===
        # Identical to yolov6n_main_detector_320_v2.py
        nwd_ratio=0.5,        # blend: 0.5*SIoU + 0.5*NWD
        nwd_constant=38.0,    # Optimized for 15-180px range

        distill_weight={
            'class': 1.0,
            'dfl': 1.0,
        },
    )
)

# Training from scratch - identical to yolov6n_main_detector_320_v2.py
solver = dict(
    optim='SGD',
    lr_scheduler='Cosine',
    lr0=0.005,                     # Standard scratch training rate
    lrf=0.01,                      # Final lr = 0.005 × 0.01 = 0.00005
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
)

# Data augmentation - identical to yolov6n_main_detector_320_v2.py
data_aug = dict(
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,              # Moderate scale aug: 0.5x-1.5x
    shear=0.0,
    flipud=0.0,             # No vertical flip for drones
    fliplr=0.5,             # Horizontal flip OK
    mosaic=0.0,             # DISABLED: harmful for small targets
    mixup=0.0,              # DISABLED: fatal for small objects
)

# ========================
# Architecture Comparison
# ========================
#
# Original YOLOv6n (P3/P4/P5):
#   - Backbone: EfficientRep
#   - Neck: RepBiFPANNeck
#   - Head: 3-scale (P3/P4/P5)
#
# This config (Original + P2):
#   - Backbone: EfficientRep (same)
#   - Neck: RepBiFPANNeckP2P3P4P5 (extended with P2)
#   - Head: 4-scale (P2/P3/P4/P5)
#
# Training params identical to:
#   configs/yolov6n_main_detector_320_v2.py
#
# Expected Results:
#   Similar to yolov6n_main_detector_320_v2 (mAP@0.5 ~87%)
#   Original architecture may have slightly different performance
#   but training parameters are optimized for this task
