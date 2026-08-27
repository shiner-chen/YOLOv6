# YOLOv6n Main Detector - P2/P3/P4/P5 Four-Scale Detection (320x320)
# Dataset: main_detector_320_v2 (48,857 images)
#   - ARD100_roi320: 45,688 (93.5%, 15-40px small targets)
#   - Anti-UAV_crop320: 1,959 (4.0%, 30-80px medium targets)
#   - Anti-UAV_crop640: 804 (1.6%, 25-90px medium-large targets)
#   - Anti-UAV_crop800: 406 (0.8%, 48-180px large targets)
#
# Architecture: EfficientRepStar + CrossTwoLevelBiFPANNeck + 4-scale EffiDeHead
# Coverage: 15-180px targets (200m-30m range)
# Training: From scratch with NWD loss
#
# This config is compatible with et-yolov6n-roi160-p2p3 branch architecture:
#   - Supports both P2/P3 2-head (num_layers=2, p2_head=False)
#   - And P2/P3/P4/P5 4-head (num_layers=4, p2_head=True)
#
# Training command:
#   torchrun --nproc_per_node=2 --master_port=29502 \
#     tools/train.py \
#     --conf configs/yolov6n_main_detector_320_v2_p2p3p4p5.py \
#     --data /home/adlink/data/main_detector_320_v2/main_detector_320.yaml \
#     --img-size 320 \
#     --batch-size 64 \
#     --epochs 200 \
#     --device 0,1 \
#     --workers 4 \
#     --output-dir runs/train \
#     --name yolov6n_main_detector_320_v2_p2p3p4p5 \
#     --eval-interval 5

training_mode = 'repvgg'

model = dict(
    type='YOLOv6n',
    pretrained=None,
    depth_multiple=0.33,
    width_multiple=0.25,           # Nano: 0.25

    backbone=dict(
        type='EfficientRepStar',
        num_repeats=[1, 6, 12, 18, 6],
        out_channels=[64, 128, 256, 512, 1024],
        fuse_P2=True,              # Enable P2 feature extraction
        cspsppf=True,              # Use CSPSPPF for better feature fusion
    ),

    neck=dict(
        type='CrossTwoLevelBiFPANNeck',
        num_repeats=[12, 12, 12, 12, 12],
        out_channels=[256, 128, 128, 256, 256, 512, 128, 128],
    ),

    head=dict(
        type='EffiDeHead',
        num_layers=4,              # *** 4-scale: P2/P3/P4/P5 ***
        p2_head=True,              # *** Enable P2 head indexing ***
        begin_indices=24,
        anchors=3,

        # Anchors for full range 15-180px in 320x320 input
        # Feature map analysis:
        #   P2 (stride=4, 80×80):  15-40px  → 3.75-10px on feature map
        #   P3 (stride=8, 40×40):  40-90px  → 5-11.25px on feature map
        #   P4 (stride=16, 20×20): 80-160px → 5-10px on feature map
        #   P5 (stride=32, 10×10): 140-220px → 4.4-6.9px on feature map
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
        # Range: 15-180px → geometric mean = sqrt(15*180) ≈ 52
        # Use 38.0 to better cover large targets (120-180px)
        # Provides smooth IoU gradient across full target range
        nwd_ratio=0.5,        # blend: 0.5*SIoU + 0.5*NWD
        nwd_constant=38.0,    # Optimized for 15-180px range

        distill_weight={
            'class': 1.0,
            'dfl': 1.0,
        },
    )
)

# Training from scratch
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

# Data augmentation for multi-scale detection
# Scale augmentation simulates dynamic ROI (200-640px crop → resize to 320)
data_aug = dict(
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,              # Moderate scale aug: 0.5x-1.5x (simulates ROI 213-480)
    shear=0.0,
    flipud=0.0,             # No vertical flip for drones
    fliplr=0.5,             # Horizontal flip OK
    mosaic=0.0,             # DISABLED: harmful for small targets (15-40px)
    mixup=0.0,              # DISABLED: fatal for small objects
)

# ========================
# Expected Results
# ========================
#
# Based on main_detector_320_v2 training (mAP@0.5 = 87.7%):
#   - Small targets (15-40px, ARD100): mAP@0.5:0.95 ~52%
#   - Medium targets (40-90px): mAP@0.5:0.95 ~58%
#   - Large targets (>96px): mAP@0.5:0.95 ~51%
#   - Overall mAP@0.5: 85-88%
#   - Precision: ~91%, Recall: ~82%
#   - Inference speed: ~0.4ms (1400+ FPS on single GPU)
#
# Model specifications:
#   - Parameters: ~2.85M
#   - GFLOPs: ~2.32
#   - Input: 320×320×3
#   - Output: 4 detection heads (P2/P3/P4/P5)
