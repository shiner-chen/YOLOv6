# YOLOv6n + P2/P3 2-Head + NWD loss — ARD100 ROI160
#
# Original YOLOv6 architecture with motion-guided ROI detection:
#   - Full EfficientRep backbone (C1-C5) for deep semantic features
#   - Full RepBiFPANNeckP2P3 for multi-scale feature fusion
#   - P2/P3 detection heads only (stride=4, stride=8)
#   - RepVGG reparameterization for optimal inference performance
#
# Key design principles:
#   ✓ Keep full Backbone/Neck: extract deep semantics → fuse into P2/P3
#   ✓ Remove P4/P5 heads only: no medium/large targets in 160×160 ROI
#   → Result: High semantic classification + precise localization + low FLOPs
#
# Feature map sizes (160×160 input):
#   P2: 40×40 (stride=4)  → detects 4-20 px targets
#   P3: 20×20 (stride=8)  → detects 16-32 px targets
#
# Training command:
#   torchrun --nproc_per_node=2 --master_port=29500 \
#       tools/train.py \
#       --conf configs/yolov6n_ard100_roi160_p2p3_nwd.py \
#       --data data/ard100_roi160_merged.yaml \
#       --img-size 160 \
#       --batch-size 128 \
#       --epochs 400 \
#       --device 0,1 \
#       --workers 4 \
#       --output-dir runs/train \
#       --name yolov6n_roi160_p2p3_nwd \
#       --eval-interval 5

training_mode = 'repvgg'  # RepVGG training mode for reparameterization

model = dict(
    type='YOLOv6n',
    pretrained=None,
    depth_multiple=0.33,
    width_multiple=0.25,           # Nano: 0.25

    backbone=dict(
        type='EfficientRep',
        # IMPORTANT: Keep FULL 5-stage backbone to extract deep semantics
        # Even though we only output P2/P3, backbone goes to C5 for rich features
        num_repeats=[1, 6, 12, 18, 6],
        out_channels=[64, 128, 256, 512, 1024],
        fuse_P2=True,              # Enable P2 output (stride=4)
        cspsppf=False,             # Use SPPF (not CSPSPPF) for nano efficiency
    ),

    neck=dict(
        type='RepBiFPANNeckP2P3',
        # IMPORTANT: Keep FULL neck to fuse multi-scale features
        # Deep features (C4/C5) are fused into P2/P3 via FPN/PAN
        # Final P2/P3 contain both semantic and spatial information
        # num_repeats: [Rep_p4, Rep_p3, Rep_n3, Rep_n4]
        num_repeats=[12, 12, 12, 12],
        # out_channels: [P5_reduce, P4_fpn, P3_fpn, P2_out, P2_down, P3_out]
        # Base values after width_multiple=0.25: [64, 32, 32, 32, 32, 64]
        out_channels=[256, 128, 128, 256, 256, 512],
    ),

    head=dict(
        type='EffiDeHead',
        num_layers=2,              # *** KEY: 2 layers (P2 + P3 only) ***
        p2_head=False,             # Use standard 2-layer indexing
        begin_indices=24,
        anchors=3,

        # Anchors for 160×160 input, 2-scale detection
        # Based on ARD100 dataset statistics in ROI160
        # Targets in ROI160 are LARGER than full-frame (tighter crop, not downscale)
        # Expected target sizes: 8-40 pixels (avg ~20px)
        # P2 (stride=4, 40×40): 4-20 pixels — primary for tiny targets
        # P3 (stride=8, 20×20): 16-40 pixels — secondary for small targets
        anchors_init=[
            [6,4,   10,7,   15,10],     # P2: 4-20px tiny targets (primary)
            [20,14, 28,20,  38,28]      # P3: 16-40px small targets (secondary)
        ],

        out_indices=[17, 20],          # *** 2 outputs: P2, P3 ***
        strides=[4, 8],                # *** stride=4, stride=8 only ***

        atss_warmup_epoch=0,
        iou_type='siou',               # SIoU loss for bbox regression
        use_dfl=False,                 # Nano model doesn't use DFL
        reg_max=0,

        # === NWD (Normalized Wasserstein Distance) Loss Settings ===
        # ROI160 has LARGER targets than full-frame 640×640
        # Measured from ARD100 ROI160: avg ~20×15 px, geometric mean ≈ 17.3
        # Use higher NWD ratio for better tiny object localization
        nwd_ratio=0.6,             # 60% NWD + 40% SIoU (higher than 320's 0.5)
        nwd_constant=17.0,         # ~sqrt(20*15) ≈ 17.3 (vs 320 ROI's 32.0)

        distill_weight={
            'class': 1.0,
            'dfl': 1.0,
        },
    )
)

# Solver settings (based on ET-YOLOv6n ROI160 3-scale successful config)
solver = dict(
    optim='SGD',
    lr_scheduler='Cosine',
    lr0=0.005,                 # Initial learning rate (same as ET-YOLO ROI160)
    lrf=0.01,                  # Final lr = 0.005 * 0.01 = 0.00005
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
)

# Data augmentation settings (same as ET-YOLO ROI160 3-scale)
data_aug = dict(
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,               # No rotation for Anti-UAV (upright targets)
    translate=0.1,
    scale=0.5,                 # Scale jitter for 160×160
    shear=0.0,
    flipud=0.0,                # No vertical flip for Anti-UAV
    fliplr=0.5,                # 50% horizontal flip
    mosaic=1.0,                # Always use mosaic augmentation
    mixup=0.0,                 # No mixup (better for small targets)
)

# Training notes:
# 1. Epochs: 400 (same as ET-YOLO ROI160, smaller input needs more iterations)
# 2. Batch size: 128 (2 GPUs × 64 per GPU)
# 3. Image size: 160×160
# 4. Dataset: ard100_roi160_merged.yaml
# 5. Evaluation interval: 5 epochs
# 6. Expected training time: ~8-10 hours on 2×RTX 3090
#
# Performance expectations:
# - FLOPs: ~0.3-0.5 GFLOPs (vs 1.2G for 320 3-scale, 4.5G for 640 full)
# - Inference: ~2-3ms on RK3588 (vs 50ms for 640 full-frame)
# - Recall: Higher than 640 full-frame (maintains original resolution)
# - Precision: Higher than shallow P2/P3 (deep semantic suppression)
#
# Comparison with ET-YOLO ROI160 3-scale:
# - Architecture: Original YOLOv6 vs ET-YOLO (RepVGG vs CrossLayerBifusion)
# - Detection heads: P2/P3 (2 heads) vs P2/P3/P4 (3 heads)
# - Expected: Lower FLOPs, better inference speed, similar accuracy
# - Advantage: RepVGG reparameterization for edge device deployment
