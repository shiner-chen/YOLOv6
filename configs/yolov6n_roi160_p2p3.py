# YOLOv6n ROI160 P2/P3 2-Head Configuration (Original YOLOv6 Architecture)
#
# Motion-guided ROI detection architecture based on ORIGINAL YOLOv6:
#   - Input: 160×160 ROI patches (cropped from motion blob centers)
#   - Backbone: Full EfficientRep (preserving deep semantic features)
#   - Neck: Full RepBiFPANNeck (feature fusion from all levels)
#   - Head: Only P2 (/4) + P3 (/8) detection heads (删除P4检测头)
#
# WHY use original YOLOv6 instead of ET-YOLO:
#   ✓ RepVGG reparameterization: 训练时多分支，推理时融合为单3x3卷积
#   ✓ Hardware-efficient: 重参后的结构对NPU/GPU极度友好
#   ✓ Proven performance: YOLOv6官方架构，经过大规模验证
#
# Architecture rationale (per discussion):
#   ✓ Keep full Backbone (C1-C5): deep layers extract semantics
#   ✓ Keep full Neck (RepBiFPAN): FPN/PAN fuses multi-scale features into P2/P3
#   ✓ Remove P4 head only: no need for stride=16 in 160×160 ROI
#   → Result: Rich semantics + precise localization + minimal FLOPs
#
# Feature map sizes (160×160 input):
#   Backbone outputs:
#     C2: 80×80   (/2)
#     C3: 40×40   (/4)  → feeds into P2
#     C4: 20×20   (/8)  → feeds into P3
#     C5: 10×10   (/16)
#
#   Neck outputs (for detection heads):
#     P2: 40×40 (stride=4)  → detects 4-20 px targets
#     P3: 20×20 (stride=8)  → detects 16-40 px targets
#     (P4: deleted, was 10×10, stride=16)
#
# Training command:
#   torchrun --nproc_per_node=2 --master_port=29500 \
#       tools/train.py \
#       --conf configs/yolov6n_roi160_p2p3.py \
#       --data data/ard100_roi160.yaml \
#       --img-size 160 \
#       --batch-size 128 \
#       --epochs 400 \
#       --device 0,1 \
#       --workers 4 \
#       --output-dir runs/train \
#       --name yolov6n_roi160_p2p3

training_mode = 'repvgg'  # RepVGG training mode for reparameterization

model = dict(
    type='YOLOv6n',
    pretrained=None,
    depth_multiple=0.33,
    width_multiple=0.25,           # Nano: 0.25

    backbone=dict(
        type='EfficientRep',
        # IMPORTANT: 保留完整的5-stage backbone
        # 即使只输出P2/P3，backbone仍需要到C5来提取深层语义特征
        # num_repeats: [C1, C2, C3, C4, C5]
        num_repeats=[1, 6, 12, 18, 6],
        # out_channels: [C1, C2, C3, C4, C5]
        out_channels=[64, 128, 256, 512, 1024],
        fuse_P2=True,              # Enable P2 output (stride=4)
        cspsppf=False,             # Use SPPF (not CSPSPPF) for nano model
    ),

    neck=dict(
        type='RepBiFPANNeckP2P3',
        # IMPORTANT: 保留完整的neck进行多尺度特征融合
        # 深层特征(C4/C5)通过FPN自顶向下融合到P2/P3
        # 再通过PAN自底向上refinement
        # 最终P2/P3包含了所有层级的语义+空间信息
        # num_repeats: [Rep_p4, Rep_p3, Rep_n3, Rep_n4]
        num_repeats=[12, 12, 12, 12],
        # out_channels: [P5_reduce, P4_fpn, P3_fpn, P2_out, P2_down, P3_out]
        # 索引: [5, 6, 7, 8, 9, 10]
        out_channels=[256, 128, 128, 256, 256, 512],
    ),

    head=dict(
        type='EffiDeHead',
        num_layers=2,              # *** KEY: 3 → 2 (only P2 + P3) ***
        p2_head=False,             # Use standard indexing for 2-layer mode
        begin_indices=24,
        anchors=3,

        # Anchors for 160×160 input, 2-scale detection
        # 针对Anti-UAV场景中4-40像素的微小目标
        # P2 (stride=4, 40×40): 适合4-20像素目标
        # P3 (stride=8, 20×20): 适合16-40像素目标
        anchors_init=[
            [6,4,   10,7,   14,10],     # P2: 4-20px tiny targets
            [20,14, 28,20,  38,28]      # P3: 16-40px small targets
        ],

        out_indices=[17, 20],          # *** 2 outputs: P2, P3 ***
        strides=[4, 8],                # *** stride=4, stride=8 only ***

        atss_warmup_epoch=0,
        iou_type='siou',
        use_dfl=False,                 # Nano model不使用DFL
        reg_max=0,

        # === NWD loss settings (for tiny objects) ===
        # 160×160 ROI中典型目标约10-20像素，几何平均≈14
        nwd_ratio=0.6,             # 60% NWD + 40% SIoU，更关注微小目标
        nwd_constant=14.0,         # ~sqrt(10*20) ≈ 14.1

        distill_weight={
            'class': 1.0,
            'dfl': 1.0,
        },
    )
)

solver = dict(
    optim='SGD',
    lr_scheduler='Cosine',
    lr0=0.005,                 # 160×160小分辨率用较小学习率
    lrf=0.01,                  # 最终lr = 0.005 * 0.01 = 0.00005
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
    scale=0.5,
    shear=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
)
