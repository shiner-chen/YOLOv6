# ET-YOLOv6n fine-tune config — ARD100 ROI @ 160×160, stage 2 (retry)
#
# Pretrained:  runs/train/et_yolov6n_ard100_roi160/weights/best_ckpt.pt
#   (ROI-160 stage-1 best, epoch 14, mAP@0.5=0.668)
#
# Stage-2 v1 失败原因：lr0_eff = 0.0005×4 = 0.002，对已收敛模型太高，
# 导致 epoch2 即从 0.668 退步至 0.628，峰值仅 0.647。
#
# 本次修正：lr0=0.00005 → lr0_eff = 0.00005×4 = 0.0002 (降低10×)
# 这样不会破坏已学好的权重，只做精细调整。

model = dict(
    type='YOLOv6n',
    pretrained='/home/adlink/chenx/et-yolo/runs/train/et_yolov6n_ard100_roi160/weights/best_ckpt.pt',
    depth_multiple=0.33,
    width_multiple=0.25,
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
        anchors_init=[[4,5,   7,7,   11,9],
                      [10,13, 19,19,  33,23],
                      [30,61, 59,59,  59,119],
                      [116,90, 185,185, 373,326]],
        out_indices=[17, 20, 23, 26],
        strides=[4, 8, 16, 32],
        atss_warmup_epoch=0,
        iou_type='siou',
        use_dfl=False,
        reg_max=0,
        distill_weight={
            'class': 1.0,
            'dfl': 1.0,
        },
    )
)

solver = dict(
    optim='SGD',
    lr_scheduler='Cosine',
    lr0=0.00005,         # effective lr = 0.00005 × (256/64) = 0.0002
                         # 10× lower than ft1 — avoids destroying converged weights
    lrf=0.1,             # final lr = 0.00005 × 0.1 = 5e-6 (effective 2e-5)
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=1.0,   # short warmup — model already in domain
    warmup_momentum=0.8,
    warmup_bias_lr=0.001,
)

training_mode = 'repvgg'

data_aug = dict(
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.05,
)
