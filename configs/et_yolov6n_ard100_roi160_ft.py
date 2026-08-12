# ET-YOLOv6n fine-tune config — ARD100 ROI @ 160×160, stage 2
#
# Pretrained:  runs/train/et_yolov6n_ard100_roi160/weights/best_ckpt.pt
#   (ROI-160 stage-1 best, epoch 14, mAP@0.5=0.668)
#
# Changes vs stage-1:
#   - pretrained → stage-1 best_ckpt (ep14, mAP@0.5=0.668)
#   - epochs: 30 → 80  (via --epochs 80)
#   - stop_aug_last_n_epoch: 15 → 20  (via --stop_aug_last_n_epoch 20)
#   - lr0: 0.001 → 0.0005  (model already adapted to ROI domain)
#   - scale: 0.5 (unchanged, [0.5,1.5] range appropriate for 15px drones)
#   - eval-interval: 3  (via --eval-interval 3)

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
    lr0=0.0005,          # lower start LR — model already adapted from stage-1 (ep14)
    lrf=0.05,            # final LR = lr0 * lrf = 2.5e-5
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=2.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.01,
)

training_mode = 'repvgg'

data_aug = dict(
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,           # [0.5, 1.5] random scale — unchanged per user preference
    shear=0.0,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.05,
)
