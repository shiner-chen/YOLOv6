# ET-YOLOv6n fine-tune config — ARD100 ROI dataset @ 320×320
#
# Dataset:   ARD100_roi200_native_to640_v0
#   Images:  200×200 ROI crops upscaled to 640×640, letterboxed to 320×320 at train
#   Purpose: Simulate ROI inference pipeline (ROI <320px → upscale → 320×320 model)
#
# Base model: et_yolov6n_ard100ft1_320/weights/best_ckpt.pt
#   (ET-YOLOv6n fine-tuned on ARD100 full-frame @ 320×320, epoch ~2)

model = dict(
    type='YOLOv6n',
    pretrained='/home/adlink/chenx/yolov6/runs/train/et_yolov6n_ard100ft1_320/weights/best_ckpt.pt',
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
        anchors_init=[[4,5,  7,7,   11,9],
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
    lr0=0.0005,          # lower LR: fine-tuning from an already fine-tuned checkpoint
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=1.0,
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
    scale=0.5,
    shear=0.0,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.05,
)
