# ET-YOLOv6n fine-tune config — ARD100 full-frame dataset @ 640×640
#
# Pretrained:  runs/train/et_yolov6n_roi320_ft1/weights/best_ckpt.pt
#   (ET-YOLOv6n trained on ARD100_roi320 @ 320×320)
#
# Dataset:     data/ard100_full.yaml  (/home/adlink/data/ARD100_full)
#   train: 21334 images  (pos=19679  neg=1655  neg%=7.8)
#   val:    5595 images  (pos= 5128  neg= 467  neg%=8.3)
#
# Strategy:
#   - Same architecture, only input resolution changes 320→640
#   - Weights transfer cleanly (no arch change), so moderate LR is safe
#   - lr0=0.001 (same as ard100ft baseline), brief warmup
#   - Mosaic + small-scale augment kept to handle tiny drone targets
#   - scale=0.9 (wider range) because full 1920×1080 frames cropped/resized
#     to 640 leave the drone very small — allow more zoom-in augmentation
#   - stop_aug_last_n_epoch=15 to consolidate without mosaic at the end

model = dict(
    type='YOLOv6n',
    pretrained='/home/adlink/chenx/et-yolo/runs/train/et_yolov6n_roi320_ft1/weights/best_ckpt.pt',
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
        # Anchors designed for 640 input; P2 (stride 4) handles tiny drones.
        # Smallest anchor 4×5 px covers a ~10px drone after stride-4 feature map.
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
    lr0=0.001,           # moderate — arch unchanged, only resolution shifts
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=2.0,   # slightly longer warmup for resolution adaptation
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
    scale=0.9,           # wide zoom range: drone is tiny in full 1920×1080 frames
    shear=0.0,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,          # mosaic on — effective for small-object detection
    mixup=0.05,
)
