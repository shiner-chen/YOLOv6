# ET-YOLOv6s — 4-scale (P2/P3/P4/P5), 640×640 input
#
# Scaled up from ET-YOLOv6n by doubling width_multiple (0.25 → 0.50).
# Architecture is identical: EfficientRepStar backbone + CrossTwoLevelBiFPANNeck + 4-scale head.
#
# Effective channel sizes after width scaling (×0.50):
#   Backbone : [32,  64, 128, 256, 512]   (nano: [16, 32, 64, 128, 256])
#   Neck     : [128, 64, 64, 128, 128, 256, 64, 64]
#
# type='ETYOLOv6s' (not 'YOLOv6s') — engine.py distill_ns path requires num_layers=3,
# but this model uses the 4-scale head (num_layers=4); the custom type avoids that conflict.
#
# Training command (single GPU, 640×640):
#   python tools/train.py \
#       --conf configs/et_yolov6s.py \
#       --data data/your_dataset.yaml \
#       --img-size 640 \
#       --batch-size 16 \
#       --epochs 300 \
#       --device 0 \
#       --output-dir runs/train/et_yolov6s
#
# Multi-GPU (e.g. 4 GPUs, effective batch = 64):
#   python -m torch.distributed.launch --nproc_per_node 4 tools/train.py \
#       --conf configs/et_yolov6s.py \
#       --data data/your_dataset.yaml \
#       --img-size 640 \
#       --batch-size 16 \
#       --epochs 300 \
#       --device 0,1,2,3 \
#       --output-dir runs/train/et_yolov6s

model = dict(
    type='ETYOLOv6s',
    pretrained=None,
    depth_multiple=0.33,
    width_multiple=0.50,          # 2× nano → ~4× parameters (~19 M vs ~5 M)
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
        # raw channels (before width scaling); index comment for reference:
        #   [5]256 [6]128 [7]128 [8]256 [9]256 [10]512 [11]128 [12]128
        out_channels=[256, 128, 128, 256, 256, 512, 128, 128],
    ),
    head=dict(
        type='EffiDeHead',
        num_layers=4,
        p2_head=True,               # strides [4,8,16,32]; adds P2 branch for small objects
        begin_indices=24,
        anchors=3,
        anchors_init=[[4,5,  7,7,   11,9 ],    # P2 stride=4  — tiny objects
                      [10,13, 19,19, 33,23],    # P3 stride=8
                      [30,61, 59,59, 59,119],   # P4 stride=16
                      [116,90, 185,185, 373,326]], # P5 stride=32
        out_indices=[17, 20, 23, 26],
        strides=[4, 8, 16, 32],
        atss_warmup_epoch=0,
        iou_type='siou',            # SIoU matches ET-YOLOv6n; swap to 'giou' if needed
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
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
)

training_mode = 'repvgg'  # training phase (Rep branches not fused)

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
    mixup=0.15,   # small amount; larger capacity of s model can absorb light mixup
                  # set to 0.0 to match nano if dataset is small (< 5k images)
)
