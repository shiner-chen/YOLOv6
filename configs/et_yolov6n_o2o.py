# ET-YOLOv6n O2O config — C2fStar backbone + CrossTwoLevelBiFPANNeck
# Dual-assignment head (O2M topk=13 + O2O topk=1), NMS-free inference.
# Borrows from SDD-YOLO: STAL, ProgLoss, WIoU, QAT ConfidenceMarginLoss.
model = dict(
    type='ETYOLOv6n_O2O',
    pretrained=None,
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
        type='EffiDeHead_O2O',
        o2o=True,                   # enables O2O head + ComputeLoss_O2O
        num_layers=4,               # P2/P3/P4/P5
        strides=[4, 8, 16, 32],
        iou_type='wiou',            # Wise-IoU for small-target robustness
        use_dfl=False,
        reg_max=0,
        # Progressive loss schedule (SDD-YOLO ProgLoss)
        prog_loss_t1=50,            # λ_o2m=2, λ_o2o=1  before t1
        prog_loss_t2=150,           # linear transition t1→t2; λ_o2m=1, λ_o2o=3 after t2
        # QAT INT8 confidence margin loss (off by default; enable for QAT finetune)
        qat_mode=False,
        confidence_threshold=0.25,
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

training_mode = 'repvgg'  # training phase (branches not fused)

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
    mixup=0.0,
)
