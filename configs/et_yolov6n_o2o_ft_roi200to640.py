# ET-YOLOv6n O2O fine-tune config — ARD100_roi200_native_to640_v0
#
# Dataset: 200x200 ROI region upscaled to 640x640, train at 320x320.
#   Effective scale factor: 640/200 * (320/640) = 1.6x vs native.
#   Target sizes @320 input: p5=27px, median=93px, p95=167px
#   Negative ratio: ~20% (vs 66% in roi320) — model sees more positives
#
# Key differences from et_yolov6n_o2o_ft.py (roi320):
#   stal_area_thr: 0.001 → 0.02  (27px→1.35x boost, 93px→1.01x, 167px→1.00x)
#   prog_loss_t1:  20    → 15    (targets easier to assign, less O2M warm-up needed)
#   prog_loss_t2:  80    → 65    (transition ends earlier, 15 epochs O2O-dominant)
#   epochs:        100   → 80    (larger targets converge faster)
#
# ProgLoss schedule:
#   epoch  <  15 : λ_o2m=2.0, λ_o2o=1.0  (O2M-dominant, head warm-up)
#   epoch 15- 65 : linear transition
#   epoch >  65  : λ_o2m=1.0, λ_o2o=3.0  (O2O-dominant, NMS-free quality)

model = dict(
    type='ETYOLOv6n_O2O',
    pretrained='runs/train/et_yolov6n_roi320_ft1/weights/best_ckpt.pt',
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
        o2o=True,
        num_layers=4,
        strides=[4, 8, 16, 32],
        iou_type='wiou',
        use_dfl=False,
        reg_max=0,
        prog_loss_t1=15,
        prog_loss_t2=65,
        stal_area_thr=0.02,   # calibrated for 27-167px targets @320 input
        qat_mode=False,
        confidence_threshold=0.25,
    )
)

solver = dict(
    optim='SGD',
    lr_scheduler='Cosine',
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
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
