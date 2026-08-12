# ET-YOLOv6n O2O QAT (quantization-aware training) config
# Weights come from --weights argument (qat_train.py loads via load_state_dict()),
# so cfg.model.pretrained is set to None here.
#
# LR rationale:
#   QAT only adjusts quantizer scales and small weight residuals.
#   lr0=0.0001 → eff. peak = 0.0002 (×2 batch scaling: bs=32, bs_per_gpu=16)
#   This is ~1/50 of the O2O fine-tune effective peak (0.01).
#
# ProgLoss schedule (QAT — O2O-dominant from the start since model is already trained):
#   epoch  <  1  :  λ_o2m=2.0, λ_o2o=1.0   (brief O2M warm-up)
#   epoch  1– 3  :  linear transition
#   epoch  >  3  :  λ_o2m=1.0, λ_o2o=3.0   (O2O-dominant for 7 of 10 epochs)

model = dict(
    type='ETYOLOv6n_O2O',
    pretrained=None,            # ignored — qat_train.py loads from --weights
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
        prog_loss_t1=1,             # quick transition — O2O-dominant from epoch 3
        prog_loss_t2=3,
        qat_mode=True,              # enable ConfidenceMarginLoss (also forced by qat_train.py)
        confidence_threshold=0.25,
    )
)

solver = dict(
    optim='SGD',
    lr_scheduler='Cosine',
    lr0=0.0001,             # QAT lr — eff. peak ~0.0002, 50× below O2O fine-tune
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=0.5,      # very short: 5% of 10 QAT epochs
    warmup_momentum=0.8,
    warmup_bias_lr=0.001,   # 1/10 of O2O ft warmup_bias_lr (0.01)
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
    mosaic=0.5,     # reduced from 1.0: mosaic scrambles activation distributions
                    # that quantizers calibrate against — keep lighter during QAT
    mixup=0.0,      # disabled: mixup alpha-blending adds noise incompatible with
                    # INT8 scale estimation
)

ptq = dict(
    num_bits=8,                         # INT8 quantization
    calib_method='histogram',           # histogram > max for accuracy
    histogram_amax_method='entropy',    # entropy gives tighter clipping than percentile
    histogram_amax_percentile=99.99,    # used only when histogram_amax_method='percentile'
    calib_batches=300,                  # consistent with --calib-batches 300 CLI arg
    calib_output_path='runs/train/et_yolov6n_o2o_qat/',
    sensitive_layers_skip=False,
    sensitive_layers_list=[],
)
