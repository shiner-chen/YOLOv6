# ET-YOLOv6n + NWD loss — ARD100 ROI640 方案4：修正Anchor + 全面优化
# 基于实测目标尺寸：16.7×11.3 px
#
# 关键发现：
#   - 640 ROI中的目标平均16.7×11.3 px（比320的19.6×13.6更小！）
#   - 原因：640 ROI覆盖更大视野，相同绝对像素目标显得更小
#   - 方案3的anchor [4,4, 6,5, 9,7] 严重偏小
#
# 方案4修正：
#   1. ✅ Anchor基于实测尺寸重新设计
#   2. ✅ 保留方案3的优化参数（lr, weight_decay, translate等）
#   3. ✅ 针对小目标特性进一步调整
#
# Anchor设计原则：
#   实测目标: 16.7×11.3 px (min=8×5, max=38×28)
#   P2 stride=4: 目标在feature map上是 4.2×2.8 cells
#   P2 anchors应覆盖 8-24px 范围
#   P3 anchors应覆盖 16-40px 范围
#
# Training command:
#   torchrun --nproc_per_node=2 --master_port=29532 \
#       tools/train.py \
#       --conf configs/et_yolov6n_ard100_roi640_nwd_v4.py \
#       --data-path data/ard100_roi640.yaml \
#       --img-size 640 \
#       --batch-size 32 \
#       --epochs 150 \
#       --device 0,1 \
#       --name et_yolov6n_roi640_nwd_v4

model = dict(
    type='YOLOv6n',
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
        type='EffiDeHead',
        num_layers=4,
        p2_head=True,
        begin_indices=24,
        anchors=3,
        # ===== 方案4：修正Anchor配置 =====
        # 基于实测数据：目标平均16.7×11.3 px，范围8-38 px
        # P2 (stride=4, 160×160 feature map): 主检测层
        #   - 目标在feature map上: 2.0-9.5 cells (平均4.2 cells)
        #   - Anchor设计: 覆盖8-24px范围
        # P3 (stride=8, 80×80): 次要检测层
        #   - 目标在feature map上: 1.0-4.8 cells (平均2.1 cells)
        #   - Anchor设计: 覆盖16-40px范围
        # P4/P5: 保留但很少触发
        anchors_init=[[10,7,  17,11, 24,17],   # P2: 实测目标范围 (10-24px)
                      [20,14, 28,20, 38,28],    # P3: 中等尺寸
                      [45,35, 60,48, 80,65],    # P4: 大目标（很少）
                      [95,80, 130,110, 180,150]], # P5: 极大目标（极少）
        out_indices=[17, 20, 23, 26],
        strides=[4, 8, 16, 32],
        atss_warmup_epoch=0,
        iou_type='siou',
        use_dfl=False,
        reg_max=0,
        # NWD settings
        nwd_ratio=0.5,
        nwd_constant=12.8,
        distill_weight={
            'class': 1.0,
            'dfl': 1.0,
        },
    )
)

# 方案3优化参数（保留）
solver = dict(
    optim='SGD',
    lr_scheduler='Cosine',
    lr0=0.0025,                    # 降低50%适应更大输入
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.002,            # 4倍正则化防止过拟合
    warmup_epochs=5.0,             # 更长预热
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
)

training_mode = 'repvgg'

# 数据增强 - 方案3优化（保留）
data_aug = dict(
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.05,                # 减小平移适应小目标
    scale=0.0,
    shear=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=0.0,
    mixup=0.0,
)
