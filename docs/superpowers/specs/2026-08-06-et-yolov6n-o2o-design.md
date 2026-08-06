# ET-YOLOv6n-O2O 设计规范

**日期**：2026-08-06  
**分支**：`feat/et-yolov6n-o2o`  
**目标**：在保持 ET-YOLOv6n 速度优势的前提下，实现 anchor-free、NMS-free、O2O 推理，降低反无人机场景误检率；同时集成 QAT 量化感知训练流程供部署使用。

---

## 1. 背景与目标

### 1.1 问题

ET-YOLOv6n（C2f-Star backbone + CrossTwoLevelBiFPANNeck）速度快，但使用标准 O2M 分配 + NMS 推理，在反无人机应用场景中会出现多余检测框，带偏云台跟踪。

SDD-YOLO 实现了 NMS-free O2O 推理，误检率极低，但其 backbone（CSP + C2PSA）比 ET-YOLOv6n 慢。

### 1.2 方案 A 核心思路

- **不改动** backbone（EfficientRepStar）和 neck（CrossTwoLevelBiFPANNeck）
- 在 head 层增加 O2O 并联分支，训练时同时运行 O2M+O2O，推理时只用 O2O
- 借鉴 SDD-YOLO 的 STAL 标签分配、ProgLoss 动态权重、WIoU
- 增加 QAT 微调流程

### 1.3 成功标准

| 指标 | 目标 |
|------|------|
| 推理速度 | 不低于原 ET-YOLOv6n |
| 误检率 | 接近 SDD-YOLO（O2O 推理无 NMS） |
| mAP | 不低于原 ET-YOLOv6n ±1% |
| QAT INT8 精度损失 | ≤ 1% mAP |

---

## 2. 整体架构

```
输入图像
  └─ EfficientRepStar Backbone（C2f-Star）     ← 不改动
       └─ CrossTwoLevelBiFPANNeck              ← 不改动
            └─ EffiDeHead_O2O（新）
                 ├─ 共享: stem / cls_conv / reg_conv  (P2/P3/P4/P5 各一套)
                 ├─ O2M 预测头: cls_pred_o2m / reg_pred_o2m
                 └─ O2O 预测头: cls_pred_o2o / reg_pred_o2o

训练: L_total = ProgLoss(epoch) × [λ_o2m·L_STAL(O2M) + λ_o2o·L_STAL(O2O)]
推理: 只跑 O2O 分支 → confidence 阈值过滤 → 无 NMS
```

---

## 3. Head 设计

**文件**：`yolov6/models/heads/effidehead_o2o.py`

### 3.1 每个 scale 的结构

```
feat[i]  (来自 neck，4 个 scale: P2/P3/P4/P5)
  └─ stem[i]          (1×1 ConvBNSiLU)  ┐
       ├─ cls_conv[i]  (3×3 ConvBNSiLU)  ├─ 共享权重，O2M 和 O2O 都走这里
       └─ reg_conv[i]  (3×3 ConvBNSiLU)  ┘
            ├─ cls_feat → cls_pred_o2m[i]  (Conv2d: ch→nc)     O2M 分支
            │           → cls_pred_o2o[i]  (Conv2d: ch→nc)     O2O 分支
            └─ reg_feat → reg_pred_o2m[i]  (Conv2d: ch→4)      O2M 分支
                        → reg_pred_o2o[i]  (Conv2d: ch→4)      O2O 分支
```

stem/cls_conv/reg_conv 三层在 O2M 和 O2O 之间**共享参数**。每个 scale 仅新增 2 个轻量 `Conv2d`（O2O 预测头），参数增量 < 1%。

`use_dfl=False, reg_max=0`（沿用 ET-YOLOv6n 现有配置，不引入 DFL）。

### 3.2 训练 forward 返回值

```python
return feats, cls_o2m, reg_o2m, cls_o2o, reg_o2o
# feats: list[Tensor]，用于生成 anchor points
# cls_*: (B, N_all, nc)，sigmoid 后的分类分数
# reg_*: (B, N_all, 4)，ltrb 格式距离预测
```

### 3.3 推理 forward 返回值

只跑 O2O 分支，输出格式与现有 `effidehead.py` 推理输出**完全相同**：

```python
return torch.cat([pred_bboxes, objectness, cls_scores], dim=-1)
# shape: (B, N_all, 4+1+nc)
# N_all = H2*W2 + H3*W3 + H4*W4 + H5*W5
```

下游 evaluate/demo 代码无需修改。

### 3.4 初始化

O2M 和 O2O 的 `cls_pred` 都用 `prior_prob=1e-2` 初始化偏置，`reg_pred` 偏置初始化为 1.0，与现有 head 一致。

---

## 4. 标签分配：STAL（Small Target Aware Label Assignment）

**文件**：`yolov6/assigners/stal_assigner.py`

基于现有 `TaskAlignedAssigner`，修改 alignment metric 计算：

```python
# 标准 TAL:
align_metric = cls_score ** alpha * iou ** beta

# STAL 新增小目标因子:
# gt_area = gt_w * gt_h（像素面积，归一化到 ori_img_size^2）
# small_factor = 1 + gamma * exp(-gt_area / area_thr)
# gamma=0.5, area_thr=0.02（即 gt 面积 < 2% 图像面积时增益）
align_metric = cls_score ** alpha * iou ** beta * small_factor
```

- O2M 分支使用：`STALAssigner(topk=13, alpha=1.0, beta=6.0, gamma=0.5)`
- O2O 分支使用：`STALAssigner(topk=1,  alpha=1.0, beta=6.0, gamma=0.5)`

topk=1 等效于"每个 GT 只分配给最高质量的一个 anchor"，推理时每个目标只有一个预测框，无需 NMS。

---

## 5. 损失函数：ProgLoss + WIoU

**文件**：`yolov6/models/losses/loss_o2o.py`

### 5.1 ProgLoss 动态权重

```python
# T1, T2 从 config 传入，默认 T1=50, T2=150（总 epoch=300）
if epoch < T1:
    lambda_o2m, lambda_o2o = 2.0, 1.0   # O2M 主导，backbone 充分收敛
elif epoch < T2:
    # 线性插值
    t = (epoch - T1) / (T2 - T1)
    lambda_o2m = 2.0 - t          # 2.0→1.0
    lambda_o2o = 1.0 + 2.0 * t   # 1.0→3.0
else:
    lambda_o2m, lambda_o2o = 1.0, 3.0   # O2O 主导，单一匹配精化
```

### 5.2 WIoU（替换 SIoU）

**文件**：`yolov6/utils/figure_iou.py`（新增 `iou_type='wiou'` 支持）

WIoU 原理：对高质量框降低梯度权重，让低质量（小目标）框获得更多训练机会。

```python
# 在 IOUloss.__call__ 中新增 wiou 分支:
# wise_factor = exp(rho^2 / C^2)   # rho=中心距, C=凸对角线
# wiou = (1 - iou) * wise_factor.detach()
```

`wise_factor.detach()` 阻止二阶梯度回传，与论文一致。

### 5.3 损失结构

```python
L_total = lambda_o2m * (L_cls_o2m + L_iou_o2m)
        + lambda_o2o * (L_cls_o2o + L_iou_o2o)

# L_cls: VarifocalLoss（沿用现有）
# L_iou: WIoU（新增）
# 不再有 L_dfl（use_dfl=False）
```

---

## 6. 模型入口

**文件**：`yolov6/models/yolo.py`

### 6.1 `build_network()` 新增分支

```python
def build_network(config, channels, num_classes, num_layers,
                  fuse_ab=False, distill_ns=False, o2o=False):  # 新增 o2o 参数
    ...
    if o2o:
        from yolov6.models.heads.effidehead_o2o import Detect, build_effidehead_layer
        head_layers = build_effidehead_layer(channels_list, num_classes,
                                             num_layers=num_layers)
        head = Detect(num_classes, num_layers, head_layers=head_layers)
    elif distill_ns:
        ...
```

### 6.2 `Model.__init__()` 透传参数

```python
def __init__(self, config, channels=3, num_classes=None,
             fuse_ab=False, distill_ns=False, o2o=False):
    self.backbone, self.neck, self.detect = build_network(
        config, channels, num_classes, num_layers,
        fuse_ab=fuse_ab, distill_ns=distill_ns, o2o=o2o)
```

### 6.3 `build_model()` 透传

```python
def build_model(cfg, num_classes, device, fuse_ab=False, distill_ns=False, o2o=False):
    model = Model(cfg, channels=3, num_classes=num_classes,
                  fuse_ab=fuse_ab, distill_ns=distill_ns, o2o=o2o).to(device)
    return model
```

---

## 7. 配置文件

**文件**：`configs/et_yolov6n_o2o.py`

基于 `configs/et_yolov6n.py`，改动如下：

```python
model = dict(
    ...
    head=dict(
        type='EffiDeHead_O2O',
        o2o=True,
        num_layers=4,
        p2_head=True,
        use_dfl=False,
        reg_max=0,
        iou_type='wiou',           # SIoU → WIoU
        loss_weight_o2m=1.0,       # 初始值，ProgLoss 会动态覆盖
        loss_weight_o2o=2.0,
        prog_loss_t1=50,           # epoch: O2M 主导结束
        prog_loss_t2=150,          # epoch: O2O 主导开始
        strides=[4, 8, 16, 32],
        atss_warmup_epoch=0,
        stal_gamma=0.5,            # 小目标增益系数
        stal_area_thr=0.02,        # GT 面积 < 2% 图像时触发
    )
)
```

solver、data_aug 等其余部分与 `et_yolov6n.py` 相同。

---

## 8. 训练入口适配

**文件**：`tools/train.py`（已有）

在 `build_model` 调用处增加 `o2o=cfg.model.head.get('o2o', False)` 透传，以及：

- loss 实例化时根据 `o2o` 标志选择 `loss_o2o.ComputeLoss` 或原有 `loss.ComputeLoss`
- `epoch_num` 已在 loss `__call__` 里传入，ProgLoss 直接使用

---

## 9. QAT 流程

**文件**：`tools/quantization/tensorrt/training_aware/qat_train.py`

### 9.1 流程

```
1. 加载 FP32 O2O 模型（--weights 参数）
2. quant_modules.initialize()  → Conv/Linear 替换为 fake-quant 版本
3. 用 calibration dataset（~1000 张）运行前向，校准量化范围（percentile 99.99）
4. QAT fine-tune：5–10 epoch，lr = 原训练 lr × 0.01
5. 导出：
   quant_nn.TensorQuantizer.use_fb_fake_quant = True
   torch.onnx.export(model, ...)  → qat_int8.onnx
   trtexec --int8 --calib ... → engine.trt
```

### 9.2 注意事项

- `proj_conv`（DFL 的 proj 层，现已无用但仍存在于代码）设为不量化，避免意外影响
- O2M 分支在 QAT 阶段**不参与推理**，但仍在模型参数里；导出前需通过 `model.detect.o2o_only = True` 关闭 O2M forward 路径（或在 export 时 trace 出无 O2M 的子图）
- 兼容 RK3588：QAT 导出的 ONNX 通过 `rknn-toolkit2` 做 PTQ 时，INT8 量化范围更准确

---

## 10. 文件清单

### 新增文件

| 文件 | 说明 |
|------|------|
| `yolov6/models/heads/effidehead_o2o.py` | 双分支 O2M/O2O 检测头 |
| `yolov6/models/losses/loss_o2o.py` | ProgLoss + STAL + WIoU 双路 loss |
| `yolov6/assigners/stal_assigner.py` | 小目标感知 TAL assigner |
| `configs/et_yolov6n_o2o.py` | 新模型配置 |
| `tools/quantization/tensorrt/training_aware/qat_train.py` | QAT 微调脚本 |

### 修改文件

| 文件 | 改动 |
|------|------|
| `yolov6/models/yolo.py` | `build_network` / `Model` / `build_model` 增加 `o2o` 参数 |
| `yolov6/utils/figure_iou.py` | `IOUloss` 新增 `iou_type='wiou'` 分支 |
| `tools/train.py` | `build_model` 调用处透传 `o2o` 标志，loss 工厂选择 |

### 不改动文件

- `yolov6/models/et_modules.py`（C2fStar、StarBlock、CrossLayerBifusion）
- `yolov6/models/efficientrep_star.py`
- `yolov6/models/reppan_cross.py`
- 所有现有 config、现有 head/loss（向后兼容）

---

## 11. 风险与注意事项

1. **O2O topk=1 稳定性**：训练初期 O2O 分支梯度稀疏，ProgLoss 前期压低 O2O 权重可缓解；如果仍不稳定，可增加 warmup 阶段让 O2M 先收敛再开放 O2O。

2. **共享 stem 的影响**：O2M 和 O2O 共享 stem/cls_conv/reg_conv，两路梯度叠加可能造成冲突。如果精度不达标，可将 O2O 的三个共享层改为独立参数（增加约 2× head 参数量，仍远小于 backbone）。

3. **WIoU 的 `wise_factor`**：需用 `.detach()` 阻断二阶梯度，否则训练不稳定。

4. **QAT O2M 分支导出**：量化后导出 ONNX 时必须确保 O2M 分支不出现在计算图里，否则 TensorRT 会优化出额外输出节点。
