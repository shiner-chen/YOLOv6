#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
ComputeLoss_O2O: Dual-assignment O2O loss for ET-YOLOv6n.

Borrows from SDD-YOLO:
  - STAL (Small-Target-Aware Label assignment)
  - ProgLoss (Progressive Loss weighting)
  - WIoU   (Wise-IoU)
  - ConfidenceMarginLoss (QAT INT8 boundary hinge)
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox, xywh2xyxy
from yolov6.assigners.stal_assigner import STALAssigner
from yolov6.models.losses.loss import VarifocalLoss, BboxLoss


class ConfidenceMarginLoss(nn.Module):
    """Hinge loss on cls_scores near the INT8 quantization boundary.

    Pushes foreground max-scores above  (threshold + margin)
    and background max-scores below     (threshold - margin),
    where margin = n_steps / 255 accounts for INT8 rounding error.
    Applied only to the O2O cls head (FP16 mixed-precision branch).
    """

    def __init__(self, threshold: float = 0.25, n_steps: int = 3):
        super().__init__()
        self.margin = n_steps / 255.0
        self.hi = threshold + self.margin   # fg lower-bound
        self.lo = threshold - self.margin   # bg upper-bound

    def forward(self, cls_scores: torch.Tensor, fg_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_scores : (B, N_all, nc)  sigmoid cls scores
            fg_mask    : (B, N_all)      bool foreground mask
        Returns:
            scalar loss
        """
        scores_max = cls_scores.max(dim=-1).values  # (B, N_all)

        pos = scores_max[fg_mask]
        neg = scores_max[~fg_mask]

        zero = scores_max.sum() * 0.0
        loss_pos = torch.clamp(self.hi - pos, min=0.0).mean() if pos.numel() > 0 else zero
        loss_neg = torch.clamp(neg - self.lo, min=0.0).mean() if neg.numel() > 0 else zero
        return loss_pos + loss_neg


class ComputeLoss_O2O:
    """Dual-assignment O2O loss for ET-YOLOv6n.

    Training uses two STAL assigners simultaneously:
      - O2M (topk=13): rich gradient signal for the backbone
      - O2O (topk=1): single best anchor per GT, enables NMS-free inference

    ProgLoss linearly shifts weight from O2M→O2O across epochs.
    WIoU is used in place of GIoU for small-target robustness.
    When qat_mode=True a ConfidenceMarginLoss is added on the O2O cls
    head to keep scores clear of the INT8 rounding boundary.
    """

    def __init__(
        self,
        num_classes: int = 80,
        ori_img_size: int = 320,
        use_dfl: bool = False,
        reg_max: int = 0,
        iou_type: str = 'wiou',
        fpn_strides: list = None,
        grid_cell_size: float = 5.0,
        grid_cell_offset: float = 0.5,
        prog_loss_t1: int = 50,
        prog_loss_t2: int = 150,
        qat_mode: bool = False,
        confidence_threshold: float = 0.25,
        loss_weight_cls: float = 1.0,
        loss_weight_iou: float = 2.5,
    ):
        if fpn_strides is None:
            fpn_strides = [4, 8, 16, 32]

        self.num_classes = num_classes
        self.ori_img_size = ori_img_size
        self.use_dfl = use_dfl      # always False for ET-YOLOv6n
        self.reg_max = reg_max      # always 0 for ET-YOLOv6n
        self.iou_type = iou_type
        self.fpn_strides = fpn_strides
        self.grid_cell_size = grid_cell_size
        self.grid_cell_offset = grid_cell_offset
        self.t1 = prog_loss_t1
        self.t2 = prog_loss_t2
        self.qat_mode = qat_mode
        self.loss_weight_cls = loss_weight_cls
        self.loss_weight_iou = loss_weight_iou

        # Anchor cache — invalidated when feature-map spatial sizes change
        self.cached_feat_sizes = None
        self.cached_anchors = None

        # O2M assigner: topk=13, many candidates for backbone gradients
        self.assigner_o2m = STALAssigner(
            topk=13, num_classes=num_classes,
            alpha=1.0, beta=6.0,
            gamma=0.5, area_thr=0.02, ori_img_size=ori_img_size,
        )
        # O2O assigner: topk=1, single best anchor per GT for NMS-free inference
        self.assigner_o2o = STALAssigner(
            topk=1, num_classes=num_classes,
            alpha=1.0, beta=6.0,
            gamma=0.5, area_thr=0.02, ori_img_size=ori_img_size,
        )

        self.varifocal_loss = VarifocalLoss().cuda()
        self.bbox_loss = BboxLoss(
            num_classes=num_classes,
            reg_max=reg_max,
            use_dfl=use_dfl,
            iou_type=iou_type,
        ).cuda()

        if qat_mode:
            self.conf_margin_loss = ConfidenceMarginLoss(
                threshold=confidence_threshold, n_steps=3,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prog_loss_weights(self, epoch_num: int):
        """Progressive loss weights (λ_o2m, λ_o2o) per SDD-YOLO schedule."""
        t1, t2 = self.t1, self.t2
        if epoch_num < t1:
            return 2.0, 1.0
        elif epoch_num < t2:
            t = (epoch_num - t1) / (t2 - t1)
            return 2.0 - t, 1.0 + 2.0 * t
        else:
            return 1.0, 3.0

    def _get_anchors(self, feats):
        """Return cached (anchors, anchor_points, n_anchors_list, stride_tensor)."""
        feat_sizes = [f.shape[2:] for f in feats]
        if feat_sizes == self.cached_feat_sizes and self.cached_anchors is not None:
            return self.cached_anchors
        self.cached_feat_sizes = feat_sizes
        self.cached_anchors = generate_anchors(
            feats, self.fpn_strides,
            self.grid_cell_size, self.grid_cell_offset,
            device=feats[0].device,
        )
        return self.cached_anchors

    def _preprocess_targets(self, targets, batch_size, scale_tensor):
        """Convert flat (n_targets, 6) target tensor to (B, max_gt, 5)."""
        targets_list = np.zeros((batch_size, 1, 5)).tolist()
        for i, item in enumerate(targets.cpu().numpy().tolist()):
            targets_list[int(item[0])].append(item[1:])
        max_len = max(len(l) for l in targets_list)
        targets = torch.from_numpy(
            np.array(list(map(
                lambda l: l + [[-1, 0, 0, 0, 0]] * (max_len - len(l)),
                targets_list,
            )))[:, 1:, :]
        ).to(scale_tensor.device)
        batch_target = targets[:, :, 1:5].mul_(scale_tensor)
        targets[..., 1:] = xywh2xyxy(batch_target)
        return targets

    def _compute_branch_loss(
        self, assigner, pred_cls, pred_reg,
        anchor_points, anchor_points_s, stride_tensor,
        gt_labels, gt_bboxes, mask_gt, step_num,
    ):
        """Run one assignment branch; return (loss_iou, loss_cls, fg_mask)."""
        # Decode ltrb distances → xyxy bboxes in stride-scaled space
        pred_bboxes = dist2bbox(pred_reg, anchor_points_s)

        try:
            target_labels, target_bboxes, target_scores, fg_mask = assigner(
                pred_cls.detach(),
                pred_bboxes.detach() * stride_tensor,
                anchor_points,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )
        except RuntimeError:
            # OOM fallback: retry on CPU
            torch.cuda.empty_cache()
            target_labels, target_bboxes, target_scores, fg_mask = assigner(
                pred_cls.detach().cpu().float(),
                (pred_bboxes.detach() * stride_tensor).cpu().float(),
                anchor_points.cpu().float(),
                gt_labels.cpu().float(),
                gt_bboxes.cpu().float(),
                mask_gt.cpu().float(),
            )
            target_labels  = target_labels.cuda()
            target_bboxes  = target_bboxes.cuda()
            target_scores  = target_scores.cuda()
            fg_mask        = fg_mask.cuda()

        if step_num % 10 == 0:
            torch.cuda.empty_cache()

        # Rescale target bboxes to stride space (same scale as pred_bboxes)
        target_bboxes_s = target_bboxes / stride_tensor

        # --- cls loss (varifocal) ---
        target_labels_cls = torch.where(
            fg_mask > 0,
            target_labels,
            torch.full_like(target_labels, self.num_classes),
        )
        one_hot_label = F.one_hot(
            target_labels_cls.long(), self.num_classes + 1
        )[..., :-1]
        loss_cls = self.varifocal_loss(pred_cls, target_scores, one_hot_label)
        target_scores_sum = target_scores.sum()
        if target_scores_sum > 1:
            loss_cls = loss_cls / target_scores_sum

        # --- bbox loss (WIoU) ---
        loss_iou, _ = self.bbox_loss(
            pred_reg,           # pred_dist — passed through but unused (use_dfl=False)
            pred_bboxes,        # decoded xyxy in stride space
            anchor_points_s,
            target_bboxes_s,
            target_scores,
            target_scores_sum,
            fg_mask,
        )

        return loss_iou, loss_cls, fg_mask

    # ------------------------------------------------------------------
    # Main call
    # ------------------------------------------------------------------

    def __call__(self, outputs, targets, epoch_num, step_num,
                 batch_height, batch_width):
        """
        Args:
            outputs       : (feats, cls_o2m, reg_o2m, cls_o2o, reg_o2o)
                              feats   — list of 4 feature maps (B, C, H, W)
                              cls_*   — (B, N_all, nc)  sigmoid cls scores
                              reg_*   — (B, N_all, 4)   ltrb distances
            targets       : (n_targets, 6)  [img_idx, cls, cx, cy, w, h]  normalized
            epoch_num     : current epoch (drives ProgLoss schedule)
            step_num      : current step  (drives anchor-cache flush)
            batch_height  : image height in pixels
            batch_width   : image width  in pixels

        Returns:
            loss       : scalar
            loss_items : (4,) detached [iou_o2m, iou_o2o, cls_o2m, cls_o2o]
        """
        feats, cls_o2m, reg_o2m, cls_o2o, reg_o2o = outputs

        # ---- anchors ----
        anchors, anchor_points, n_anchors_list, stride_tensor = self._get_anchors(feats)
        anchor_points_s = anchor_points / stride_tensor     # stride-scaled

        # ---- targets ----
        gt_scale = torch.tensor(
            [batch_width, batch_height, batch_width, batch_height],
            dtype=cls_o2m.dtype, device=cls_o2m.device,
        )
        batch_size = cls_o2m.shape[0]
        targets_proc = self._preprocess_targets(targets, batch_size, gt_scale)
        gt_labels = targets_proc[:, :, :1]      # (B, max_gt, 1)
        gt_bboxes = targets_proc[:, :, 1:]      # (B, max_gt, 4) xyxy
        mask_gt   = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # ---- ProgLoss weights ----
        lambda_o2m, lambda_o2o = self._prog_loss_weights(epoch_num)

        # ---- O2M branch ----
        loss_iou_o2m, loss_cls_o2m, _ = self._compute_branch_loss(
            self.assigner_o2m, cls_o2m, reg_o2m,
            anchor_points, anchor_points_s, stride_tensor,
            gt_labels, gt_bboxes, mask_gt, step_num,
        )

        # ---- O2O branch ----
        loss_iou_o2o, loss_cls_o2o, fg_mask_o2o = self._compute_branch_loss(
            self.assigner_o2o, cls_o2o, reg_o2o,
            anchor_points, anchor_points_s, stride_tensor,
            gt_labels, gt_bboxes, mask_gt, step_num,
        )

        # ---- Combine with ProgLoss ----
        loss = (
            lambda_o2m * (
                self.loss_weight_iou * loss_iou_o2m +
                self.loss_weight_cls * loss_cls_o2m
            ) +
            lambda_o2o * (
                self.loss_weight_iou * loss_iou_o2o +
                self.loss_weight_cls * loss_cls_o2o
            )
        )

        # ---- QAT ConfidenceMarginLoss (O2O cls head only) ----
        if self.qat_mode:
            loss_conf = self.conf_margin_loss(cls_o2o, fg_mask_o2o.bool())
            loss = loss + loss_conf

        loss_items = torch.stack([
            (self.loss_weight_iou * loss_iou_o2m).detach(),
            (self.loss_weight_iou * loss_iou_o2o).detach(),
            (self.loss_weight_cls * loss_cls_o2m).detach(),
            (self.loss_weight_cls * loss_cls_o2o).detach(),
        ])

        return loss, loss_items
