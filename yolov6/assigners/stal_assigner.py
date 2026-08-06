"""STAL (Small-Target Aware Label) Assigner.

Extends TaskAlignedAssigner by injecting a size-dependent boost factor into
the alignment metric, making small GT boxes more competitive in the top-k
candidate selection step.

Boost formula (per GT box):
    small_factor = 1.0 + gamma * exp(-area_ratio / area_thr)

where:
    area_ratio = gt_box_area / (ori_img_size ** 2)

For small boxes (area_ratio << area_thr), small_factor ≈ 1 + gamma.
For large boxes (area_ratio >> area_thr), small_factor ≈ 1.
"""

import torch
from yolov6.assigners.tal_assigner import TaskAlignedAssigner


class STALAssigner(TaskAlignedAssigner):
    """Small-Target Aware Label Assigner.

    Args:
        topk (int): Number of top candidates per GT. Default: 13.
        num_classes (int): Number of foreground classes. Default: 80.
        alpha (float): Score exponent in alignment metric. Default: 1.0.
        beta (float): IoU exponent in alignment metric. Default: 6.0.
        gamma (float): Boost magnitude for small targets. Default: 0.5.
        area_thr (float): Area-ratio threshold that separates small from
            large objects (fraction of image area). Default: 0.02.
        ori_img_size (int): Reference image side length in pixels used to
            normalise GT box area. Default: 640.
    """

    def __init__(self,
                 topk=13,
                 num_classes=80,
                 alpha=1.0,
                 beta=6.0,
                 gamma=0.5,
                 area_thr=0.02,
                 ori_img_size=640,
                 eps=1e-9):
        super().__init__(topk=topk, num_classes=num_classes,
                         alpha=alpha, beta=beta, eps=eps)
        self.gamma = gamma
        self.area_thr = area_thr
        self.ori_img_size = ori_img_size

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
        """Compute box metrics with small-target aware boost.

        Overrides parent to inject size-dependent boost into align_metric.
        """
        # Get base metrics from parent
        align_metric, overlaps = super().get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes)

        # --- Small-target boost -------------------------------------------------
        # GT box area in pixels: (bs, n_max_boxes)
        gt_w = (gt_bboxes[..., 2] - gt_bboxes[..., 0]).clamp(min=0.0)
        gt_h = (gt_bboxes[..., 3] - gt_bboxes[..., 1]).clamp(min=0.0)
        gt_area = gt_w * gt_h

        img_area = float(self.ori_img_size) ** 2
        area_ratio = gt_area / img_area  # (bs, n_max_boxes)

        # Boost factor: large for small boxes, near 1 for large boxes
        # shape: (bs, n_max_boxes)
        small_factor = 1.0 + self.gamma * torch.exp(-area_ratio / self.area_thr)

        # Apply boost: align_metric shape (bs, n_max_boxes, n_anchors)
        align_metric = align_metric * small_factor.unsqueeze(-1)
        # -------------------------------------------------------------------------

        return align_metric, overlaps
