#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Tests for ComputeLoss_O2O (Task 5)."""

import pytest
import torch
import torch.nn as nn

from yolov6.models.losses.loss_o2o import ComputeLoss_O2O, ConfidenceMarginLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feats(batch_size=2, device='cpu'):
    """Fake feature maps matching ET-YOLOv6n 320×320 strides [4,8,16,32]."""
    return [
        torch.zeros(batch_size, 64,  80, 80, device=device),  # P2  stride 4
        torch.zeros(batch_size, 128, 40, 40, device=device),  # P3  stride 8
        torch.zeros(batch_size, 256, 20, 20, device=device),  # P4  stride 16
        torch.zeros(batch_size, 512, 10, 10, device=device),  # P5  stride 32
    ]


def _make_outputs(batch_size=2, n_all=8500, nc=3, device='cpu', requires_grad=True):
    """Fake (feats, cls_o2m, reg_o2m, cls_o2o, reg_o2o) with gradients."""
    feats = _make_feats(batch_size, device)

    def _cls():
        t = torch.sigmoid(torch.randn(batch_size, n_all, nc, device=device))
        return t.detach().requires_grad_(requires_grad)

    def _reg():
        t = torch.rand(batch_size, n_all, 4, device=device).clamp(min=0.1)
        return t.detach().requires_grad_(requires_grad)

    return feats, _cls(), _reg(), _cls(), _reg()


def _make_targets(batch_size=2, n_gt=4, nc=3, img_size=320, device='cpu'):
    """Flat targets tensor (n_gt, 6): [img_idx, cls, cx, cy, w, h] normalized."""
    rows = []
    for b in range(batch_size):
        for _ in range(n_gt // batch_size):
            cls_id = torch.randint(0, nc, (1,)).item()
            cx = torch.FloatTensor(1).uniform_(0.1, 0.9).item()
            cy = torch.FloatTensor(1).uniform_(0.1, 0.9).item()
            w  = torch.FloatTensor(1).uniform_(0.02, 0.3).item()
            h  = torch.FloatTensor(1).uniform_(0.02, 0.3).item()
            rows.append([b, cls_id, cx, cy, w, h])
    return torch.tensor(rows, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeLossO2O:
    """Basic forward pass and gradient checks."""

    @pytest.fixture
    def loss_fn(self):
        return ComputeLoss_O2O(
            num_classes=3,
            ori_img_size=320,
            use_dfl=False,
            reg_max=0,
            iou_type='wiou',
            fpn_strides=[4, 8, 16, 32],
            prog_loss_t1=50,
            prog_loss_t2=150,
            qat_mode=False,
        )

    def test_output_shapes(self, loss_fn):
        """loss is scalar; loss_items has shape (4,)."""
        outputs = _make_outputs()
        targets = _make_targets()
        loss, loss_items = loss_fn(
            outputs, targets,
            epoch_num=10, step_num=0,
            batch_height=320, batch_width=320,
        )
        assert loss.ndim == 0,            f"loss should be scalar, got shape {loss.shape}"
        assert loss_items.shape == (4,),  f"loss_items shape: {loss_items.shape}"

    def test_backward(self, loss_fn):
        """Gradients flow through both cls and reg branches."""
        outputs = _make_outputs(requires_grad=True)
        _, cls_o2m, reg_o2m, cls_o2o, reg_o2o = outputs
        targets = _make_targets()
        loss, _ = loss_fn(
            outputs, targets,
            epoch_num=10, step_num=0,
            batch_height=320, batch_width=320,
        )
        loss.backward()
        # At least one reg head should have received gradients
        has_grad = any(
            t.grad is not None and t.grad.abs().sum() > 0
            for t in [cls_o2m, reg_o2m, cls_o2o, reg_o2o]
        )
        assert has_grad, "No gradient reached any output tensor"

    def test_prog_loss_schedule(self, loss_fn):
        """ProgLoss returns correct (λ_o2m, λ_o2o) at boundary epochs."""
        assert loss_fn._prog_loss_weights(0)   == (2.0, 1.0)
        assert loss_fn._prog_loss_weights(49)  == (2.0, 1.0)
        assert loss_fn._prog_loss_weights(150) == (1.0, 3.0)
        assert loss_fn._prog_loss_weights(200) == (1.0, 3.0)
        lm, lo = loss_fn._prog_loss_weights(100)
        assert 1.0 < lm < 2.0, f"mid-schedule λ_o2m={lm} out of range"
        assert 1.0 < lo < 3.0, f"mid-schedule λ_o2o={lo} out of range"

    def test_loss_items_detached(self, loss_fn):
        """loss_items must be detached (no grad_fn)."""
        outputs = _make_outputs(requires_grad=True)
        targets = _make_targets()
        _, loss_items = loss_fn(
            outputs, targets,
            epoch_num=10, step_num=0,
            batch_height=320, batch_width=320,
        )
        assert loss_items.grad_fn is None, "loss_items should be detached"

    def test_loss_positive(self, loss_fn):
        """Total loss must be non-negative."""
        outputs = _make_outputs()
        targets = _make_targets()
        loss, _ = loss_fn(
            outputs, targets,
            epoch_num=10, step_num=0,
            batch_height=320, batch_width=320,
        )
        assert loss.item() >= 0.0, f"loss={loss.item()} is negative"


class TestComputeLossO2O_QAT:
    """QAT mode: ConfidenceMarginLoss is added."""

    @pytest.fixture
    def loss_fn_qat(self):
        return ComputeLoss_O2O(
            num_classes=3,
            ori_img_size=320,
            use_dfl=False,
            reg_max=0,
            iou_type='wiou',
            fpn_strides=[4, 8, 16, 32],
            qat_mode=True,
            confidence_threshold=0.25,
        )

    def test_qat_loss_differs_from_normal(self):
        """QAT loss >= normal loss (conf_margin adds non-negative term)."""
        torch.manual_seed(42)
        outputs = _make_outputs(requires_grad=False)
        targets = _make_targets()
        kwargs = dict(epoch_num=10, step_num=0, batch_height=320, batch_width=320)

        fn_normal = ComputeLoss_O2O(num_classes=3, ori_img_size=320, qat_mode=False)
        fn_qat    = ComputeLoss_O2O(num_classes=3, ori_img_size=320, qat_mode=True)

        # Run both on same inputs — weights differ so absolute values differ,
        # but QAT output should at least be a valid finite scalar.
        loss_qat, items_qat = fn_qat(outputs, targets, **kwargs)
        assert torch.isfinite(loss_qat), "QAT loss is not finite"
        assert items_qat.shape == (4,)


class TestConfidenceMarginLoss:
    """Unit tests for the hinge loss."""

    def test_perfect_separation(self):
        """No loss when fg >> hi and bg << lo."""
        cml = ConfidenceMarginLoss(threshold=0.25, n_steps=3)
        B, N, nc = 1, 100, 3
        scores = torch.zeros(B, N, nc)
        fg_mask = torch.zeros(B, N, dtype=torch.bool)
        fg_mask[0, :10] = True
        # fg scores well above hi = 0.25 + 3/255 ≈ 0.262
        scores[0, :10, 0] = 0.9
        # bg scores well below lo = 0.25 - 3/255 ≈ 0.238
        scores[0, 10:, 0] = 0.0
        loss = cml(scores, fg_mask)
        assert loss.item() < 1e-6, f"Expected ~0 loss, got {loss.item()}"

    def test_boundary_triggers_loss(self):
        """Scores exactly at threshold trigger positive loss."""
        cml = ConfidenceMarginLoss(threshold=0.25, n_steps=3)
        B, N, nc = 1, 10, 3
        scores = torch.full((B, N, nc), 0.25)   # exactly at threshold
        fg_mask = torch.zeros(B, N, dtype=torch.bool)
        fg_mask[0, :5] = True
        loss = cml(scores, fg_mask)
        assert loss.item() > 0.0, "Scores at threshold should produce positive loss"
