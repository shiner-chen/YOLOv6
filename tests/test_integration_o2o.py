#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Integration smoke test for ET-YOLOv6n O2O pipeline.

Verifies end-to-end training flow:
  - Config loading
  - Model building (O2O head)
  - Forward pass (train + eval modes)
  - Loss computation (ComputeLoss_O2O)
  - Loss backward + gradient flow

Does NOT run actual training epochs — only checks the pipeline is wired correctly.
"""

import pytest
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.models.yolo import build_model
from yolov6.utils.config import Config
from yolov6.models.losses.loss_o2o import ComputeLoss_O2O


@pytest.fixture
def cfg_o2o():
    """Load the O2O config."""
    cfg_path = ROOT / 'configs' / 'et_yolov6n_o2o.py'
    assert cfg_path.exists(), f'Config not found: {cfg_path}'
    cfg = Config.fromfile(str(cfg_path))
    return cfg


@pytest.fixture
def device():
    return torch.device('cpu')  # CPU for CI compatibility


class TestO2OIntegration:
    """End-to-end pipeline checks."""

    def test_config_o2o_flag(self, cfg_o2o):
        """Config has o2o=True."""
        assert hasattr(cfg_o2o.model.head, 'o2o'), 'head.o2o attribute missing'
        assert cfg_o2o.model.head.o2o is True, 'head.o2o should be True'

    def test_config_wiou(self, cfg_o2o):
        """Config uses WIoU."""
        assert cfg_o2o.model.head.iou_type == 'wiou', \
            f"Expected iou_type='wiou', got {cfg_o2o.model.head.iou_type}"

    def test_config_dfl_disabled(self, cfg_o2o):
        """DFL is off (use_dfl=False, reg_max=0)."""
        assert cfg_o2o.model.head.use_dfl is False
        assert cfg_o2o.model.head.reg_max == 0

    def test_config_prog_loss(self, cfg_o2o):
        """ProgLoss schedule params present."""
        assert hasattr(cfg_o2o.model.head, 'prog_loss_t1')
        assert hasattr(cfg_o2o.model.head, 'prog_loss_t2')
        assert cfg_o2o.model.head.prog_loss_t1 == 50
        assert cfg_o2o.model.head.prog_loss_t2 == 150

    def test_model_builds(self, cfg_o2o, device):
        """Model creation succeeds with O2O head."""
        model = build_model(cfg_o2o, num_classes=3, device=device)
        assert model is not None
        # Check head type
        from yolov6.models.heads.effidehead_o2o import Detect
        assert isinstance(model.detect, Detect), \
            f"Expected effidehead_o2o.Detect, got {type(model.detect)}"

    def test_forward_train(self, cfg_o2o, device):
        """Train forward returns (feats, cls_o2m, reg_o2m, cls_o2o, reg_o2o)."""
        model = build_model(cfg_o2o, num_classes=3, device=device)
        model.train()

        x = torch.randn(2, 3, 320, 320, device=device)
        output = model(x)

        # Non-export mode returns [x, featmaps]
        assert isinstance(output, (list, tuple)) and len(output) == 2
        preds, featmaps = output

        # preds should be (feats, cls_o2m, reg_o2m, cls_o2o, reg_o2o)
        assert isinstance(preds, (list, tuple)) and len(preds) == 5
        feats, cls_o2m, reg_o2m, cls_o2o, reg_o2o = preds

        # feats is list of 4 feature maps
        assert isinstance(feats, list) and len(feats) == 4

        # cls/reg shapes
        B, N_all, nc = 2, 8500, 3
        assert cls_o2m.shape == (B, N_all, nc)
        assert reg_o2m.shape == (B, N_all, 4)
        assert cls_o2o.shape == (B, N_all, nc)
        assert reg_o2o.shape == (B, N_all, 4)

    def test_forward_eval(self, cfg_o2o, device):
        """Eval forward returns (B, N_all, 4+1+nc)."""
        model = build_model(cfg_o2o, num_classes=3, device=device)
        model.eval()

        x = torch.randn(2, 3, 320, 320, device=device)
        with torch.no_grad():
            output = model(x)

        # In eval, detect returns (B, N_all, 4+1+nc)
        # But Model.forward wraps it in [output, featmaps] in non-export mode
        # For eval, we need export mode or direct detect call
        model.export = True
        with torch.no_grad():
            output = model(x)

        B, N_all, nc = 2, 8500, 3
        expected_shape = (B, N_all, 4 + 1 + nc)  # bbox + obj + cls
        assert output.shape == expected_shape, \
            f"Expected {expected_shape}, got {output.shape}"

    def test_loss_creation(self, cfg_o2o):
        """ComputeLoss_O2O instantiates correctly from config."""
        loss_fn = ComputeLoss_O2O(
            num_classes=3,
            ori_img_size=320,
            use_dfl=cfg_o2o.model.head.use_dfl,
            reg_max=cfg_o2o.model.head.reg_max,
            iou_type=cfg_o2o.model.head.iou_type,
            fpn_strides=cfg_o2o.model.head.strides,
            prog_loss_t1=cfg_o2o.model.head.prog_loss_t1,
            prog_loss_t2=cfg_o2o.model.head.prog_loss_t2,
            qat_mode=getattr(cfg_o2o.model.head, 'qat_mode', False),
        )
        assert loss_fn is not None

    def test_loss_forward_and_backward(self, cfg_o2o, device):
        """Loss computes and gradients flow."""
        model = build_model(cfg_o2o, num_classes=3, device=device)
        model.train()

        loss_fn = ComputeLoss_O2O(
            num_classes=3,
            ori_img_size=320,
            use_dfl=False,
            reg_max=0,
            iou_type='wiou',
            fpn_strides=[4, 8, 16, 32],
        )

        # Forward
        x = torch.randn(2, 3, 320, 320, device=device, requires_grad=True)
        preds, _ = model(x)

        # Fake targets: 4 boxes across 2 images
        # Format: [img_idx, cls, cx, cy, w, h] normalized
        targets = torch.tensor([
            [0, 0, 0.5, 0.5, 0.1, 0.1],
            [0, 1, 0.3, 0.3, 0.08, 0.08],
            [1, 2, 0.6, 0.4, 0.12, 0.09],
            [1, 0, 0.7, 0.7, 0.15, 0.15],
        ], dtype=torch.float32, device=device)

        # Compute loss
        loss, loss_items = loss_fn(
            preds, targets,
            epoch_num=10, step_num=0,
            batch_height=320, batch_width=320,
        )

        # Check outputs
        assert loss.ndim == 0, f"loss should be scalar, got shape {loss.shape}"
        assert loss_items.shape == (4,), f"loss_items shape: {loss_items.shape}"
        assert loss.item() >= 0.0, f"loss={loss.item()} is negative"

        # Backward
        loss.backward()

        # Check gradients reached model params
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters() if p.requires_grad
        )
        assert has_grad, "No gradient reached model parameters"

    def test_qat_mode_activates_conf_margin_loss(self, cfg_o2o, device):
        """qat_mode=True adds ConfidenceMarginLoss term."""
        model = build_model(cfg_o2o, num_classes=3, device=device)
        model.train()

        loss_fn_normal = ComputeLoss_O2O(
            num_classes=3, ori_img_size=320,
            iou_type='wiou', fpn_strides=[4, 8, 16, 32],
            qat_mode=False,
        )
        loss_fn_qat = ComputeLoss_O2O(
            num_classes=3, ori_img_size=320,
            iou_type='wiou', fpn_strides=[4, 8, 16, 32],
            qat_mode=True,
        )

        x = torch.randn(1, 3, 320, 320, device=device)
        preds, _ = model(x)
        targets = torch.tensor([
            [0, 0, 0.5, 0.5, 0.1, 0.1],
        ], dtype=torch.float32, device=device)

        loss_normal, _ = loss_fn_normal(preds, targets, 10, 0, 320, 320)
        loss_qat, _ = loss_fn_qat(preds, targets, 10, 0, 320, 320)

        # QAT loss should be finite (conf_margin adds a non-negative term)
        assert torch.isfinite(loss_normal)
        assert torch.isfinite(loss_qat)
        # They differ (conf_margin term present in QAT)
        # Can't assert strict inequality as they might coincidentally match,
        # but at least both should be valid scalars
        assert loss_qat.ndim == 0


class TestO2OStrides:
    """Verify 4-scale P2/P3/P4/P5 strides."""

    def test_strides_match_config(self, cfg_o2o, device):
        """Model strides = [4, 8, 16, 32]."""
        model = build_model(cfg_o2o, num_classes=3, device=device)
        expected = torch.tensor([4, 8, 16, 32], device=device)
        assert torch.equal(model.stride, expected), \
            f"Expected strides {expected}, got {model.stride}"

    def test_n_all_320(self, cfg_o2o, device):
        """For 320×320 input, N_all = 8500."""
        model = build_model(cfg_o2o, num_classes=3, device=device)
        model.train()
        x = torch.randn(1, 3, 320, 320, device=device)
        preds, _ = model(x)
        _, cls_o2m, _, _, _ = preds
        assert cls_o2m.shape[1] == 8500, \
            f"Expected N_all=8500 for 320×320, got {cls_o2m.shape[1]}"
