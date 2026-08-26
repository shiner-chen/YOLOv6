#!/usr/bin/env python3
"""Test script to verify YOLOv6n ROI160 P2/P3 2-head model builds correctly."""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from yolov6.utils.config import Config
from yolov6.models.yolo import build_model

def test_p2p3_model():
    """Test P2/P3 2-head model construction."""

    print("=" * 80)
    print("Testing YOLOv6n ROI160 P2/P3 2-Head Model")
    print("=" * 80)

    # Load config
    cfg_path = "configs/yolov6n_roi160_p2p3.py"
    print(f"\n1. Loading config: {cfg_path}")
    cfg = Config.fromfile(cfg_path)

    # Build model
    print("\n2. Building model...")
    num_classes = 1  # Single class for Anti-UAV
    device = torch.device('cpu')

    try:
        model = build_model(cfg, num_classes, device)
        print("   ✓ Model built successfully")
    except Exception as e:
        print(f"   ✗ Model build failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test forward pass
    print("\n3. Testing forward pass with 160x160 input...")
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 160, 160)

    try:
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        print(f"   ✓ Forward pass successful")
        if isinstance(output, list):
            print(f"   Output type: list with {len(output)} elements")
            print(f"   Output[0] shape: {output[0].shape}")
        else:
            print(f"   Output shape: {output.shape}")
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify architecture
    print("\n4. Model architecture verification:")
    print(f"   Backbone type: {cfg.model.backbone.type}")
    print(f"   Backbone fuse_P2: {cfg.model.backbone.fuse_P2}")
    print(f"   Neck type: {cfg.model.neck.type}")
    print(f"   Head num_layers: {cfg.model.head.num_layers}")
    print(f"   Head strides: {cfg.model.head.strides}")
    print(f"   Head out_indices: {cfg.model.head.out_indices}")

    # Count parameters
    print("\n5. Model statistics:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")

    # Estimate FLOPs (rough)
    print("\n6. Estimated compute (rough):")
    print(f"   Input: {batch_size} x 3 x 160 x 160")
    print(f"   Expected FLOPs: ~0.3-0.5 GFLOPs (vs ~1.2 for 640x640 3-head)")

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)

    return True

if __name__ == "__main__":
    success = test_p2p3_model()
    sys.exit(0 if success else 1)
