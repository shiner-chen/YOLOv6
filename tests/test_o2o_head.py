# tests/test_o2o_head.py
import torch, sys
sys.path.insert(0, '.')


def test_o2o_head_training_mode():
    """Training returns 5-tuple (feats, cls_o2m, reg_o2m, cls_o2o, reg_o2o)."""
    from yolov6.models.heads.effidehead_o2o import Detect, build_effidehead_layer

    nc, nl = 1, 4
    channels_list = [32]*13
    head_layers = build_effidehead_layer(channels_list, nc, num_layers=nl)
    head = Detect(nc, num_layers=nl, head_layers=head_layers)

    feats = [torch.randn(2, 32, 80, 80),
             torch.randn(2, 32, 40, 40),
             torch.randn(2, 32, 20, 20),
             torch.randn(2, 32, 10, 10)]

    head.train()
    output = head(feats)
    assert len(output) == 5, f"Expected 5-tuple, got {len(output)}"
    feats_out, cls_o2m, reg_o2m, cls_o2o, reg_o2o = output
    N_all = 80*80 + 40*40 + 20*20 + 10*10
    assert cls_o2m.shape == (2, N_all, nc), f"cls_o2m: {cls_o2m.shape}"
    assert reg_o2m.shape == (2, N_all, 4),  f"reg_o2m: {reg_o2m.shape}"
    assert cls_o2o.shape == (2, N_all, nc), f"cls_o2o: {cls_o2o.shape}"
    assert reg_o2o.shape == (2, N_all, 4),  f"reg_o2o: {reg_o2o.shape}"


def test_o2o_head_inference_mode():
    """Inference returns (B, N_all, 4+1+nc) matching existing effidehead."""
    from yolov6.models.heads.effidehead_o2o import Detect, build_effidehead_layer

    nc, nl = 1, 4
    channels_list = [32]*13
    head_layers = build_effidehead_layer(channels_list, nc, num_layers=nl)
    head = Detect(nc, num_layers=nl, head_layers=head_layers)

    feats = [torch.randn(2, 32, 80, 80), torch.randn(2, 32, 40, 40),
             torch.randn(2, 32, 20, 20), torch.randn(2, 32, 10, 10)]

    head.eval()
    with torch.no_grad():
        output = head(feats)

    N_all = 80*80 + 40*40 + 20*20 + 10*10
    assert output.shape == (2, N_all, 4+1+nc), f"Got {output.shape}"
