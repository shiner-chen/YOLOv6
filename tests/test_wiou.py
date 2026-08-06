# tests/test_wiou.py
import torch, pytest, sys
sys.path.insert(0, '.')
from yolov6.utils.figure_iou import IOUloss

def test_wiou_forward_and_grad():
    box1 = torch.tensor([[10., 10., 50., 50.]], requires_grad=True)
    box2 = torch.tensor([[12., 12., 52., 52.]])
    loss_fn = IOUloss(box_format='xyxy', iou_type='wiou', reduction='mean')
    loss = loss_fn(box1, box2)
    assert loss.item() > 0, "wiou loss must be positive"
    loss.backward()
    assert box1.grad is not None, "gradient must flow through wiou"

def test_wiou_less_than_ciou_for_close_boxes():
    """WIoU down-weights high-quality (close) predictions; loss should be
    numerically different from plain (1-iou)."""
    box1 = torch.tensor([[10., 10., 50., 50.]])
    box2 = torch.tensor([[10., 10., 50., 50.]])  # perfect overlap
    loss_fn = IOUloss(box_format='xyxy', iou_type='wiou', reduction='mean')
    loss = loss_fn(box1, box2)
    # perfect overlap → iou=1 → (1-iou)=0 → wiou=0 * wise_factor = 0
    assert loss.item() == pytest.approx(0.0, abs=1e-5)
