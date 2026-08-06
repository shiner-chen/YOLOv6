# tests/test_stal.py
import torch, sys
sys.path.insert(0, '.')


def test_stal_small_box_gets_higher_metric():
    """STAL should amplify align_metric by a larger factor for small GT boxes.

    Uses controlled predicted bboxes that closely match each GT box so that
    TAL metrics are non-trivial (IoU > 0.6), making the STAL boost factor
    measurable via a ratio comparison.
    """
    from yolov6.assigners.stal_assigner import STALAssigner
    from yolov6.assigners.tal_assigner import TaskAlignedAssigner

    B, N_gt, N_anc, nc = 1, 1, 10, 1
    torch.manual_seed(42)
    pd_scores = torch.rand(B, N_anc, nc)
    gt_labels = torch.zeros(B, N_gt, 1, dtype=torch.long)
    mask_gt = torch.ones(B, N_gt, 1)

    tal = TaskAlignedAssigner(topk=N_anc, num_classes=nc)
    stal = STALAssigner(topk=N_anc, num_classes=nc,
                        gamma=0.5, area_thr=0.02, ori_img_size=320)
    tal.bs = B; tal.n_max_boxes = N_gt
    stal.bs = B; stal.n_max_boxes = N_gt

    # ── Small GT box: 10×10 px ──────────────────────────────────────────────
    # area_ratio = 100 / 102400 ≈ 0.001 → small_factor ≈ 1.494
    gt_small = torch.tensor([[[155., 155., 165., 165.]]])
    # Near-perfect prediction: [154,154,166,166] → IoU ≈ 0.69
    pd_bboxes_s = torch.zeros(B, N_anc, 4)
    pd_bboxes_s[..., 0] = 154.; pd_bboxes_s[..., 1] = 154.
    pd_bboxes_s[..., 2] = 166.; pd_bboxes_s[..., 3] = 166.
    anc_pts_s = torch.full((N_anc, 2), 160.)  # all inside small GT

    _, tal_s, _ = tal.get_pos_mask(
        pd_scores, pd_bboxes_s, gt_labels, gt_small, anc_pts_s, mask_gt)
    _, stal_s, _ = stal.get_pos_mask(
        pd_scores, pd_bboxes_s, gt_labels, gt_small, anc_pts_s, mask_gt)

    # ── Large GT box: 200×200 px ─────────────────────────────────────────────
    # area_ratio = 40000 / 102400 ≈ 0.39 → small_factor ≈ 1.0
    gt_large = torch.tensor([[[60., 60., 260., 260.]]])
    # Near-perfect prediction: [59,59,261,261] → IoU ≈ 0.98
    pd_bboxes_l = torch.zeros(B, N_anc, 4)
    pd_bboxes_l[..., 0] = 59.; pd_bboxes_l[..., 1] = 59.
    pd_bboxes_l[..., 2] = 261.; pd_bboxes_l[..., 3] = 261.
    anc_pts_l = torch.full((N_anc, 2), 160.)  # all inside large GT

    _, tal_l, _ = tal.get_pos_mask(
        pd_scores, pd_bboxes_l, gt_labels, gt_large, anc_pts_l, mask_gt)
    _, stal_l, _ = stal.get_pos_mask(
        pd_scores, pd_bboxes_l, gt_labels, gt_large, anc_pts_l, mask_gt)

    # Both should have meaningful TAL metrics
    assert tal_s.max() > 1e-6, f"TAL small metric too low: {tal_s.max():.2e}"
    assert tal_l.max() > 1e-6, f"TAL large metric too low: {tal_l.max():.2e}"

    # Use 1e-12 eps so it does not pollute the ratio when metrics are ~0.01–0.7
    boost_small = (stal_s.max() / (tal_s.max() + 1e-12)).item()
    boost_large = (stal_l.max() / (tal_l.max() + 1e-12)).item()

    assert boost_small > boost_large, (
        f"Small box should get bigger boost than large: "
        f"small={boost_small:.4f}, large={boost_large:.4f}"
    )
    # Expected factors: small ≈ 1.494, large ≈ 1.0
    assert boost_small > 1.4, \
        f"Expected factor_small ≈ 1.494, got {boost_small:.4f}"
    assert boost_large < 1.01, \
        f"Expected factor_large ≈ 1.0, got {boost_large:.4f}"
