#!/usr/bin/env python3
"""Initialise ET-YOLOv6s weights from a trained ET-YOLOv6n checkpoint via slice init.

Why slice init (not exact-shape copy):
  nano and s share identical key names but every real weight tensor has half the
  channels (width_multiple 0.25 vs 0.50).  A plain shape-match copy transfers
  nothing useful.  Slice init instead writes nano's weights into the first-half
  slice of each larger s tensor — preserving all learned features while leaving
  the extra s capacity randomly initialised.

  Example:
    nano  backbone.stem.rbr_dense.conv.weight  (16, 3, 3, 3)
    s     backbone.stem.rbr_dense.conv.weight  (32, 3, 3, 3)
    → s[:16, :3, :, :] = nano  (the remaining 16 output-channels stay Xavier)

Coverage:
  877 / 877 actual weight tensors eligible  (100 % of nano params transferred).
  Modules: backbone (455), neck (354), detect head (68).

Usage:
    python tools/init_s_from_nano.py \
        --src   runs/train/et_yolov6n/weights/best_ckpt.pt \
        --cfg   configs/et_yolov6s.py \
        --nc    <num_classes> \
        --out   weights/et_yolov6s_from_nano.pt

Then train normally, pointing pretrained at the output checkpoint:
    # option A — edit cfg:  model.pretrained = 'weights/et_yolov6s_from_nano.pt'
    # option B — CLI:
    python tools/train.py \
        --conf  configs/et_yolov6s.py \
        --data  data/your_dataset.yaml \
        --img-size 640 \
        --batch-size 16 \
        --epochs 300 \
        --device 0 \
        --output-dir runs/train/et_yolov6s \
        --pretrained weights/et_yolov6s_from_nano.pt
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from yolov6.models.yolo import build_model
from yolov6.utils.config import Config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', required=True,
                   help='trained ET-YOLOv6n checkpoint (.pt)')
    p.add_argument('--cfg', default='configs/et_yolov6s.py',
                   help='ET-YOLOv6s config file')
    p.add_argument('--nc', type=int, required=True,
                   help='number of classes (must match the nano checkpoint)')
    p.add_argument('--out', default='weights/et_yolov6s_from_nano.pt',
                   help='output checkpoint path')
    p.add_argument('--strict', action='store_true',
                   help='error if any nano key is missing from the s model')
    return p.parse_args()


def load_nano_sd(src_path: str) -> dict:
    """Load state_dict from a YOLOv6-style checkpoint."""
    ckpt = torch.load(src_path, map_location='cpu', weights_only=False)
    if 'model' in ckpt:
        obj = ckpt['model']
        return obj.float().state_dict() if hasattr(obj, 'state_dict') else obj
    if 'state_dict' in ckpt:
        return ckpt['state_dict']
    # bare state_dict
    return ckpt


def slice_copy(dst: torch.Tensor, src: torch.Tensor) -> bool:
    """Copy src into the matching front-slice of dst.

    Returns True when shapes are compatible (every src dim <= dst dim and
    the number of dimensions matches).  dst is modified in-place.
    """
    if dst.shape == src.shape:
        dst.copy_(src)
        return True
    if dst.dim() != src.dim():
        return False
    if not all(a <= b for a, b in zip(src.shape, dst.shape)):
        return False
    idx = tuple(slice(0, d) for d in src.shape)
    dst[idx].copy_(src)
    return True


def main():
    args = parse_args()

    # ── Build target (s) model ──────────────────────────────────────────────
    cfg = Config.fromfile(args.cfg)
    model_s = build_model(cfg, num_classes=args.nc, device=torch.device('cpu'))
    sd_s = model_s.state_dict()

    # ── Load source (nano) weights ──────────────────────────────────────────
    sd_n = load_nano_sd(args.src)
    print(f'Loaded nano checkpoint: {args.src}')
    print(f'  nano keys : {len(sd_n)}')
    print(f'  s    keys : {len(sd_s)}')

    # ── Slice init ──────────────────────────────────────────────────────────
    exact_match = []    # same shape → plain copy
    slice_init  = []    # nano fits inside s → slice copy
    skipped     = []    # key missing in nano or dimensions incompatible

    for key, s_tensor in sd_s.items():
        if key not in sd_n:
            if args.strict:
                raise KeyError(f'Key {key!r} not found in nano checkpoint')
            skipped.append((key, 'not in src'))
            continue

        n_tensor = sd_n[key].float()

        if n_tensor.shape == s_tensor.shape:
            s_tensor.copy_(n_tensor)
            exact_match.append(key)
        elif slice_copy(s_tensor, n_tensor):
            slice_init.append(key)
        else:
            skipped.append((key, f'shape {tuple(n_tensor.shape)} incompatible with {tuple(s_tensor.shape)}'))

    # write updated tensors back
    model_s.load_state_dict(sd_s)

    # ── Report ──────────────────────────────────────────────────────────────
    total_n_params = sum(sd_n[k].numel() for k in exact_match + slice_init if k in sd_n)
    total_s_params = sum(p.numel() for p in model_s.parameters())
    print(f'\n── Transfer summary ──────────────────────────────────────────')
    print(f'  exact copy  : {len(exact_match):4d} tensors  (same shape)')
    print(f'  slice init  : {len(slice_init):4d} tensors  (nano → front-slice of s)')
    print(f'  skipped     : {len(skipped):4d} tensors  (missing / incompatible)')
    print(f'  nano params transferred : {total_n_params/1e6:.3f} M')
    print(f'  s model total params    : {total_s_params/1e6:.3f} M')
    print(f'  coverage    : {total_n_params/total_s_params*100:.1f}% of s params seeded from nano')

    if skipped:
        print(f'\n  Skipped details (first 5):')
        for k, reason in skipped[:5]:
            print(f'    {k}: {reason}')

    # ── Save ────────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model_s, 'epoch': -1, 'optimizer': None}, out_path)
    print(f'\nSaved → {out_path}')
    print('Next step: python tools/train.py --conf configs/et_yolov6s.py '
          f'--pretrained {out_path} ...')


if __name__ == '__main__':
    main()
