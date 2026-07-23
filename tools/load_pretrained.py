#!/usr/bin/env python3
"""Load YOLOv6n pretrained weights into ET-YOLOv6n (partial weight transfer).

Reusable layers (~40% of backbone):
  - backbone.stem                 (RepVGGBlock, identical)
  - backbone.ERBlock_2/3/4/5[0]   (stride-2 RepVGGBlock, identical)
  - backbone.ERBlock_5[2]         (SimCSPSPPF, identical)

Non-reusable (shape mismatch or new architecture):
  - backbone.ERBlock_2/3/4/5[1]   (C2fStar replaces RepBlock)
  - neck.*                         (CrossLayerBifusion replaces BiFusion)
  - head.*                         (4 scales vs 3)

Usage:
    python tools/load_pretrained.py \
        --src  weights/yolov6n.pt \
        --cfg  configs/et_yolov6n.py \
        --nc   1 \
        --out  weights/et_yolov6n_init.pt
"""

import argparse
import sys
from pathlib import Path

import torch

# make yolov6 importable
sys.path.insert(0, str(Path(__file__).parent.parent))
from yolov6.models.yolo import build_model
from yolov6.utils.config import Config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--src',  default='weights/yolov6n.pt',
                   help='path to YOLOv6n pretrained checkpoint')
    p.add_argument('--cfg',  default='configs/et_yolov6n.py',
                   help='ET-YOLOv6n config')
    p.add_argument('--nc',   type=int, default=1,
                   help='number of classes for the new model')
    p.add_argument('--out',  default='weights/et_yolov6n_init.pt',
                   help='output checkpoint path')
    return p.parse_args()


def load_src_weights(src_path):
    ckpt = torch.load(src_path, map_location='cpu')
    # YOLOv6 checkpoints store the state_dict under 'model' → .state_dict()
    if 'model' in ckpt:
        model_obj = ckpt['model']
        if hasattr(model_obj, 'state_dict'):
            return model_obj.state_dict()
        return model_obj   # already a state_dict
    if 'state_dict' in ckpt:
        return ckpt['state_dict']
    return ckpt


def main():
    args = parse_args()

    # Build the new model
    cfg = Config.fromfile(args.cfg)
    device = torch.device('cpu')
    model = build_model(cfg, num_classes=args.nc, device=device)
    new_sd = model.state_dict()

    # Load source weights
    src_sd = load_src_weights(args.src)

    matched = []
    skipped = []
    for key, tensor in new_sd.items():
        if key in src_sd and src_sd[key].shape == tensor.shape:
            new_sd[key] = src_sd[key]
            matched.append(key)
        else:
            skipped.append(key)

    model.load_state_dict(new_sd)

    # Report
    total = len(new_sd)
    print(f'\nWeight transfer: {len(matched)}/{total} layers matched '
          f'({100*len(matched)/total:.1f}%)')
    print(f'Matched  ({len(matched)}):')
    for k in matched[:20]:
        print(f'  {k}')
    if len(matched) > 20:
        print(f'  ... and {len(matched)-20} more')
    print(f'\nSkipped  ({len(skipped)}) — random init:')
    for k in skipped[:20]:
        print(f'  {k}')
    if len(skipped) > 20:
        print(f'  ... and {len(skipped)-20} more')

    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model, 'epoch': -1, 'optimizer': None}, args.out)
    print(f'\nSaved initialised checkpoint → {args.out}')


if __name__ == '__main__':
    main()
