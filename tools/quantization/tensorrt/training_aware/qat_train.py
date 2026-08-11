#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""QAT fine-tuning script for ET-YOLOv6n O2O.

Mixed-precision INT8 strategy:
  - backbone + neck + head stem/cls_conv/reg_conv/reg_pred → INT8 (pytorch-quantization)
  - cls_pred_o2o layers → FP16 (quantization disabled to preserve confidence scores
    near the INT8 rounding boundary; guarded by ConfidenceMarginLoss during training)

Usage:
    python tools/quantization/tensorrt/training_aware/qat_train.py \\
        --weights runs/train/exp/weights/best_ckpt.pt \\
        --conf-file configs/et_yolov6n_o2o.py \\
        --data-path data/antiuav.yaml \\
        --img-size 320 \\
        --batch-size 16 \\
        --epochs 10 \\
        --device 0
"""

import argparse
import os
import sys
import datetime
import os.path as osp
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.core.engine import Trainer
from yolov6.models.yolo import build_model
from yolov6.utils.config import Config
from yolov6.utils.checkpoint import load_state_dict
from yolov6.utils.envs import get_envs, select_device, set_random_seed
from yolov6.utils.events import LOGGER, save_yaml
from yolov6.utils.general import increment_name, check_img_size


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

def init_quantization():
    """Initialize pytorch-quantization with TensorRT-compatible INT8 config."""
    try:
        from pytorch_quantization import quant_modules
        from pytorch_quantization import nn as quant_nn
        from pytorch_quantization.tensor_quant import QuantDescriptor

        # Use per-tensor symmetric INT8 (TRT default)
        quant_desc = QuantDescriptor(num_bits=8, calib_method='histogram')
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc)
        quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc)

        quant_modules.initialize()
        LOGGER.info('pytorch-quantization initialized — all Conv2d → QuantConv2d (INT8)')
        return True
    except ImportError:
        LOGGER.warning(
            'pytorch-quantization not found. Install with:\n'
            '  pip install pytorch-quantization --extra-index-url '
            'https://pypi.ngc.nvidia.com\n'
            'Running in float-only mode (ConfidenceMarginLoss still active).'
        )
        return False


def disable_cls_o2o_quantization(model):
    """Disable INT8 quantization on cls_pred_o2o layers (keep FP16).

    These layers sit closest to the confidence threshold decision boundary.
    Keeping them FP16 avoids discretization rounding errors that could flip
    foreground/background classification at INT8 precision.
    """
    from yolov6.utils.ema import de_parallel
    detect = de_parallel(model).detect

    disabled = 0
    for conv in detect.cls_preds_o2o:
        for module in conv.modules():
            if hasattr(module, '_input_quantizer'):
                module._input_quantizer.disable()
                module._weight_quantizer.disable()
                disabled += 1

    LOGGER.info(
        f'INT8 quantization disabled on {disabled} quantizer(s) in '
        f'cls_pred_o2o (these layers remain FP16)'
    )
    return model


def calibrate_quantizers(model, dataloader, device, num_batches=200,
                         amax_method='entropy', amax_percentile=99.99):
    """Run PTQ histogram calibration before QAT fine-tuning.

    Calibration gives quantizers a good initial scale before gradient
    updates begin; without it QAT can diverge in early epochs.

    Args:
        amax_method: passed to HistogramCalibrator.compute_amax() as 'method'
                     ('entropy', 'percentile', or 'mse').
        amax_percentile: used when amax_method='percentile'.
    """
    try:
        from pytorch_quantization import nn as quant_nn
        from pytorch_quantization.calib import HistogramCalibrator
    except ImportError:
        LOGGER.warning('pytorch-quantization not available — skipping calibration')
        return

    LOGGER.info(f'Calibrating quantizers on {num_batches} batches (method={amax_method})...')

    # Enable calibration mode
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.enable_calib()
            module.disable_quant()

    model.eval()
    with torch.no_grad():
        for i, (images, *_) in enumerate(dataloader):
            if i >= num_batches:
                break
            images = images.to(device, non_blocking=True).float() / 255.0
            model(images)

    # Load calibrated scales back; HistogramCalibrator.compute_amax requires
    # method as a positional arg — MaxCalibrator takes no extra args.
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, HistogramCalibrator):
                    module.load_calib_amax(amax_method, percentile=amax_percentile, strict=False)
                else:
                    module.load_calib_amax(strict=False)
            module.enable_quant()
            module.disable_calib()

    LOGGER.info('Calibration complete.')


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description='QAT fine-tuning for ET-YOLOv6n O2O', add_help=add_help
    )
    parser.add_argument('--weights',    required=True, type=str,
                        help='path to pretrained O2O checkpoint (.pt)')
    parser.add_argument('--conf-file',  default='./configs/et_yolov6n_o2o.py', type=str)
    parser.add_argument('--data-path',  default='./data/antiuav.yaml', type=str)
    parser.add_argument('--img-size',   default=320, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs',     default=10, type=int,
                        help='QAT fine-tune epochs (10–20 recommended)')
    parser.add_argument('--device',     default='0', type=str)
    parser.add_argument('--workers',    default=4, type=int)
    parser.add_argument('--output-dir', default='./runs/qat', type=str)
    parser.add_argument('--name',       default='et_yolov6n_o2o_qat', type=str)
    parser.add_argument('--calib-batches', default=200, type=int,
                        help='number of batches for PTQ histogram calibration')
    parser.add_argument('--skip-calib', action='store_true',
                        help='skip PTQ calibration (use if quantizers already calibrated)')
    parser.add_argument('--eval-interval',          default=5, type=int)
    parser.add_argument('--eval-final-only',        action='store_true')
    parser.add_argument('--heavy-eval-range',       default=5,  type=int)
    parser.add_argument('--stop_aug_last_n_epoch',  default=5,  type=int)
    parser.add_argument('--save_ckpt_on_last_n_epoch', default=3, type=int)
    parser.add_argument('--write_trainbatch_tb',    action='store_true')
    parser.add_argument('--check-images', action='store_true')
    parser.add_argument('--check-labels', action='store_true')
    parser.add_argument('--specific-shape', action='store_true')
    parser.add_argument('--height',     type=int, default=None)
    parser.add_argument('--width',      type=int, default=None)
    parser.add_argument('--cache-ram',  action='store_true')
    parser.add_argument('--rect',       action='store_true')
    parser.add_argument('--dist_url',   default='env://', type=str)
    parser.add_argument('--gpu_count',  type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--bs_per_gpu', default=16, type=int)
    # these are required by Trainer but unused in QAT mode
    parser.add_argument('--resume',     nargs='?', const=False, default=False)
    parser.add_argument('--distill',    action='store_true', default=False)
    parser.add_argument('--distill_feat', action='store_true', default=False)
    parser.add_argument('--quant',      action='store_true', default=True)
    parser.add_argument('--calib',      action='store_true', default=False)
    parser.add_argument('--teacher_model_path', type=str, default=None)
    parser.add_argument('--temperature', type=int, default=20)
    parser.add_argument('--fuse_ab',    action='store_true', default=False)
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    args.local_rank, args.rank, args.world_size = get_envs()
    master_process = args.rank in [-1, 0]

    # Output directory
    args.save_dir = str(increment_name(osp.join(args.output_dir, args.name)))
    if master_process:
        os.makedirs(args.save_dir, exist_ok=True)

    # Image size
    if args.specific_shape:
        args.height = check_img_size(args.height, 32, floor=256)
        args.width  = check_img_size(args.width,  32, floor=256)
    else:
        args.img_size = check_img_size(args.img_size, 32, floor=256)

    device = select_device(args.device)
    set_random_seed(1 + args.rank, deterministic=(args.rank == -1))

    if master_process:
        save_yaml(vars(args), osp.join(args.save_dir, 'args.yaml'))

    # ---- Load config and enable QAT mode ----
    cfg = Config.fromfile(args.conf_file)
    if not hasattr(cfg, 'training_mode'):
        setattr(cfg, 'training_mode', 'repvgg')

    # Force qat_mode=True so ConfidenceMarginLoss is active in ComputeLoss_O2O
    cfg.model.head.qat_mode = True
    LOGGER.info('QAT mode enabled: ConfidenceMarginLoss active on O2O cls head')

    # ---- Initialize quantization before model creation ----
    quant_available = init_quantization()

    # ---- Build model via Trainer (handles distributed setup, EMA, etc.) ----
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        LOGGER.info('Initializing process group...')
        dist.init_process_group(
            backend='nccl' if dist.is_nccl_available() else 'gloo',
            init_method=args.dist_url,
            rank=args.local_rank,
            world_size=args.world_size,
            timeout=datetime.timedelta(seconds=7200),
        )

    trainer = Trainer(args, cfg, device)

    # ---- Load pretrained O2O weights ----
    LOGGER.info(f'Loading pretrained O2O weights from: {args.weights}')
    trainer.model = load_state_dict(args.weights, trainer.model, map_location=device)

    # ---- Mixed-precision: disable INT8 on cls_pred_o2o ----
    if quant_available:
        trainer.model = disable_cls_o2o_quantization(trainer.model)

        # PTQ calibration pass (establishes good initial quantizer scales)
        if not args.skip_calib:
            calibrate_quantizers(
                trainer.model,
                trainer.train_loader,
                device,
                num_batches=args.calib_batches,
                amax_method=getattr(cfg.ptq, 'histogram_amax_method', 'entropy'),
                amax_percentile=getattr(cfg.ptq, 'histogram_amax_percentile', 99.99),
            )
            # Move amax tensors to GPU after calibration
            trainer.model.to(device)

    # ---- QAT fine-tuning ----
    LOGGER.info(
        f'Starting QAT fine-tuning: {args.epochs} epochs, '
        f'batch={args.batch_size}, img={args.img_size}'
    )
    trainer.train()

    if args.world_size > 1 and args.rank == 0:
        LOGGER.info('Destroying process group...')
        dist.destroy_process_group()


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
