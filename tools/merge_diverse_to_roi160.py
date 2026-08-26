#!/usr/bin/env python3
"""
merge_diverse_to_roi160.py — Merge diverse drone dataset into ARD100_roi160 using subdirectories.

Strategy:
---------
1. Use subdirectory structure to keep datasets separate:
   images/train/ard100/  — original ROI160 images
   images/train/diverse/ — diversity dataset images (letterbox resized to 160x160)

2. Letterbox resize: maintains aspect ratio, adds gray padding to reach 160x160

3. Split diverse dataset:
   - train: put into train/diverse/
   - val: put into val/diverse/
   - No test split for diverse dataset (ARD100 test remains pure)

Output structure:
-----------------
  ARD100_roi160/
    images/
      train/
        ard100/     # original roi160 images (symlinks)
        diverse/    # diversity dataset (letterbox resized)
      val/
        ard100/
        diverse/
      test/
        ard100/     # test remains pure ARD100
    labels/
      train/
        ard100/
        diverse/
      val/
        ard100/
        diverse/
      test/
        ard100/
    ard100_roi160_merged.yaml  # updated config

Usage:
------
  python3 tools/merge_diverse_to_roi160.py \
      --roi160-dir /home/adlink/data/ARD100_roi160 \
      --diverse-dir /data-robot/DroneDataSet/cropped_images \
      --output-size 160
"""

from __future__ import annotations

import argparse
import os
import shutil
import yaml
from pathlib import Path
from typing import Tuple
from PIL import Image
import numpy as np


def letterbox_resize(
    img: Image.Image,
    target_size: int,
    fill_color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[Image.Image, float, Tuple[int, int]]:
    """
    Resize image to target_size×target_size using letterbox (maintain aspect ratio).

    Returns:
        resized_img: The letterbox resized image
        scale: The scaling factor applied
        pad: (pad_w, pad_h) padding added on each side
    """
    img_w, img_h = img.size

    # Calculate scale to fit the target size
    scale = min(target_size / img_w, target_size / img_h)

    # New dimensions after scaling
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    # Resize image
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Create canvas with fill color
    canvas = Image.new('RGB', (target_size, target_size), fill_color)

    # Calculate padding to center the image
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2

    # Paste resized image onto canvas
    canvas.paste(resized, (pad_w, pad_h))

    return canvas, scale, (pad_w, pad_h)


def convert_bbox_letterbox(
    bbox_yolo: list,
    original_size: Tuple[int, int],
    scale: float,
    pad: Tuple[int, int],
    target_size: int
) -> list:
    """
    Convert YOLO bbox coordinates after letterbox resize.

    Args:
        bbox_yolo: [class_id, cx, cy, w, h] in relative coords (0-1)
        original_size: (orig_w, orig_h)
        scale: scaling factor
        pad: (pad_w, pad_h)
        target_size: target image size

    Returns:
        [class_id, new_cx, new_cy, new_w, new_h] in relative coords
    """
    class_id = bbox_yolo[0]
    cx, cy, w, h = bbox_yolo[1:]

    orig_w, orig_h = original_size
    pad_w, pad_h = pad

    # Convert from relative to absolute in original image
    cx_abs = cx * orig_w
    cy_abs = cy * orig_h
    w_abs = w * orig_w
    h_abs = h * orig_h

    # Apply scaling and padding
    new_cx_abs = cx_abs * scale + pad_w
    new_cy_abs = cy_abs * scale + pad_h
    new_w_abs = w_abs * scale
    new_h_abs = h_abs * scale

    # Convert back to relative coordinates
    new_cx = new_cx_abs / target_size
    new_cy = new_cy_abs / target_size
    new_w = new_w_abs / target_size
    new_h = new_h_abs / target_size

    return [class_id, new_cx, new_cy, new_w, new_h]


def process_diverse_dataset(
    diverse_dir: str,
    roi160_dir: str,
    target_size: int = 160,
) -> dict:
    """
    Process diverse dataset and merge into roi160 directory structure.

    Returns statistics dict.
    """
    diverse_path = Path(diverse_dir)
    roi160_path = Path(roi160_dir)

    stats = {'train': 0, 'val': 0}

    for split in ['train', 'val']:
        src_img_dir = diverse_path / split / 'images'
        src_lbl_dir = diverse_path / split / 'labels'

        if not src_img_dir.exists():
            print(f"[WARN] {src_img_dir} does not exist, skipping {split}")
            continue

        # Create destination directories
        dst_img_dir = roi160_path / 'images' / split / 'diverse'
        dst_lbl_dir = roi160_path / 'labels' / split / 'diverse'
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        # Process each image
        img_files = sorted(src_img_dir.glob('*.jpg'))
        print(f'\nProcessing {split}: {len(img_files)} images')

        for img_file in img_files:
            stem = img_file.stem
            lbl_file = src_lbl_dir / f'{stem}.txt'

            if not lbl_file.exists():
                print(f"  [WARN] Label not found for {img_file.name}, skipping")
                continue

            try:
                # Read and resize image
                img = Image.open(img_file).convert('RGB')
                orig_size = img.size  # (w, h)

                # Letterbox resize
                resized_img, scale, pad = letterbox_resize(img, target_size)

                # Save resized image with 'diverse_' prefix
                out_img_path = dst_img_dir / f'diverse_{stem}.jpg'
                resized_img.save(out_img_path, 'JPEG', quality=90)

                # Read and convert labels
                with open(lbl_file, 'r') as f:
                    lines = f.readlines()

                new_labels = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])

                    # Convert bbox coordinates
                    new_bbox = convert_bbox_letterbox(
                        [class_id, cx, cy, w, h],
                        orig_size,
                        scale,
                        pad,
                        target_size
                    )

                    new_labels.append(
                        f"{new_bbox[0]} {new_bbox[1]:.6f} {new_bbox[2]:.6f} "
                        f"{new_bbox[3]:.6f} {new_bbox[4]:.6f}"
                    )

                # Save converted labels
                out_lbl_path = dst_lbl_dir / f'diverse_{stem}.txt'
                with open(out_lbl_path, 'w') as f:
                    f.write('\n'.join(new_labels) + '\n' if new_labels else '')

                stats[split] += 1

                if stats[split] % 200 == 0:
                    print(f"  Processed {stats[split]} images...")

            except Exception as e:
                print(f"  [ERROR] Failed to process {img_file.name}: {e}")
                continue

        print(f"  {split}: {stats[split]} images processed")

    return stats


def create_subdirectory_structure(roi160_dir: str):
    """
    Reorganize original roi160 images into ard100 subdirectories using symlinks.
    This keeps the original files in place while creating the new structure.
    """
    roi160_path = Path(roi160_dir)

    print("\nReorganizing original ROI160 images into subdirectories...")

    for split in ['train', 'val', 'test']:
        img_dir = roi160_path / 'images' / split
        lbl_dir = roi160_path / 'labels' / split

        if not img_dir.exists():
            continue

        # Create ard100 subdirectories
        ard100_img_dir = img_dir / 'ard100'
        ard100_lbl_dir = lbl_dir / 'ard100'
        ard100_img_dir.mkdir(exist_ok=True)
        ard100_lbl_dir.mkdir(exist_ok=True)

        # Move original files to ard100 subdirectory
        img_files = [f for f in img_dir.iterdir() if f.is_file() and f.suffix == '.jpg']
        lbl_files = [f for f in lbl_dir.iterdir() if f.is_file() and f.suffix == '.txt']

        print(f"  {split}: moving {len(img_files)} images, {len(lbl_files)} labels")

        for img_file in img_files:
            shutil.move(str(img_file), str(ard100_img_dir / img_file.name))

        for lbl_file in lbl_files:
            shutil.move(str(lbl_file), str(ard100_lbl_dir / lbl_file.name))


def update_yaml(roi160_dir: str, stats: dict):
    """
    Create updated YAML config that references the new directory structure.
    """
    roi160_path = Path(roi160_dir)

    # Read original manifest for metadata
    manifest_path = roi160_path / 'split_manifest.json'
    import json
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    yaml_content = f"""# ARD100 160×160 ROI-crop + Diverse Drone Dataset (MERGED)
# Original ROI160: stride={manifest['stride']}  seed={manifest['seed']}
# ROI={manifest['roi_size']}  margin={manifest['roi_margin']}  vis_thresh={manifest['vis_thresh']}
# neg_per_pos={manifest['neg_per_pos']}  neg_no_overlap={manifest['neg_no_overlap']}
#
# Diverse dataset: letterbox resized to 160×160
# train diverse: {stats.get('train', 0)} images
# val diverse: {stats.get('val', 0)} images
# test: pure ARD100 (no diverse)

path: {roi160_dir}
train: images/train
val: images/val
test: images/test

is_coco: False
nc: 1
names: ['drone']
"""

    output_yaml = roi160_path / 'ard100_roi160_merged.yaml'
    with open(output_yaml, 'w') as f:
        f.write(yaml_content)

    print(f"\nUpdated YAML: {output_yaml}")


def main():
    ap = argparse.ArgumentParser(
        description='Merge diverse drone dataset into ARD100_roi160 using subdirectories')
    ap.add_argument('--roi160-dir', default='/home/adlink/data/ARD100_roi160',
                    help='Path to ARD100_roi160 dataset')
    ap.add_argument('--diverse-dir', default='/data-robot/DroneDataSet/cropped_images',
                    help='Path to diverse drone dataset')
    ap.add_argument('--output-size', type=int, default=160,
                    help='Target size for letterbox resize (default: 160)')
    ap.add_argument('--skip-reorganize', action='store_true',
                    help='Skip reorganizing original roi160 files (if already done)')
    args = ap.parse_args()

    print(f"Merging diverse dataset into ROI160...")
    print(f"  ROI160 dir: {args.roi160_dir}")
    print(f"  Diverse dir: {args.diverse_dir}")
    print(f"  Target size: {args.output_size}×{args.output_size}")

    # Step 1: Reorganize original roi160 files into subdirectories
    if not args.skip_reorganize:
        create_subdirectory_structure(args.roi160_dir)
    else:
        print("\nSkipping reorganization (--skip-reorganize flag)")

    # Step 2: Process and merge diverse dataset
    stats = process_diverse_dataset(
        args.diverse_dir,
        args.roi160_dir,
        args.output_size
    )

    # Step 3: Update YAML config
    update_yaml(args.roi160_dir, stats)

    print("\n=== MERGE COMPLETE ===")
    print(f"  Diverse train: {stats.get('train', 0)} images")
    print(f"  Diverse val: {stats.get('val', 0)} images")
    print(f"\nDirectory structure:")
    print(f"  {args.roi160_dir}/images/{{train,val,test}}/ard100/    — original ROI160")
    print(f"  {args.roi160_dir}/images/{{train,val}}/diverse/       — diverse dataset")
    print(f"\nUse: {args.roi160_dir}/ard100_roi160_merged.yaml for training")


if __name__ == '__main__':
    main()
