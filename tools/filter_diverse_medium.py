#!/usr/bin/env python3
"""
Filter Diverse dataset to keep only medium-sized targets (40-120px).
Remove large targets (>120px) to prevent bias toward large objects.
"""

import os
import shutil
from pathlib import Path

def analyze_and_filter(img_dir, label_dir, out_img_dir, out_label_dir,
                       min_size=40, max_size=120, target_img_size=320):
    """
    Filter images based on target size.
    Only keep images where ALL targets are within [min_size, max_size] range.
    """
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    img_files = list(Path(img_dir).glob('*.jpg')) + list(Path(img_dir).glob('*.png'))

    kept = 0
    discarded = 0
    stats = {'too_small': 0, 'in_range': 0, 'too_large': 0}

    print(f"Analyzing {len(img_files)} images...")

    for img_path in img_files:
        label_path = Path(label_dir) / (img_path.stem + '.txt')

        if not label_path.exists():
            discarded += 1
            continue

        # Check all targets in this image
        targets_in_range = True
        target_sizes = []

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    w, h = float(parts[3]), float(parts[4])
                    w_px = w * target_img_size
                    h_px = h * target_img_size
                    size = (w_px * h_px) ** 0.5  # Geometric mean
                    target_sizes.append(size)

                    if size < min_size:
                        stats['too_small'] += 1
                        targets_in_range = False
                    elif size > max_size:
                        stats['too_large'] += 1
                        targets_in_range = False
                    else:
                        stats['in_range'] += 1

        # Keep image only if ALL targets are in range
        if targets_in_range and target_sizes:
            shutil.copy2(img_path, out_img_dir)
            shutil.copy2(label_path, out_label_dir)
            kept += 1
        else:
            discarded += 1

    return kept, discarded, stats


def main():
    src_root = '/home/adlink/data/ARD100_roi320'
    out_root = '/home/adlink/data/ARD100_roi320'

    min_size = 40   # Minimum target size (px)
    max_size = 120  # Maximum target size (px)

    print("=" * 70)
    print("Filtering Diverse Dataset - Keep Medium Targets (40-120px)")
    print("=" * 70)
    print(f"Size range: {min_size}-{max_size} px (geometric mean)")
    print(f"Strategy: Keep images where ALL targets are in range")

    # Backup original diverse directory
    print("\n[1/3] Backing up original diverse data...")
    for split in ['train', 'val']:
        diverse_img_dir = Path(src_root) / 'images' / split / 'diverse'
        diverse_label_dir = Path(src_root) / 'labels' / split / 'diverse'
        backup_img_dir = Path(src_root) / 'images' / split / 'diverse_original'
        backup_label_dir = Path(src_root) / 'labels' / split / 'diverse_original'

        if backup_img_dir.exists():
            print(f"  {split}: images backup already exists, skipping")
        else:
            shutil.copytree(diverse_img_dir, backup_img_dir)
            print(f"  {split}: backed up images to diverse_original/")

        if backup_label_dir.exists():
            print(f"  {split}: labels backup already exists, skipping")
        else:
            shutil.copytree(diverse_label_dir, backup_label_dir)
            print(f"  {split}: backed up labels to diverse_original/")

    # Create filtered directories
    print("\n[2/3] Filtering training set...")
    kept_train, disc_train, stats_train = analyze_and_filter(
        img_dir=os.path.join(src_root, 'images/train/diverse_original'),
        label_dir=os.path.join(src_root, 'labels/train/diverse_original'),
        out_img_dir=os.path.join(src_root, 'images/train/diverse_filtered'),
        out_label_dir=os.path.join(src_root, 'labels/train/diverse_filtered'),
        min_size=min_size,
        max_size=max_size
    )

    print("\n[3/3] Filtering validation set...")
    kept_val, disc_val, stats_val = analyze_and_filter(
        img_dir=os.path.join(src_root, 'images/val/diverse_original'),
        label_dir=os.path.join(src_root, 'labels/val/diverse_original'),
        out_img_dir=os.path.join(src_root, 'images/val/diverse_filtered'),
        out_label_dir=os.path.join(src_root, 'labels/val/diverse_filtered'),
        min_size=min_size,
        max_size=max_size
    )

    print("\n" + "=" * 70)
    print("Filtering Results")
    print("=" * 70)
    print(f"\nTraining set:")
    print(f"  Original:  1,797 images")
    print(f"  Kept:      {kept_train:4d} images ({kept_train/1797*100:.1f}%)")
    print(f"  Discarded: {disc_train:4d} images")
    print(f"  Target distribution:")
    print(f"    Too small (<{min_size}px): {stats_train['too_small']}")
    print(f"    In range ({min_size}-{max_size}px): {stats_train['in_range']}")
    print(f"    Too large (>{max_size}px): {stats_train['too_large']}")

    print(f"\nValidation set:")
    print(f"  Original:  316 images")
    print(f"  Kept:      {kept_val:4d} images ({kept_val/316*100:.1f}%)")
    print(f"  Discarded: {disc_val:4d} images")
    print(f"  Target distribution:")
    print(f"    Too small (<{min_size}px): {stats_val['too_small']}")
    print(f"    In range ({min_size}-{max_size}px): {stats_val['in_range']}")
    print(f"    Too large (>{max_size}px): {stats_val['too_large']}")

    print(f"\n" + "=" * 70)
    print("Next Steps")
    print("=" * 70)
    print("1. Review filtered data:")
    print(f"   ls {src_root}/images/train/diverse_filtered | head")
    print("2. If satisfied, replace diverse/ with diverse_filtered/")
    print("3. Create training config with NWD=20 (covers 10-120px)")
    print("4. Train with 2-stage approach")


if __name__ == '__main__':
    main()
