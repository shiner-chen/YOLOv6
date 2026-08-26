#!/usr/bin/env python3
"""
Integrate diverse drone dataset into ARD100 ROI320 dataset.
- Letterbox resize diverse images from various sizes to 320×320
- Copy corresponding labels and adjust coordinates
- Create train/diverse and val/diverse subdirectories
"""

import os
import cv2
import shutil
from pathlib import Path
import numpy as np


def letterbox_resize(img, target_size=320):
    """Letterbox resize: maintain aspect ratio, pad to square"""
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create canvas and paste
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized

    return canvas, scale, top, left


def adjust_label(label_path, scale, pad_top, pad_left, orig_h, orig_w, target_size=320):
    """Adjust YOLO label coordinates after letterbox resize"""
    if not os.path.exists(label_path):
        return None

    adjusted_lines = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls = parts[0]
            x_center, y_center, width, height = map(float, parts[1:5])

            # Convert from normalized to absolute coordinates (original image)
            x_abs = x_center * orig_w
            y_abs = y_center * orig_h
            w_abs = width * orig_w
            h_abs = height * orig_h

            # Apply scale
            x_abs_scaled = x_abs * scale
            y_abs_scaled = y_abs * scale
            w_abs_scaled = w_abs * scale
            h_abs_scaled = h_abs * scale

            # Apply padding
            x_abs_final = x_abs_scaled + pad_left
            y_abs_final = y_abs_scaled + pad_top

            # Convert back to normalized (320×320)
            x_norm = x_abs_final / target_size
            y_norm = y_abs_final / target_size
            w_norm = w_abs_scaled / target_size
            h_norm = h_abs_scaled / target_size

            # Clip to [0, 1]
            x_norm = np.clip(x_norm, 0, 1)
            y_norm = np.clip(y_norm, 0, 1)
            w_norm = np.clip(w_norm, 0, 1)
            h_norm = np.clip(h_norm, 0, 1)

            adjusted_lines.append(f"{cls} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    return adjusted_lines


def process_split(src_img_dir, src_label_dir, dst_img_dir, dst_label_dir, target_size=320):
    """Process one split (train/val)"""
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)

    img_files = list(Path(src_img_dir).glob('*.jpg')) + list(Path(src_img_dir).glob('*.png'))

    print(f"Processing {len(img_files)} images...")
    for idx, img_path in enumerate(img_files, 1):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(img_files)}")

        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Failed to read {img_path}")
            continue

        orig_h, orig_w = img.shape[:2]

        # Letterbox resize
        resized_img, scale, pad_top, pad_left = letterbox_resize(img, target_size)

        # Save resized image
        dst_img_path = os.path.join(dst_img_dir, img_path.name)
        cv2.imwrite(dst_img_path, resized_img)

        # Process label
        label_path = os.path.join(src_label_dir, img_path.stem + '.txt')
        if os.path.exists(label_path):
            adjusted_lines = adjust_label(label_path, scale, pad_top, pad_left,
                                         orig_h, orig_w, target_size)
            if adjusted_lines:
                dst_label_path = os.path.join(dst_label_dir, img_path.stem + '.txt')
                with open(dst_label_path, 'w') as f:
                    f.writelines(adjusted_lines)


def main():
    # Source: diverse dataset
    src_root = '/data-robot/DroneDataSet/cropped_images'

    # Destination: ROI320 dataset
    dst_root = '/home/adlink/data/ARD100_roi320'

    target_size = 320

    print("=" * 60)
    print("Integrating Diverse Dataset into ARD100 ROI320")
    print("=" * 60)

    # Process train split
    print("\n[1/2] Processing training set...")
    process_split(
        src_img_dir=os.path.join(src_root, 'train/images'),
        src_label_dir=os.path.join(src_root, 'train/labels'),
        dst_img_dir=os.path.join(dst_root, 'images/train/diverse'),
        dst_label_dir=os.path.join(dst_root, 'labels/train/diverse'),
        target_size=target_size
    )

    # Process val split
    print("\n[2/2] Processing validation set...")
    process_split(
        src_img_dir=os.path.join(src_root, 'val/images'),
        src_label_dir=os.path.join(src_root, 'val/labels'),
        dst_img_dir=os.path.join(dst_root, 'images/val/diverse'),
        dst_label_dir=os.path.join(dst_root, 'labels/val/diverse'),
        target_size=target_size
    )

    # Count results
    train_diverse_imgs = len(list(Path(dst_root, 'images/train/diverse').glob('*.jpg')))
    val_diverse_imgs = len(list(Path(dst_root, 'images/val/diverse').glob('*.jpg')))

    print("\n" + "=" * 60)
    print("Integration Complete!")
    print("=" * 60)
    print(f"Train diverse images: {train_diverse_imgs}")
    print(f"Val diverse images: {val_diverse_imgs}")
    print(f"\nNext steps:")
    print(f"1. Verify: ls {dst_root}/images/train/diverse | head -5")
    print(f"2. Create merged config: data/ard100_roi320_merged.yaml")
    print(f"3. Start training with merged dataset")


if __name__ == '__main__':
    main()
