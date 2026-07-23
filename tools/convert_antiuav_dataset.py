#!/usr/bin/env python3
"""Convert Anti-UAV-Tracking-V0 dataset (video tracking format) to YOLO detection format.

Input:
    /data-robot/DroneDataSet/Anti-UAV-Tracking-V0.zip
        Anti-UAV-Tracking-V0/videoXX/XXXXX.jpg

    /data-robot/DroneDataSet/Anti-UAV-Tracking-V0GT.zip
        Anti-UAV-Tracking-V0GT/videoXX_gt.txt  (one line per frame: x y w h, absolute pixels)
        Invisible frames have bbox = 0 0 0 0 — these are skipped.

Output (YOLO format):
    <out_dir>/images/{train,val}/videoXX_YYYYY.jpg
    <out_dir>/labels/{train,val}/videoXX_YYYYY.txt   (class xc yc w h, normalised)
    <out_dir>/antiuav.yaml

Split:  video01–16 → train,  video17–20 → val
Image resolution: 1920×1080 (fixed for this dataset)
"""

import argparse
import os
import zipfile
from pathlib import Path


IMG_W, IMG_H = 1920, 1080
TRAIN_VIDEOS = [f'video{i:02d}' for i in range(1, 17)]
VAL_VIDEOS   = [f'video{i:02d}' for i in range(17, 21)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--img-zip', default='/data-robot/DroneDataSet/Anti-UAV-Tracking-V0.zip')
    p.add_argument('--gt-zip',  default='/data-robot/DroneDataSet/Anti-UAV-Tracking-V0GT.zip')
    p.add_argument('--out-dir', default='/home/adlink/chenx/datasets/antiuav')
    return p.parse_args()


def load_gt(gt_zip_path):
    """Return dict: video_name → list of (x, y, w, h) or None (invisible)."""
    gt = {}
    with zipfile.ZipFile(gt_zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith('_gt.txt'):
                continue
            # Anti-UAV-Tracking-V0GT/video01_gt.txt → video01
            stem = Path(name).stem.replace('_gt', '')
            lines = zf.read(name).decode().strip().splitlines()
            bboxes = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 4:
                    bboxes.append(None)
                    continue
                x, y, w, h = [float(v) for v in parts[:4]]
                bboxes.append((x, y, w, h) if w > 0 and h > 0 else None)
            gt[stem] = bboxes
    return gt


def bbox_to_yolo(x, y, w, h, img_w=IMG_W, img_h=IMG_H):
    """Convert top-left (x,y,w,h) absolute to normalised (xc,yc,w,h)."""
    xc = (x + w / 2) / img_w
    yc = (y + h / 2) / img_h
    wn = w / img_w
    hn = h / img_h
    # clamp to valid range
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    wn = max(0.0, min(1.0, wn))
    hn = max(0.0, min(1.0, hn))
    return xc, yc, wn, hn


def extract_split(img_zip_path, gt, out_dir, split, video_list):
    img_out = Path(out_dir) / 'images' / split
    lbl_out = Path(out_dir) / 'labels' / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    kept = skipped = 0
    with zipfile.ZipFile(img_zip_path) as zf:
        all_names = zf.namelist()
        for video in video_list:
            bboxes = gt.get(video, [])
            # collect frame paths sorted
            prefix = f'Anti-UAV-Tracking-V0/{video}/'
            frames = sorted(n for n in all_names
                            if n.startswith(prefix) and n.endswith('.jpg'))
            if not frames:
                print(f'  WARNING: no frames found for {video}')
                continue
            if len(frames) != len(bboxes):
                print(f'  WARNING: {video}: {len(frames)} frames but {len(bboxes)} GT lines')

            for frame_path, bbox in zip(frames, bboxes):
                if bbox is None:
                    skipped += 1
                    continue   # UAV invisible / not annotated

                frame_name = Path(frame_path).name          # e.g. 00001.jpg
                stem = Path(frame_name).stem                # 00001
                out_stem = f'{video}_{stem}'

                # write image
                img_data = zf.read(frame_path)
                img_dst = img_out / f'{out_stem}.jpg'
                img_dst.write_bytes(img_data)

                # write label
                xc, yc, w, h = bbox_to_yolo(*bbox)
                lbl_dst = lbl_out / f'{out_stem}.txt'
                lbl_dst.write_text(f'0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n')
                kept += 1

    print(f'  {split}: kept {kept} frames, skipped {skipped} invisible')
    return kept


def write_yaml(out_dir):
    yaml_path = Path(out_dir) / 'antiuav.yaml'
    yaml_path.write_text(f"""\
path: {out_dir}
train: images/train
val:   images/val

nc: 1
names: ['uav']
""")
    print(f'  wrote {yaml_path}')


def main():
    args = parse_args()
    print('Loading GT annotations...')
    gt = load_gt(args.gt_zip)
    print(f'  loaded GT for {len(gt)} videos')

    print('Extracting train split...')
    extract_split(args.img_zip, gt, args.out_dir, 'train', TRAIN_VIDEOS)

    print('Extracting val split...')
    extract_split(args.img_zip, gt, args.out_dir, 'val', VAL_VIDEOS)

    write_yaml(args.out_dir)
    print('Done.')


if __name__ == '__main__':
    main()
