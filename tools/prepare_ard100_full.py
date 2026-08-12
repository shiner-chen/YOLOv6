#!/usr/bin/env python3
"""
prepare_ard100_full.py — Generate YOLO-format train/val/test splits from ARD100 video sequences.

Frame sampling strategy
-----------------------
* Positive frames  : every STRIDE-th frame (0, stride, 2*stride, …) to break temporal
                     correlation between adjacent frames.
* Negative frames  : all kept — they represent only ~1.6 % of raw data, so no down-
                     sampling is needed; keeping every negative improves neg-sample coverage.
* 60 fps videos    : stride is automatically doubled so temporal spacing stays ~same.

Train / Val split
-----------------
Sequences (not frames) are split 80 / 20 with a fixed seed so no sequence appears in
both partitions (prevents sequence-level temporal leakage).

Output layout
-------------
  <dst>/
    images/{train,val,test}/<seq>_<frame:06d>.jpg
    labels/{train,val,test}/<seq>_<frame:06d>.txt   (empty for negatives)
    ard100_full.yaml

Usage
-----
  python tools/prepare_ard100_full.py \\
      --src  /data-robot/DroneDataSet/ARD100 \\
      --dst  /data-robot/DroneDataSet/ARD100/ARD100_full \\
      --stride 5 \\
      --val-split 0.2 \\
      --seed 42 \\
      --workers 4
"""
import argparse
import os
import random
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
CLASSES = ['drone']   # Pascal VOC label → YOLO class id
# ──────────────────────────────────────────────────────────────────────────────


# ── helpers ──────────────────────────────────────────────────────────────────

def ffprobe_video(video_path: str) -> Tuple[int, int, float]:
    """Return (width, height, fps) for *video_path* via ffprobe."""
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate',
        '-of', 'csv=s=,:p=0',
        video_path,
    ]
    out = subprocess.check_output(cmd, text=True).strip().split(',')
    w, h = int(out[0]), int(out[1])
    num, den = out[2].split('/')
    fps = float(num) / float(den)
    return w, h, fps


def parse_voc_xml(xml_path: str, img_w: int, img_h: int) -> List[str]:
    """Return YOLO-format label lines (empty list for negative frames)."""
    root = ET.parse(xml_path).getroot()
    lines = []
    for obj in root.findall('object'):
        cls_name = obj.find('name').text.strip().lower()
        cls_id = CLASSES.index(cls_name) if cls_name in CLASSES else -1
        if cls_id < 0:
            continue
        bb = obj.find('bndbox')
        xmin = max(0.0, float(bb.find('xmin').text))
        ymin = max(0.0, float(bb.find('ymin').text))
        xmax = min(float(img_w), float(bb.find('xmax').text))
        ymax = min(float(img_h), float(bb.find('ymax').text))
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h
        if bw <= 0 or bh <= 0:
            continue
        cx = (xmin + xmax) / 2.0 / img_w
        cy = (ymin + ymax) / 2.0 / img_h
        lines.append(f'{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
    return lines


def collect_sequence_annotations(ann_dir: str, seq: str) -> List[Tuple[int, str]]:
    """
    Return sorted list of (frame_0idx, xml_path) for every annotated frame.
    Annotation filenames follow the pattern  <seq>_<NNNN>.xml  (1-indexed).
    """
    seq_dir = os.path.join(ann_dir, seq)
    entries = []
    for fname in sorted(os.listdir(seq_dir)):
        if not fname.endswith('.xml'):
            continue
        # extract 1-based frame number from filename
        stem = fname[:-4]                      # e.g. phantom09_0001
        frame_1 = int(stem.split('_')[-1])     # last numeric token
        frame_0 = frame_1 - 1                  # convert to 0-based
        entries.append((frame_0, os.path.join(seq_dir, fname)))
    return entries


def build_frame_selection(
    entries: List[Tuple[int, str]],
    stride: int,
) -> List[Tuple[int, str]]:
    """
    Apply sampling policy:
      * positive frame  → include only if frame_0idx % stride == 0
      * negative frame  → always include
    Returns sorted list of (frame_0idx, xml_path) to extract.
    """
    selected = []
    for frame_0, xml_path in entries:
        root = ET.parse(xml_path).getroot()
        is_positive = root.find('object') is not None
        if is_positive:
            if frame_0 % stride == 0:
                selected.append((frame_0, xml_path))
        else:
            selected.append((frame_0, xml_path))   # keep ALL negatives
    return selected


# ── core extraction ───────────────────────────────────────────────────────────

def extract_and_write(
    video_path: str,
    selection: List[Tuple[int, str]],
    out_img_dir: str,
    out_lbl_dir: str,
    seq: str,
    img_w: int,
    img_h: int,
    jpeg_quality: int = 90,
) -> Tuple[int, int]:
    """
    Stream raw frames from ffmpeg, save selected ones as JPEG + write YOLO labels.
    Returns (n_pos, n_neg) written.
    """
    from PIL import Image

    if not selection:
        return 0, 0

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    want = {f0: xml for f0, xml in selection}
    want_set = set(want.keys())
    max_frame = max(want_set)

    frame_bytes = img_w * img_h * 3   # RGB24

    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-i', video_path,
        '-vf', f'select=lte(n\\,{max_frame})',
        '-vsync', 'passthrough',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        'pipe:1',
    ]

    n_pos = n_neg = 0
    current_frame = 0

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break

            if current_frame in want_set:
                xml_path = want[current_frame]
                labels = parse_voc_xml(xml_path, img_w, img_h)

                stem = f'{seq}_{current_frame:06d}'
                img_out = os.path.join(out_img_dir, stem + '.jpg')
                lbl_out = os.path.join(out_lbl_dir, stem + '.txt')

                img = Image.frombytes('RGB', (img_w, img_h), raw)
                img.save(img_out, 'JPEG', quality=jpeg_quality)

                with open(lbl_out, 'w') as f:
                    f.write('\n'.join(labels))
                    if labels:
                        f.write('\n')

                if labels:
                    n_pos += 1
                else:
                    n_neg += 1

            current_frame += 1

    finally:
        proc.stdout.close()
        proc.wait()

    return n_pos, n_neg


# ── sequence-level worker ─────────────────────────────────────────────────────

def process_sequence(
    seq: str,
    split: str,
    src: str,
    dst: str,
    base_stride: int,
    jpeg_quality: int,
) -> Dict:
    """Process one video sequence: probe → select frames → extract → write labels."""
    video_path = os.path.join(src, f'{split}_videos', f'{seq}.mp4')
    ann_dir    = os.path.join(src, 'annotations')

    # ── probe video ──────────────────────────────────────────────────────────
    try:
        img_w, img_h, fps = ffprobe_video(video_path)
    except Exception as e:
        print(f'  [WARN] ffprobe failed for {seq}: {e}', flush=True)
        return {'seq': seq, 'error': str(e)}

    # Double stride for 60fps videos so temporal spacing stays ~same
    stride = base_stride * 2 if fps > 45 else base_stride

    # ── annotations → frame selection ────────────────────────────────────────
    try:
        entries   = collect_sequence_annotations(ann_dir, seq)
        selection = build_frame_selection(entries, stride)
    except Exception as e:
        print(f'  [WARN] annotation parse failed for {seq}: {e}', flush=True)
        return {'seq': seq, 'error': str(e)}

    # ── output dirs (split is train/val/test from the dataset perspective) ───
    out_img_dir = os.path.join(dst, 'images', split)
    out_lbl_dir = os.path.join(dst, 'labels', split)

    # ── extract frames ───────────────────────────────────────────────────────
    n_pos, n_neg = extract_and_write(
        video_path, selection, out_img_dir, out_lbl_dir,
        seq, img_w, img_h, jpeg_quality,
    )

    result = {
        'seq': seq, 'fps': round(fps, 2), 'stride': stride,
        'total_annotations': len(entries),
        'selected': len(selection),
        'written_pos': n_pos, 'written_neg': n_neg,
    }
    print(
        f'  {seq:<20} fps={fps:5.1f}  stride={stride}'
        f'  annotations={len(entries):5d}  selected={len(selection):5d}'
        f'  pos={n_pos:5d}  neg={n_neg:4d}',
        flush=True,
    )
    return result


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Prepare ARD100 YOLO dataset from raw videos.')
    ap.add_argument('--src',  default='/data-robot/DroneDataSet/ARD100',
                    help='ARD100 dataset root')
    ap.add_argument('--dst',  default='/data-robot/DroneDataSet/ARD100/ARD100_full',
                    help='Output dataset root')
    ap.add_argument('--stride',     type=int,   default=5,
                    help='Frame stride for positive samples (default 5; auto×2 for 60fps)')
    ap.add_argument('--val-split',  type=float, default=0.2,
                    help='Fraction of train sequences to use for val (default 0.2)')
    ap.add_argument('--seed',       type=int,   default=42)
    ap.add_argument('--jpeg-quality', type=int, default=90,
                    help='JPEG quality 1-95 (default 90)')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print selection stats only, do not write images')
    args = ap.parse_args()

    src = args.src
    dst = args.dst
    ann_dir = os.path.join(src, 'annotations')

    # ── collect sequence lists ────────────────────────────────────────────────
    train_all = sorted(
        f.replace('.mp4', '')
        for f in os.listdir(os.path.join(src, 'train_videos'))
        if f.endswith('.mp4')
    )
    test_seqs = sorted(
        f.replace('.mp4', '')
        for f in os.listdir(os.path.join(src, 'test_videos'))
        if f.endswith('.mp4')
    )

    # ── train / val split by sequence ────────────────────────────────────────
    rng = random.Random(args.seed)
    shuffled = train_all[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * args.val_split))
    val_seqs   = sorted(shuffled[:n_val])
    train_seqs = sorted(shuffled[n_val:])

    print(f'Sequences  train={len(train_seqs)}  val={len(val_seqs)}  test={len(test_seqs)}')
    print(f'Stride base={args.stride}  JPEG quality={args.jpeg_quality}')
    print(f'Output → {dst}\n')

    if args.dry_run:
        # just print selection stats
        for seq in train_seqs + val_seqs + test_seqs:
            # determine video folder
            if seq in train_seqs or seq in val_seqs:
                vid_folder = 'train_videos'
            else:
                vid_folder = 'test_videos'
            video_path = os.path.join(src, vid_folder, f'{seq}.mp4')
            try:
                _, _, fps = ffprobe_video(video_path)
            except Exception:
                fps = 30.0
            stride = args.stride * 2 if fps > 45 else args.stride
            entries   = collect_sequence_annotations(ann_dir, seq)
            selection = build_frame_selection(entries, stride)
            pos = sum(1 for f0, x in selection if ET.parse(x).getroot().find('object') is not None)
            neg = len(selection) - pos
            print(f'  {seq:<20} fps={fps:5.1f}  stride={stride}  selected={len(selection):5d}  pos={pos:5d}  neg={neg:4d}')
        return

    # ── create output directories ─────────────────────────────────────────────
    for split in ('train', 'val', 'test'):
        os.makedirs(os.path.join(dst, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(dst, 'labels', split), exist_ok=True)

    stats = {'train': [], 'val': [], 'test': []}

    # ── process train sequences ───────────────────────────────────────────────
    print('=== TRAIN ===')
    for seq in train_seqs:
        r = process_sequence(seq, 'train', src, dst, args.stride, args.jpeg_quality)
        stats['train'].append(r)

    # ── process val sequences ─────────────────────────────────────────────────
    print('\n=== VAL ===')
    for seq in val_seqs:
        r = process_sequence(seq, 'train', src, dst, args.stride, args.jpeg_quality)
        # move images/labels to val
        for kind in ('images', 'labels'):
            ext = '.jpg' if kind == 'images' else '.txt'
            src_dir = os.path.join(dst, kind, 'train')
            tgt_dir = os.path.join(dst, kind, 'val')
            os.makedirs(tgt_dir, exist_ok=True)
            for f in os.listdir(src_dir):
                if f.startswith(seq + '_'):
                    os.rename(os.path.join(src_dir, f), os.path.join(tgt_dir, f))
        stats['val'].append(r)

    # ── process test sequences ────────────────────────────────────────────────
    print('\n=== TEST ===')
    for seq in test_seqs:
        r = process_sequence(seq, 'test', src, dst, args.stride, args.jpeg_quality)
        stats['test'].append(r)

    # ── summary ───────────────────────────────────────────────────────────────
    def total(split_stats, key):
        return sum(r.get(key, 0) for r in split_stats if 'error' not in r)

    print('\n=== SUMMARY ===')
    for split in ('train', 'val', 'test'):
        s = stats[split]
        pos = total(s, 'written_pos')
        neg = total(s, 'written_neg')
        print(f'  {split:<6}  seqs={len(s):3d}  images={pos+neg:6d}  '
              f'pos={pos:6d}  neg={neg:5d}  neg_ratio={neg/(pos+neg)*100:.1f}%')

    # ── write YAML ────────────────────────────────────────────────────────────
    yaml_path = os.path.join(dst, 'ard100_full.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f'# ARD100 full-frame YOLO dataset\n')
        f.write(f'# stride={args.stride}  val_split={args.val_split}  seed={args.seed}\n')
        for split in ('train', 'val', 'test'):
            s = stats[split]
            pos = total(s, 'written_pos')
            neg = total(s, 'written_neg')
            f.write(f'# {split}: pos={pos}  neg={neg}\n')
        f.write(f'\npath: {dst}\n')
        f.write(f'train: images/train\n')
        f.write(f'val:   images/val\n')
        f.write(f'test:  images/test\n')
        f.write(f'\nnc: {len(CLASSES)}\n')
        f.write(f"names: {CLASSES}\n")
    print(f'\nYAML written → {yaml_path}')

    # ── save split manifest ───────────────────────────────────────────────────
    import json
    manifest = {
        'train_sequences': train_seqs,
        'val_sequences':   val_seqs,
        'test_sequences':  test_seqs,
        'stride':          args.stride,
        'seed':            args.seed,
    }
    manifest_path = os.path.join(dst, 'split_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'Manifest  written → {manifest_path}')


if __name__ == '__main__':
    main()
