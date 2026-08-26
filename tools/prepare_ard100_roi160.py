#!/usr/bin/env python3
"""
prepare_ard100_roi160.py — Generate 160×160 ROI-crop YOLO dataset from ARD100 video sequences.

Key differences from prepare_ard100_roi320.py:
- ROI_SIZE = 160 (vs 320) — smaller patches for more challenging detection
- ROI_MARGIN = 16 (vs 32) — maintains 10% relative margin
- NEG_PER_POS = 6 (same) — positive:negative ratio of 1:6
- NEG_MARGIN = 16 (vs 32) — scaled proportionally with ROI size
- NEG_NO_OVERLAP = True — 6 negative samples must not overlap with each other

ROI extraction strategy
-----------------------
* For each positive frame a single 160×160 ROI is cropped so that the drone target
  falls at a **uniformly random position** inside the ROI (not always centred), with
  a configurable edge margin (default 16 px = 10 % of ROI).
* Visibility constraint: at least VIS_THRESH (default 0.70) of the target's bounding-box
  area must lie inside the ROI, AND the visible portion must be ≥ MIN_VISIBLE_PX × MIN_VISIBLE_PX
  pixels (default 4×4). If the random placement violates this after clamping to image
  bounds, up to MAX_ROI_RETRY attempts are made; frames that still fail are skipped.
* All GT boxes inside the ROI are written to the label file (not just the triggering
  target — handles multi-target frames gracefully).

Negative sample strategy
------------------------
* For every positive ROI, NEG_PER_POS (default 6) random 160×160 crops are generated
  from the same frame such that each crop:
  1. Has zero overlap with every GT bbox plus a safety margin of NEG_MARGIN px (default 16)
  2. Does NOT overlap with any previously generated negative ROI (ensures diversity)
* Up to MAX_NEG_RETRY attempts per slot; if a valid crop cannot be found the slot is
  simply skipped (label file written as empty).
* For annotated-negative frames (no drone in the entire frame) every stride-th frame
  yields one random 160×160 crop as a negative sample.

Frame sampling
--------------
* Positive frames : every STRIDE-th frame (0-indexed), auto×2 for 60 fps videos.
* Annotated-negative frames : all kept (they are very rare in ARD100).

Train / Val split
-----------------
Sequences (NOT frames) are split 80 / 20 with a fixed seed so no sequence appears in
both partitions — identical policy to prepare_ard100_roi320.py.  Test sequences come
from the test_videos folder and are kept entirely separate.

Output layout
-------------
  <dst>/
    images/{train,val,test}/<seq>_<frame:06d>_roi<idx>.jpg
    labels/{train,val,test}/<seq>_<frame:06d>_roi<idx>.txt
    ard100_roi160.yaml
    split_manifest.json

Usage
-----
  python tools/prepare_ard100_roi160.py \
      --src  /data-robot/DroneDataSet/ARD100 \
      --dst  /home/adlink/data/ARD100_roi160 \
      --stride 20 \
      --val-split 0.2 \
      --seed 42 \
      --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── constants ─────────────────────────────────────────────────────────────────
CLASSES        = ['drone']   # Pascal VOC label → YOLO class id (case-insensitive)
ROI_SIZE       = 160         # Output patch size (square) — CHANGED from 320
ROI_MARGIN     = 16          # Min pixels from target centre to ROI edge (10% of ROI)
VIS_THRESH     = 0.70        # Min fraction of GT bbox area that must be inside ROI
MIN_VISIBLE_PX = 4           # Min visible width AND height in pixels
MAX_ROI_RETRY  = 20          # Attempts to find a valid ROI placement per target
NEG_PER_POS    = 6           # Negative crops per positive frame — maintains 1:6 ratio
NEG_MARGIN     = 16          # Safety margin (px) around GT boxes for negative crops
NEG_NO_OVERLAP = True        # Negative ROIs must not overlap with each other
MAX_NEG_RETRY  = 50          # Attempts per negative crop slot
JPEG_QUALITY   = 90
# ──────────────────────────────────────────────────────────────────────────────


# ── data structures ───────────────────────────────────────────────────────────

class BBox:
    """Integer pixel bounding box (xmin, ymin, xmax, ymax), 0-indexed, exclusive end."""
    __slots__ = ('xmin', 'ymin', 'xmax', 'ymax')

    def __init__(self, xmin: int, ymin: int, xmax: int, ymax: int):
        self.xmin = xmin; self.ymin = ymin
        self.xmax = xmax; self.ymax = ymax

    @property
    def w(self) -> int: return self.xmax - self.xmin
    @property
    def h(self) -> int: return self.ymax - self.ymin
    @property
    def area(self) -> int: return self.w * self.h
    @property
    def cx(self) -> float: return (self.xmin + self.xmax) / 2.0
    @property
    def cy(self) -> float: return (self.ymin + self.ymax) / 2.0

    def intersection_area(self, other: 'BBox') -> int:
        ix1 = max(self.xmin, other.xmin); iy1 = max(self.ymin, other.ymin)
        ix2 = min(self.xmax, other.xmax); iy2 = min(self.ymax, other.ymax)
        return max(0, ix2 - ix1) * max(0, iy2 - iy1)

    def visible_inside(self, roi: 'BBox') -> 'BBox':
        """Return the portion of this bbox clipped to *roi* (absolute coords)."""
        return BBox(
            max(self.xmin, roi.xmin), max(self.ymin, roi.ymin),
            min(self.xmax, roi.xmax), min(self.ymax, roi.ymax),
        )

    def to_yolo(self, roi: 'BBox') -> Optional[str]:
        """
        Return YOLO label line (relative to *roi*) or None if below visibility thresholds.
        The class id is always 0 (drone).
        """
        clipped = self.visible_inside(roi)
        if clipped.w <= 0 or clipped.h <= 0:
            return None
        vis_ratio = clipped.area / max(self.area, 1)
        if vis_ratio < VIS_THRESH:
            return None
        if clipped.w < MIN_VISIBLE_PX or clipped.h < MIN_VISIBLE_PX:
            return None
        roi_w = roi.w; roi_h = roi.h
        cx = (clipped.xmin + clipped.xmax) / 2.0 - roi.xmin
        cy = (clipped.ymin + clipped.ymax) / 2.0 - roi.ymin
        bw = clipped.w; bh = clipped.h
        return (f'0 {cx/roi_w:.6f} {cy/roi_h:.6f}'
                f' {bw/roi_w:.6f} {bh/roi_h:.6f}')

    def __repr__(self):
        return f'BBox({self.xmin},{self.ymin},{self.xmax},{self.ymax})'


# ── XML parsing ───────────────────────────────────────────────────────────────

def parse_voc_xml(xml_path: str) -> Tuple[int, int, List[BBox]]:
    """Return (img_w, img_h, list_of_bboxes). Empty list → negative frame."""
    root = ET.parse(xml_path).getroot()
    size = root.find('size')
    img_w = int(size.find('width').text)
    img_h = int(size.find('height').text)
    boxes: List[BBox] = []
    for obj in root.findall('object'):
        cls_name = obj.find('name').text.strip().lower()
        if cls_name not in CLASSES:
            continue
        bb = obj.find('bndbox')
        xmin = max(0, int(float(bb.find('xmin').text)))
        ymin = max(0, int(float(bb.find('ymin').text)))
        xmax = min(img_w, int(float(bb.find('xmax').text)))
        ymax = min(img_h, int(float(bb.find('ymax').text)))
        if xmax > xmin and ymax > ymin:
            boxes.append(BBox(xmin, ymin, xmax, ymax))
    return img_w, img_h, boxes


# ── ffprobe ───────────────────────────────────────────────────────────────────

def ffprobe_video(video_path: str) -> Tuple[int, int, float]:
    """Return (width, height, fps)."""
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


# ── annotation collection ─────────────────────────────────────────────────────

def collect_sequence_annotations(ann_dir: str, seq: str) -> List[Tuple[int, str]]:
    """Return sorted (frame_0idx, xml_path) for every annotated frame."""
    seq_dir = os.path.join(ann_dir, seq)
    entries = []
    for fname in sorted(os.listdir(seq_dir)):
        if not fname.endswith('.xml'):
            continue
        stem = fname[:-4]
        frame_1 = int(stem.split('_')[-1])
        frame_0 = frame_1 - 1
        entries.append((frame_0, os.path.join(seq_dir, fname)))
    return entries


def build_frame_selection(
    entries: List[Tuple[int, str]],
    stride: int,
) -> List[Tuple[int, str]]:
    """Positive frames: keep every stride-th. Negative frames: keep all."""
    selected = []
    for frame_0, xml_path in entries:
        root = ET.parse(xml_path).getroot()
        is_pos = root.find('object') is not None
        if is_pos:
            if frame_0 % stride == 0:
                selected.append((frame_0, xml_path))
        else:
            selected.append((frame_0, xml_path))
    return selected


# ── ROI placement ─────────────────────────────────────────────────────────────

def sample_roi_for_target(
    target: BBox,
    img_w: int,
    img_h: int,
    rng: random.Random,
) -> Optional[BBox]:
    """
    Sample a 160×160 ROI (in image coordinates) such that *target* lands at a
    uniformly random position inside the ROI (with ROI_MARGIN clearance from
    every edge).  Returns None after MAX_ROI_RETRY failed attempts.
    """
    cx = int(target.cx); cy = int(target.cy)
    # Offset = position of target centre within ROI
    lo = ROI_MARGIN
    hi = ROI_SIZE - ROI_MARGIN
    for _ in range(MAX_ROI_RETRY):
        rx = rng.randint(lo, hi)
        ry = rng.randint(lo, hi)
        roi_left = cx - rx
        roi_top  = cy - ry
        # Clamp to image bounds
        roi_left = max(0, min(img_w - ROI_SIZE, roi_left))
        roi_top  = max(0, min(img_h - ROI_SIZE, roi_top))
        roi = BBox(roi_left, roi_top, roi_left + ROI_SIZE, roi_top + ROI_SIZE)
        # Check visibility
        clipped = target.visible_inside(roi)
        if clipped.w <= 0 or clipped.h <= 0:
            continue
        if clipped.area / max(target.area, 1) < VIS_THRESH:
            continue
        if clipped.w < MIN_VISIBLE_PX or clipped.h < MIN_VISIBLE_PX:
            continue
        return roi
    return None


def sample_negative_roi(
    all_boxes: List[BBox],
    img_w: int,
    img_h: int,
    used_rois: List[BBox],
    rng: random.Random,
) -> Optional[BBox]:
    """
    Sample a random 160×160 ROI that:
    1. Has zero overlap with every GT bbox (expanded by NEG_MARGIN)
    2. Does NOT overlap with any previously used ROI (if NEG_NO_OVERLAP is True)
    Returns None after MAX_NEG_RETRY attempts.
    """
    # Expand GT boxes by safety margin for exclusion test
    expanded = [
        BBox(
            max(0, b.xmin - NEG_MARGIN),
            max(0, b.ymin - NEG_MARGIN),
            min(img_w, b.xmax + NEG_MARGIN),
            min(img_h, b.ymax + NEG_MARGIN),
        )
        for b in all_boxes
    ]
    for _ in range(MAX_NEG_RETRY):
        left = rng.randint(0, max(0, img_w - ROI_SIZE))
        top  = rng.randint(0, max(0, img_h - ROI_SIZE))
        roi  = BBox(left, top, left + ROI_SIZE, top + ROI_SIZE)

        # Reject if roi overlaps any expanded GT box
        if any(roi.intersection_area(ex) > 0 for ex in expanded):
            continue

        # Reject if overlaps with previously used ROI (for negative diversity)
        if NEG_NO_OVERLAP and used_rois:
            if any(roi.intersection_area(u) > 0 for u in used_rois):
                continue

        return roi
    return None


# ── frame-level crop generation ───────────────────────────────────────────────

def generate_crops_for_frame(
    xml_path: str,
    img_w: int,
    img_h: int,
    rng: random.Random,
) -> List[Tuple[BBox, List[str], str]]:
    """
    Return a list of (roi_bbox, yolo_label_lines, tag) tuples for one frame.
    tag is 'pos' or 'neg'.

    * Positive frame → one ROI per target (all visible targets labelled) + NEG_PER_POS negatives.
    * Negative frame → one random ROI as a pure negative sample.
    """
    _, _, boxes = parse_voc_xml(xml_path)
    crops: List[Tuple[BBox, List[str], str]] = []

    if not boxes:
        # Annotated-negative frame: one random 160×160 crop
        left = rng.randint(0, max(0, img_w - ROI_SIZE))
        top  = rng.randint(0, max(0, img_h - ROI_SIZE))
        roi  = BBox(left, top, left + ROI_SIZE, top + ROI_SIZE)
        crops.append((roi, [], 'neg'))
        return crops

    # Positive frame: one ROI per target (ARD100 is almost always single-target)
    pos_rois: List[BBox] = []
    for target in boxes:
        roi = sample_roi_for_target(target, img_w, img_h, rng)
        if roi is None:
            continue  # could not satisfy visibility constraint — skip this target
        # Collect labels for ALL boxes visible inside this ROI
        labels = []
        for b in boxes:
            line = b.to_yolo(roi)
            if line is not None:
                labels.append(line)
        if not labels:
            continue  # safety check: should not happen
        crops.append((roi, labels, 'pos'))
        pos_rois.append(roi)

    if not pos_rois:
        return crops  # no usable positive ROI found for this frame

    # Negative crops from the same frame (6 non-overlapping negative samples)
    used_neg_rois = list(pos_rois)
    for _ in range(NEG_PER_POS):
        neg_roi = sample_negative_roi(boxes, img_w, img_h, used_neg_rois, rng)
        if neg_roi is None:
            continue
        crops.append((neg_roi, [], 'neg'))
        used_neg_rois.append(neg_roi)

    return crops


# ── per-frame write ───────────────────────────────────────────────────────────

def write_crop(
    raw_rgb: bytes,
    img_w: int,
    img_h: int,
    roi: BBox,
    labels: List[str],
    out_img_path: str,
    out_lbl_path: str,
) -> None:
    """Crop *raw_rgb* to *roi*, save JPEG + YOLO label file."""
    from PIL import Image

    img = Image.frombytes('RGB', (img_w, img_h), raw_rgb)
    patch = img.crop((roi.xmin, roi.ymin, roi.xmax, roi.ymax))
    patch.save(out_img_path, 'JPEG', quality=JPEG_QUALITY)

    with open(out_lbl_path, 'w') as f:
        if labels:
            f.write('\n'.join(labels) + '\n')
        # empty file for negatives — intentional


# ── core extraction loop ──────────────────────────────────────────────────────

def extract_and_write(
    video_path: str,
    selection: List[Tuple[int, str]],   # (frame_0idx, xml_path)
    out_img_dir: str,
    out_lbl_dir: str,
    seq: str,
    img_w: int,
    img_h: int,
    rng: random.Random,
) -> Tuple[int, int]:
    """
    Stream raw frames via ffmpeg, crop selected frames, write patches.
    Returns (n_pos_patches, n_neg_patches).
    """
    if not selection:
        return 0, 0

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    want: Dict[int, str] = {f0: xml for f0, xml in selection}
    max_frame = max(want)
    frame_bytes = img_w * img_h * 3  # RGB24

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

            if current_frame in want:
                xml_path = want[current_frame]
                _, _, boxes = parse_voc_xml(xml_path)
                crops = generate_crops_for_frame(xml_path, img_w, img_h, rng)

                for roi_idx, (roi, labels, tag) in enumerate(crops):
                    stem = f'{seq}_{current_frame:06d}_roi{roi_idx:02d}'
                    img_out = os.path.join(out_img_dir, stem + '.jpg')
                    lbl_out = os.path.join(out_lbl_dir, stem + '.txt')
                    write_crop(raw, img_w, img_h, roi, labels, img_out, lbl_out)
                    if tag == 'pos':
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
    vid_folder: str,   # 'train_videos' or 'test_videos'
    split: str,        # 'train', 'val', or 'test' (output partition)
    src: str,
    dst: str,
    base_stride: int,
    seed: int,
) -> Dict:
    video_path = os.path.join(src, vid_folder, f'{seq}.mp4')
    ann_dir    = os.path.join(src, 'annotations')

    try:
        img_w, img_h, fps = ffprobe_video(video_path)
    except Exception as e:
        print(f'  [WARN] ffprobe failed for {seq}: {e}', flush=True)
        return {'seq': seq, 'error': str(e)}

    stride = base_stride * 2 if fps > 45 else base_stride

    try:
        entries   = collect_sequence_annotations(ann_dir, seq)
        selection = build_frame_selection(entries, stride)
    except Exception as e:
        print(f'  [WARN] annotation parse failed for {seq}: {e}', flush=True)
        return {'seq': seq, 'error': str(e)}

    out_img_dir = os.path.join(dst, 'images', split)
    out_lbl_dir = os.path.join(dst, 'labels', split)

    # Per-sequence RNG derived from global seed + sequence name for reproducibility
    rng = random.Random(seed + hash(seq) & 0xFFFFFFFF)

    n_pos, n_neg = extract_and_write(
        video_path, selection, out_img_dir, out_lbl_dir,
        seq, img_w, img_h, rng,
    )

    result = {
        'seq': seq, 'fps': round(fps, 2), 'stride': stride,
        'total_annotations': len(entries),
        'selected_frames': len(selection),
        'written_pos': n_pos, 'written_neg': n_neg,
    }
    print(
        f'  {seq:<22} fps={fps:5.1f}  stride={stride}'
        f'  frames={len(selection):5d}'
        f'  pos_patches={n_pos:5d}  neg_patches={n_neg:5d}',
        flush=True,
    )
    return result


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Prepare ARD100 160×160 ROI-crop dataset from raw videos.')
    ap.add_argument('--src',     default='/data-robot/DroneDataSet/ARD100')
    ap.add_argument('--dst',     default='/home/adlink/data/ARD100_roi160')
    ap.add_argument('--stride',  type=int,   default=20,
                    help='Frame stride for positive frames (auto×2 for 60fps)')
    ap.add_argument('--val-split', type=float, default=0.2,
                    help='Fraction of train sequences to use for val (default 0.2)')
    ap.add_argument('--seed',    type=int,   default=42)
    ap.add_argument('--workers', type=int,   default=1,
                    help='Parallel workers (default 1; >1 uses multiprocessing)')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print selection stats only, do not write images')
    args = ap.parse_args()

    src     = args.src
    dst     = args.dst
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

    # ── sequence-level train / val split ─────────────────────────────────────
    rng_split = random.Random(args.seed)
    shuffled  = train_all[:]
    rng_split.shuffle(shuffled)
    n_val      = max(1, round(len(shuffled) * args.val_split))
    val_seqs   = sorted(shuffled[:n_val])
    train_seqs = sorted(shuffled[n_val:])

    print(f'Sequences  train={len(train_seqs)}  val={len(val_seqs)}  test={len(test_seqs)}')
    print(f'Stride={args.stride}  val_split={args.val_split}  seed={args.seed}')
    print(f'ROI={ROI_SIZE}px  margin={ROI_MARGIN}px  vis_thresh={VIS_THRESH}'
          f'  neg_per_pos={NEG_PER_POS}  neg_no_overlap={NEG_NO_OVERLAP}')
    print(f'Output → {dst}\n')

    if args.dry_run:
        for seq, vf in [(s, 'train_videos') for s in train_seqs + val_seqs] + \
                       [(s, 'test_videos')  for s in test_seqs]:
            try:
                _, _, fps = ffprobe_video(os.path.join(src, vf, f'{seq}.mp4'))
            except Exception:
                fps = 30.0
            stride  = args.stride * 2 if fps > 45 else args.stride
            entries  = collect_sequence_annotations(ann_dir, seq)
            selected = build_frame_selection(entries, stride)
            pos_f = sum(
                1 for _, x in selected
                if ET.parse(x).getroot().find('object') is not None
            )
            neg_f = len(selected) - pos_f
            est_patches = pos_f * (1 + NEG_PER_POS) + neg_f
            print(f'  {seq:<22} fps={fps:5.1f}  stride={stride}'
                  f'  frames={len(selected):5d}  pos_f={pos_f:5d}'
                  f'  neg_f={neg_f:4d}  est_patches≈{est_patches:6d}')
        return

    # ── create output directories ─────────────────────────────────────────────
    for split in ('train', 'val', 'test'):
        os.makedirs(os.path.join(dst, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(dst, 'labels', split), exist_ok=True)

    # ── build work list: (seq, vid_folder, split) ─────────────────────────────
    work_list = (
        [(seq, 'train_videos', 'train') for seq in train_seqs] +
        [(seq, 'train_videos', 'val')   for seq in val_seqs]   +
        [(seq, 'test_videos',  'test')  for seq in test_seqs]
    )

    stats: Dict[str, List[Dict]] = {'train': [], 'val': [], 'test': []}

    if args.workers > 1:
        import multiprocessing as mp
        # Set global variables for worker processes
        global _worker_src, _worker_dst, _worker_stride, _worker_seed
        _worker_src = src
        _worker_dst = dst
        _worker_stride = args.stride
        _worker_seed = args.seed
        with mp.Pool(args.workers) as pool:
            results = pool.map(_mp_worker, work_list)
        for (seq, _, split), result in zip(work_list, results):
            stats[split].append(result)
    else:
        # ── train ──────────────────────────────────────────────────────────────
        print('=== TRAIN ===')
        for seq in train_seqs:
            r = process_sequence(seq, 'train_videos', 'train',
                                 src, dst, args.stride, args.seed)
            stats['train'].append(r)

        # ── val ────────────────────────────────────────────────────────────────
        print('\n=== VAL ===')
        for seq in val_seqs:
            r = process_sequence(seq, 'train_videos', 'val',
                                 src, dst, args.stride, args.seed)
            stats['val'].append(r)

        # ── test ───────────────────────────────────────────────────────────────
        print('\n=== TEST ===')
        for seq in test_seqs:
            r = process_sequence(seq, 'test_videos', 'test',
                                 src, dst, args.stride, args.seed)
            stats['test'].append(r)

    # ── summary ───────────────────────────────────────────────────────────────
    def total(split_stats: List[Dict], key: str) -> int:
        return sum(r.get(key, 0) for r in split_stats if 'error' not in r)

    print('\n=== SUMMARY ===')
    for split in ('train', 'val', 'test'):
        s = stats[split]
        pos   = total(s, 'written_pos')
        neg   = total(s, 'written_neg')
        total_ = pos + neg
        print(f'  {split:<6}  seqs={len(s):3d}  patches={total_:7d}'
              f'  pos={pos:7d}  neg={neg:7d}'
              f'  neg_ratio={neg/max(total_,1)*100:.1f}%')

    # ── write YAML ────────────────────────────────────────────────────────────
    yaml_path = os.path.join(dst, 'ard100_roi160.yaml')
    with open(yaml_path, 'w') as f:
        f.write('# ARD100 160×160 ROI-crop YOLO dataset\n')
        f.write(f'# stride={args.stride}  val_split={args.val_split}  seed={args.seed}\n')
        f.write(f'# ROI={ROI_SIZE}  margin={ROI_MARGIN}  vis_thresh={VIS_THRESH}'
                f'  neg_per_pos={NEG_PER_POS}  neg_no_overlap={NEG_NO_OVERLAP}\n')
        for split in ('train', 'val', 'test'):
            s = stats[split]
            pos = total(s, 'written_pos'); neg = total(s, 'written_neg')
            f.write(f'# {split}: patches={pos+neg}  pos={pos}  neg={neg}\n')
        f.write(f'\npath: {dst}\n')
        f.write(f'train: images/train\n')
        f.write(f'val:   images/val\n')
        f.write(f'test:  images/test\n')
        f.write(f'\nis_coco: False\nnc: {len(CLASSES)}\n')
        f.write(f"names: {CLASSES}\n")
    print(f'\nYAML     → {yaml_path}')

    # ── write manifest ────────────────────────────────────────────────────────
    manifest = {
        'train_sequences': train_seqs,
        'val_sequences':   val_seqs,
        'test_sequences':  test_seqs,
        'stride': args.stride, 'seed': args.seed,
        'roi_size': ROI_SIZE, 'roi_margin': ROI_MARGIN,
        'vis_thresh': VIS_THRESH, 'neg_per_pos': NEG_PER_POS,
        'neg_no_overlap': NEG_NO_OVERLAP,
    }
    manifest_path = os.path.join(dst, 'split_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'Manifest → {manifest_path}')


# ── multiprocessing helper ────────────────────────────────────────────────────

# Global variables for multiprocessing worker
_worker_src = None
_worker_dst = None
_worker_stride = None
_worker_seed = None

def _mp_worker(args_tuple):
    """Picklable worker function for multiprocessing.Pool.map."""
    seq, vid_folder, split = args_tuple
    return process_sequence(seq, vid_folder, split, _worker_src, _worker_dst, _worker_stride, _worker_seed)


if __name__ == '__main__':
    main()
