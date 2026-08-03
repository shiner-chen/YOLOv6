"""
Convert Pascal VOC XML annotations to YOLO TXT format.
Usage: python tools/convert_voc_to_yolo.py
"""
import xml.etree.ElementTree as ET
import os
from pathlib import Path

DATASET_ROOT = "/data-robot/DroneDataSet/Anti-UAV-Detect"
SPLITS = ["train", "val", "test"]
CLASSES = ["UAV"]  # class index 0


def convert_bbox(size, box):
    """Convert VOC bbox (xmin,ymin,xmax,ymax) to YOLO (cx,cy,w,h) normalized."""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    cx = (box[0] + box[2]) / 2.0 * dw
    cy = (box[1] + box[3]) / 2.0 * dh
    w  = (box[2] - box[0]) * dw
    h  = (box[3] - box[1]) * dh
    # Clamp to [0,1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.0, min(1.0, w))
    h  = max(0.0, min(1.0, h))
    return cx, cy, w, h


def convert_split(split):
    xml_dir   = Path(DATASET_ROOT) / split / "xml"
    label_dir = Path(DATASET_ROOT) / split / "labels"
    label_dir.mkdir(exist_ok=True)

    converted, skipped, empty = 0, 0, 0
    for xml_path in sorted(xml_dir.glob("*.xml")):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            print(f"  [WARN] parse error {xml_path.name}: {e}")
            skipped += 1
            continue

        size_el = root.find("size")
        w_img = int(size_el.find("width").text)
        h_img = int(size_el.find("height").text)

        lines = []
        for obj in root.findall("object"):
            cls_name = obj.find("name").text.strip()
            if cls_name not in CLASSES:
                continue  # skip unknown classes
            cls_id = CLASSES.index(cls_name)

            bb = obj.find("bndbox")
            xmin = float(bb.find("xmin").text)
            ymin = float(bb.find("ymin").text)
            xmax = float(bb.find("xmax").text)
            ymax = float(bb.find("ymax").text)

            # Skip degenerate boxes
            if xmax <= xmin or ymax <= ymin:
                continue

            cx, cy, bw, bh = convert_bbox((w_img, h_img), (xmin, ymin, xmax, ymax))
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        label_path = label_dir / (xml_path.stem + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(lines))

        if len(lines) == 0:
            empty += 1
        converted += 1

    print(f"  {split}: {converted} labels written, {empty} empty, {skipped} skipped")
    return converted


if __name__ == "__main__":
    print("Converting DUT Anti-UAV VOC XML → YOLO TXT ...")
    for split in SPLITS:
        convert_split(split)
    print("Done.")
