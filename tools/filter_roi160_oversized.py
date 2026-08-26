#!/usr/bin/env python3
"""
Filter oversized targets from ROI160 dataset
剔除ROI160数据集中的大目标图片
"""

import os
import shutil
from pathlib import Path
import argparse


def parse_yolo_label(label_path):
    """解析YOLO格式标签，返回目标的宽高(归一化)"""
    targets = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # YOLO format: class x_center y_center width height (normalized)
                cls, cx, cy, w, h = map(float, parts[:5])
                targets.append((cls, cx, cy, w, h))
    return targets


def should_filter_image(label_path, roi_size=160, max_size_ratio=0.5):
    """
    判断图片是否应该被过滤

    Args:
        label_path: 标签文件路径
        roi_size: ROI尺寸
        max_size_ratio: 最大尺寸比例 (0.5 = 80px/160px)

    Returns:
        (should_filter, reason, max_target_size)
    """
    if not os.path.exists(label_path):
        return False, "No label file", 0

    targets = parse_yolo_label(label_path)

    if not targets:
        return False, "No targets", 0

    # 找到最大的目标
    max_w = max(t[3] for t in targets)
    max_h = max(t[4] for t in targets)
    max_edge = max(max_w, max_h)  # 归一化值

    max_pixel_size = int(max_edge * roi_size)

    if max_edge > max_size_ratio:
        return True, f"Oversized target: {max_pixel_size}px", max_pixel_size

    return False, "OK", max_pixel_size


def filter_dataset(
    data_dir,
    output_dir,
    roi_size=160,
    max_size_px=80,
    dry_run=False
):
    """
    过滤数据集中的大目标图片

    Args:
        data_dir: 原始数据目录
        output_dir: 输出目录
        roi_size: ROI尺寸
        max_size_px: 最大目标尺寸(像素)
        dry_run: 是否只统计不实际过滤
    """
    max_size_ratio = max_size_px / roi_size

    data_path = Path(data_dir)
    output_path = Path(output_dir)

    # 统计信息
    stats = {
        'total': 0,
        'filtered': 0,
        'kept': 0,
        'size_distribution': {}
    }

    filtered_images = []

    # 遍历train和val
    for split in ['train', 'val']:
        images_dir = data_path / split / 'images'
        labels_dir = data_path / split / 'labels'

        if not images_dir.exists():
            print(f"⚠️  {images_dir} not found, skipping...")
            continue

        print(f"\n{'='*70}")
        print(f"Processing {split} split...")
        print(f"{'='*70}")

        # 创建输出目录
        if not dry_run:
            (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

        # 遍历所有图片
        for img_file in sorted(images_dir.glob('*.jpg')) + \
                        sorted(images_dir.glob('*.png')):
            stats['total'] += 1

            # 对应的标签文件
            label_file = labels_dir / (img_file.stem + '.txt')

            # 判断是否过滤
            should_filter, reason, target_size = should_filter_image(
                label_file, roi_size, max_size_ratio
            )

            # 统计尺寸分布
            size_bin = (target_size // 10) * 10  # 按10px分组
            stats['size_distribution'][size_bin] = \
                stats['size_distribution'].get(size_bin, 0) + 1

            if should_filter:
                stats['filtered'] += 1
                filtered_images.append((split, img_file.name, target_size))
                print(f"  ❌ {img_file.name}: {reason}")
            else:
                stats['kept'] += 1

                # 复制到输出目录
                if not dry_run:
                    shutil.copy2(
                        img_file,
                        output_path / split / 'images' / img_file.name
                    )
                    if label_file.exists():
                        shutil.copy2(
                            label_file,
                            output_path / split / 'labels' / label_file.name
                        )

    # 打印统计信息
    print(f"\n{'='*70}")
    print("统计结果")
    print(f"{'='*70}")
    print(f"总图片数:     {stats['total']}")
    print(f"保留:         {stats['kept']} ({stats['kept']/stats['total']*100:.1f}%)")
    print(f"过滤:         {stats['filtered']} ({stats['filtered']/stats['total']*100:.1f}%)")

    print(f"\n目标尺寸分布:")
    for size_bin in sorted(stats['size_distribution'].keys()):
        count = stats['size_distribution'][size_bin]
        percent = count / stats['total'] * 100
        bar = '█' * int(percent / 2)
        print(f"  {size_bin:3d}-{size_bin+9:3d}px: {count:4d} ({percent:5.1f}%) {bar}")

    if filtered_images:
        print(f"\n过滤的图片 (前20个):")
        for split, img_name, size in filtered_images[:20]:
            print(f"  {split}/{img_name} - {size}px")
        if len(filtered_images) > 20:
            print(f"  ... 还有 {len(filtered_images)-20} 个")

    if dry_run:
        print(f"\n💡 这是dry-run模式，没有实际修改文件")
        print(f"   移除 --dry-run 参数来执行实际过滤")
    else:
        print(f"\n✅ 过滤完成！输出目录: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Filter oversized targets from ROI160 dataset'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Input dataset directory (contains train/val folders)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for filtered dataset'
    )
    parser.add_argument(
        '--roi-size',
        type=int,
        default=160,
        help='ROI size (default: 160)'
    )
    parser.add_argument(
        '--max-size',
        type=int,
        default=80,
        help='Maximum target size in pixels (default: 80)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run - only print statistics without copying files'
    )

    args = parser.parse_args()

    print("="*70)
    print("ROI160 Dataset Filter - Remove Oversized Targets")
    print("="*70)
    print(f"Input:      {args.data_dir}")
    print(f"Output:     {args.output_dir}")
    print(f"ROI size:   {args.roi_size}×{args.roi_size}")
    print(f"Max target: {args.max_size}px ({args.max_size/args.roi_size*100:.1f}% of ROI)")
    print(f"Threshold:  边长>{args.max_size/args.roi_size*100:.0f}%, 面积>{(args.max_size/args.roi_size)**2*100:.0f}%")
    print(f"Mode:       {'DRY RUN' if args.dry_run else 'EXECUTE'}")
    print("="*70)

    filter_dataset(
        args.data_dir,
        args.output_dir,
        args.roi_size,
        args.max_size,
        args.dry_run
    )
