#!/usr/bin/env python3
"""
Filter ROI160 dataset by data source
按数据来源分别处理：ARD100保留，Diverse过滤大目标

Strategy:
  - ARD100 original: Keep all (already clean, <0.1% >80px)
  - Diverse dataset: Filter >80px targets
"""

import os
import shutil
from pathlib import Path
import argparse


def parse_yolo_label(label_path, img_size=160):
    """解析YOLO标签，返回目标的像素尺寸"""
    targets = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls, cx, cy, w, h = map(float, parts[:5])
                    # 转换为像素
                    w_px = w * img_size
                    h_px = h * img_size
                    max_size = max(w_px, h_px)
                    targets.append({
                        'cls': cls,
                        'cx': cx,
                        'cy': cy,
                        'w': w,
                        'h': h,
                        'w_px': w_px,
                        'h_px': h_px,
                        'max_size': max_size
                    })
    except Exception as e:
        print(f"Error parsing {label_path}: {e}")
    return targets


def should_filter(label_path, img_size, max_size_threshold):
    """判断是否应该过滤"""
    targets = parse_yolo_label(label_path, img_size)

    if not targets:
        return False, "No targets", 0

    max_target = max(targets, key=lambda t: t['max_size'])
    max_size = max_target['max_size']

    if max_size > max_size_threshold:
        return True, f"Oversized: {max_size:.1f}px", max_size

    return False, "OK", max_size


def filter_dataset_by_source(
    input_dir,
    output_dir,
    roi_size=160,
    max_size=80,
    ard100_pattern="ard100",
    diverse_pattern="diverse",
    dry_run=False
):
    """
    按数据来源分别处理数据集

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        roi_size: ROI尺寸
        max_size: 最大目标尺寸阈值（只应用于Diverse）
        ard100_pattern: ARD100文件名匹配模式
        diverse_pattern: Diverse文件名匹配模式
        dry_run: 是否只统计不实际复制
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    stats = {
        'ard100': {
            'total': 0,
            'kept': 0,
            'filtered': 0,
            'size_dist': {}
        },
        'diverse': {
            'total': 0,
            'kept': 0,
            'filtered': 0,
            'size_dist': {}
        },
        'other': {
            'total': 0,
            'kept': 0,
            'filtered': 0,
            'size_dist': {}
        }
    }

    filtered_list = []

    # 处理train和val
    for split in ['train', 'val']:
        images_dir = input_path / split / 'images'
        labels_dir = input_path / split / 'labels'

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
        img_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))

        for img_file in sorted(img_files):
            # 判断数据来源
            if ard100_pattern in img_file.name.lower():
                source = 'ard100'
            elif diverse_pattern in img_file.name.lower():
                source = 'diverse'
            else:
                source = 'other'

            stats[source]['total'] += 1

            # 对应的标签文件
            label_file = labels_dir / (img_file.stem + '.txt')

            # 判断是否过滤
            filter_decision = False
            reason = "Keep all"
            target_size = 0

            if source == 'ard100':
                # ARD100：全部保留
                filter_decision = False
                reason = "ARD100 - Keep all"
                if label_file.exists():
                    targets = parse_yolo_label(label_file, roi_size)
                    if targets:
                        target_size = max(t['max_size'] for t in targets)

            elif source == 'diverse':
                # Diverse：过滤>80px
                if label_file.exists():
                    filter_decision, reason, target_size = should_filter(
                        label_file, roi_size, max_size
                    )

            else:
                # Other：保留
                filter_decision = False
                reason = "Other - Keep"

            # 统计尺寸分布
            size_bin = int(target_size // 10) * 10
            stats[source]['size_dist'][size_bin] = \
                stats[source]['size_dist'].get(size_bin, 0) + 1

            # 执行过滤或保留
            if filter_decision:
                stats[source]['filtered'] += 1
                filtered_list.append((split, source, img_file.name, target_size))
                print(f"  ❌ [{source.upper()}] {img_file.name}: {reason}")
            else:
                stats[source]['kept'] += 1

                # 复制文件
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

    # 打印统计
    print_statistics(stats, filtered_list, max_size, dry_run)

    return stats


def print_statistics(stats, filtered_list, max_size, dry_run):
    """打印统计结果"""
    print(f"\n{'='*70}")
    print("数据集过滤统计 - 按来源分组")
    print(f"{'='*70}\n")

    total_all = 0
    kept_all = 0
    filtered_all = 0

    for source in ['ard100', 'diverse', 'other']:
        total = stats[source]['total']
        kept = stats[source]['kept']
        filtered = stats[source]['filtered']

        if total == 0:
            continue

        total_all += total
        kept_all += kept
        filtered_all += filtered

        print(f"📊 {source.upper()} 数据:")
        print(f"  总图片数:  {total:6d}")
        print(f"  保留:      {kept:6d} ({kept/total*100:5.1f}%)")
        print(f"  过滤:      {filtered:6d} ({filtered/total*100:5.1f}%)")

        # 尺寸分布
        if stats[source]['size_dist']:
            print(f"  尺寸分布:")
            max_count = max(stats[source]['size_dist'].values())
            for size_bin in sorted(stats[source]['size_dist'].keys()):
                count = stats[source]['size_dist'][size_bin]
                percent = count / total * 100
                bar_len = int((count / max_count) * 30)
                bar = '█' * bar_len

                marker = ''
                if source == 'diverse' and size_bin == (max_size // 10) * 10:
                    marker = ' ← 过滤线'

                print(f"    {size_bin:3d}-{size_bin+9:3d}px: {count:4d} ({percent:5.1f}%) {bar}{marker}")
        print()

    # 总体统计
    print(f"{'='*70}")
    print(f"📊 总体统计:")
    print(f"  总图片数:  {total_all:6d}")
    print(f"  保留:      {kept_all:6d} ({kept_all/total_all*100:5.1f}%)")
    print(f"  过滤:      {filtered_all:6d} ({filtered_all/total_all*100:5.1f}%)")
    print(f"{'='*70}\n")

    # 过滤详情
    if filtered_list:
        print(f"📋 被过滤的图片 (前20个):")
        for split, source, name, size in filtered_list[:20]:
            print(f"  [{source.upper()}] {split}/{name}: {size:.1f}px")
        if len(filtered_list) > 20:
            print(f"  ... 还有 {len(filtered_list)-20} 个")
        print()

    if dry_run:
        print(f"💡 这是dry-run模式，没有实际修改文件")
        print(f"   移除 --dry-run 参数来执行实际过滤\n")
    else:
        print(f"✅ 过滤完成！\n")


def main():
    parser = argparse.ArgumentParser(
        description='Filter ROI160 dataset by data source (ARD100 vs Diverse)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategy:
  - ARD100 original: Keep ALL (already clean, <0.1%% >80px)
  - Diverse dataset: Filter >80px targets
  - Other files: Keep all

Example:
  # Dry run first
  python tools/filter_by_source.py \\
      --input data/ard100_roi160_merged \\
      --output data/ard100_roi160_filtered \\
      --dry-run

  # Execute filtering
  python tools/filter_by_source.py \\
      --input data/ard100_roi160_merged \\
      --output data/ard100_roi160_filtered
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input dataset directory'
    )
    parser.add_argument(
        '--output',
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
        help='Max target size for Diverse dataset (default: 80)'
    )
    parser.add_argument(
        '--ard100-pattern',
        type=str,
        default='ard100',
        help='Pattern to identify ARD100 files (default: "ard100")'
    )
    parser.add_argument(
        '--diverse-pattern',
        type=str,
        default='diverse',
        help='Pattern to identify Diverse files (default: "diverse")'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run - only show statistics without copying files'
    )

    args = parser.parse_args()

    print("="*70)
    print("ROI160 Dataset Filter by Source")
    print("="*70)
    print(f"Input:           {args.input}")
    print(f"Output:          {args.output}")
    print(f"ROI size:        {args.roi_size}×{args.roi_size}")
    print(f"Max size:        {args.max_size}px (for Diverse only)")
    print(f"ARD100 pattern:  '{args.ard100_pattern}'")
    print(f"Diverse pattern: '{args.diverse_pattern}'")
    print(f"Mode:            {'DRY RUN' if args.dry_run else 'EXECUTE'}")
    print("="*70)
    print("\nStrategy:")
    print(f"  ✅ ARD100:  Keep ALL (already clean)")
    print(f"  ⚠️  Diverse: Filter >{args.max_size}px")
    print(f"  ✅ Other:   Keep ALL")
    print("="*70)

    # 检查输入目录
    if not os.path.exists(args.input):
        print(f"\n❌ 错误: 输入目录不存在: {args.input}")
        return 1

    # 执行过滤
    stats = filter_dataset_by_source(
        args.input,
        args.output,
        args.roi_size,
        args.max_size,
        args.ard100_pattern,
        args.diverse_pattern,
        args.dry_run
    )

    return 0


if __name__ == '__main__':
    exit(main())
