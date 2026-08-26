#!/usr/bin/env python3
"""
Estimate filtering impact on Motion-guided ROI160 dataset
评估Motion-guided ROI160数据集过滤后的剩余比例
"""

import os
import sys
from pathlib import Path


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
                    targets.append((cls, cx, cy, w_px, h_px, max_size))
    except:
        pass
    return targets


def estimate_filtering_impact(data_dir, roi_size=160, max_size=80):
    """
    评估过滤影响

    Args:
        data_dir: 数据目录
        roi_size: ROI尺寸
        max_size: 最大目标尺寸阈值

    Returns:
        统计结果字典
    """
    data_path = Path(data_dir)

    stats = {
        'total_images': 0,
        'images_with_targets': 0,
        'images_filtered': 0,
        'images_kept': 0,
        'total_targets': 0,
        'targets_oversized': 0,
        'size_distribution': {},
        'filtered_details': []
    }

    # 遍历train和val
    for split in ['train', 'val']:
        labels_dir = data_path / split / 'labels'
        images_dir = data_path / split / 'images'

        if not labels_dir.exists():
            continue

        print(f"\n{'='*70}")
        print(f"Analyzing {split} split...")
        print(f"{'='*70}")

        label_files = list(labels_dir.glob('*.txt'))

        for label_file in label_files:
            stats['total_images'] += 1

            targets = parse_yolo_label(label_file, roi_size)

            if not targets:
                # 无目标的图片
                stats['images_kept'] += 1
                continue

            stats['images_with_targets'] += 1
            stats['total_targets'] += len(targets)

            # 找最大目标
            max_target_size = max(t[5] for t in targets)

            # 统计尺寸分布
            size_bin = int(max_target_size // 10) * 10
            stats['size_distribution'][size_bin] = \
                stats['size_distribution'].get(size_bin, 0) + 1

            # 判断是否过滤
            if max_target_size > max_size:
                stats['images_filtered'] += 1
                stats['targets_oversized'] += len(targets)
                stats['filtered_details'].append((
                    split,
                    label_file.name,
                    max_target_size,
                    len(targets)
                ))
            else:
                stats['images_kept'] += 1

    return stats


def print_statistics(stats, max_size):
    """打印统计结果"""

    print(f"\n{'='*70}")
    print("过滤影响评估结果")
    print(f"{'='*70}")
    print(f"过滤阈值: >{max_size}px")
    print(f"{'='*70}\n")

    # 基本统计
    total = stats['total_images']
    kept = stats['images_kept']
    filtered = stats['images_filtered']

    print(f"📊 总体统计:")
    print(f"  总图片数:           {total:6d}")
    print(f"  有目标的图片:        {stats['images_with_targets']:6d}")
    print(f"  保留图片:           {kept:6d} ({kept/total*100:5.1f}%)")
    print(f"  过滤图片:           {filtered:6d} ({filtered/total*100:5.1f}%)")
    print()

    # 目标统计
    print(f"🎯 目标统计:")
    print(f"  总目标数:           {stats['total_targets']:6d}")
    print(f"  超大目标数:         {stats['targets_oversized']:6d}")
    print()

    # 尺寸分布
    print(f"📏 目标尺寸分布:")
    if stats['size_distribution']:
        max_count = max(stats['size_distribution'].values())
        for size_bin in sorted(stats['size_distribution'].keys()):
            count = stats['size_distribution'][size_bin]
            percent = count / total * 100
            bar_len = int((count / max_count) * 40)
            bar = '█' * bar_len
            marker = ' ← 过滤线' if size_bin == (max_size // 10) * 10 else ''
            print(f"  {size_bin:3d}-{size_bin+9:3d}px: {count:5d} ({percent:5.1f}%) {bar}{marker}")
    print()

    # 预测
    if filtered > 0:
        print(f"⚠️  警告: 将损失 {filtered/total*100:.1f}% 的图片！")
        print()
        print(f"💡 典型Motion-guided场景预期:")
        print(f"   - 正常情况: 过滤率应在 1-5%")
        print(f"   - 如果超过10%: 可能ROI切片策略有问题")
        print(f"   - 当前: {filtered/total*100:.1f}%")
        print()

        # 显示一些被过滤的样本
        if stats['filtered_details']:
            print(f"📋 被过滤的图片样本 (前10个):")
            for split, name, size, num_targets in stats['filtered_details'][:10]:
                print(f"  {split}/{name}: {size:.1f}px ({num_targets}个目标)")
            if len(stats['filtered_details']) > 10:
                print(f"  ... 还有 {len(stats['filtered_details'])-10} 个")
    else:
        print(f"✅ 太好了！没有图片需要过滤。")
        print(f"   所有目标都在 {max_size}px 以内。")

    print(f"\n{'='*70}")

    return kept, filtered, total


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Estimate filtering impact on Motion-guided ROI160 dataset'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/ard100_roi160_merged',
        help='Dataset directory (contains train/val folders)'
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
        help='Maximum target size threshold (default: 80)'
    )

    args = parser.parse_args()

    print("="*70)
    print("Motion-guided ROI160 过滤影响评估")
    print("="*70)
    print(f"数据目录:  {args.data_dir}")
    print(f"ROI尺寸:   {args.roi_size}×{args.roi_size}")
    print(f"过滤阈值:  >{args.max_size}px")
    print("="*70)

    # 检查目录
    if not os.path.exists(args.data_dir):
        print(f"\n❌ 错误: 数据目录不存在: {args.data_dir}")
        print(f"\n💡 如果数据集还未准备，这个工具可以基于经验数据给出估计：")
        print(f"\n典型Motion-guided ROI160场景的经验数据:")
        estimate_from_experience(args.max_size)
        return

    # 分析实际数据
    stats = estimate_filtering_impact(args.data_dir, args.roi_size, args.max_size)
    kept, filtered, total = print_statistics(stats, args.max_size)

    # 建议
    print(f"\n💡 建议:")
    filter_rate = filtered / total * 100 if total > 0 else 0

    if filter_rate < 1:
        print(f"  ✅ 过滤率极低({filter_rate:.1f}%)，数据质量很好！")
        print(f"  ✅ 可以放心使用 {args.max_size}px 阈值过滤")
    elif filter_rate < 5:
        print(f"  ✅ 过滤率正常({filter_rate:.1f}%)，符合Motion-guided预期")
        print(f"  ✅ 建议使用 {args.max_size}px 阈值过滤")
    elif filter_rate < 10:
        print(f"  ⚠️  过滤率偏高({filter_rate:.1f}%)，需要检查数据来源")
        print(f"  💡 建议: 检查ROI切片策略，确认是否真的是Motion-guided")
    else:
        print(f"  ❌ 过滤率过高({filter_rate:.1f}%)，数据可能有问题！")
        print(f"  💡 可能原因:")
        print(f"     1. 混入了非Motion-guided数据（全图裁剪？）")
        print(f"     2. ROI切片时选择了过近的距离")
        print(f"     3. 包含了多样性数据集（应该分开处理）")
        print(f"  💡 建议: 重新检查数据准备流程")


def estimate_from_experience(max_size):
    """基于经验数据给出估计"""
    print(f"\n{'='*70}")
    print("基于经验的过滤影响估计")
    print(f"{'='*70}\n")

    print(f"📊 典型Motion-guided ROI160场景:")
    print(f"  场景: 从全图1920×1080中，以motion blob中心截取160×160 ROI")
    print(f"  目标: 远距离无人机，原图中约8-32像素")
    print()

    print(f"📏 预期目标尺寸分布:")
    print(f"  8-16px:   ~40%  (极小目标)")
    print(f"  16-32px:  ~45%  (典型目标)")
    print(f"  32-48px:  ~10%  (稍大目标)")
    print(f"  48-64px:   ~3%  (较大目标)")
    print(f"  64-80px:   ~1%  (接近阈值)")
    print(f"  >80px:    ~1%   (超大目标，需要过滤)")
    print()

    print(f"✅ 预期过滤影响 (阈值>{max_size}px):")
    print(f"  过滤图片:  ~1-2%  (约10-50张/5000张)")
    print(f"  保留图片:  ~98-99%")
    print()

    print(f"💡 如果实际过滤率 >5%:")
    print(f"  说明数据集中混入了:")
    print(f"  - 非Motion-guided数据 (随机裁剪/固定ROI)")
    print(f"  - 多样性数据集 (近距离拍摄)")
    print(f"  - 手动标注的全图数据")
    print()

    print(f"🎯 建议:")
    print(f"  1. 先运行实际评估: python tools/estimate_filtering_impact.py --data-dir <your_data>")
    print(f"  2. 如果过滤率<5%: 直接使用80px阈值过滤")
    print(f"  3. 如果过滤率>5%: 检查数据来源，可能需要分层处理")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
