#!/usr/bin/env python3
"""
分析不同ROI尺寸下的目标占比问题
"""

def analyze_target_ratio(roi_size, target_size_range):
    """
    分析目标在ROI中的占比

    Args:
        roi_size: ROI尺寸 (160, 320, 640)
        target_size_range: 目标尺寸范围 (min, max) in pixels
    """
    print(f"\n{'='*70}")
    print(f"ROI Size: {roi_size}×{roi_size}")
    print(f"{'='*70}")

    for target_size in target_size_range:
        area_ratio = (target_size ** 2) / (roi_size ** 2)
        edge_ratio = target_size / roi_size

        print(f"\n目标尺寸: {target_size}×{target_size} px")
        print(f"  面积占比: {area_ratio*100:.1f}%")
        print(f"  边长占比: {edge_ratio*100:.1f}%")

        # 判断是否需要剔除
        if area_ratio > 0.5:  # 面积超过50%
            status = "❌ 建议剔除 - 目标过大，缺乏上下文"
        elif area_ratio > 0.25:  # 面积超过25%
            status = "⚠️  考虑剔除 - 目标较大，可能影响检测"
        elif edge_ratio > 0.8:  # 边长超过80%
            status = "⚠️  考虑剔除 - 接近边界，容易截断"
        else:
            status = "✅ 保留 - 合适的目标尺寸"

        print(f"  决策: {status}")

print("="*70)
print("ROI数据集目标尺寸分析")
print("="*70)
print("\n假设：Anti-UAV目标在原图中约16-32像素")
print("Motion-guided ROI: 以blob中心截取，目标相对居中")

# ROI 640 分析
print("\n\n## 场景1: ROI640 (从全图1920×1080裁剪)")
analyze_target_ratio(640, [16, 20, 32, 48, 64, 100, 150, 200])

# ROI 320 分析
print("\n\n## 场景2: ROI320 (从全图裁剪)")
analyze_target_ratio(320, [12, 16, 20, 32, 48, 64, 100, 120, 150])

# ROI 160 分析
print("\n\n## 场景3: ROI160 (从全图裁剪)")
analyze_target_ratio(160, [8, 12, 16, 20, 32, 48, 64, 80, 100, 120])

print("\n\n" + "="*70)
print("结论与建议")
print("="*70)

conclusions = """
1. ROI640:
   - 典型目标(16-32px): 占0.06-0.25%，非常小
   - 100px以上: 占2.4%+，开始较大
   - 建议: 剔除100px+的图片（面积>2.4%）

2. ROI320:
   - 典型目标(16-32px): 占0.25-1%，较小
   - 64px: 占4%，边长20%
   - 100px+: 占9.7%+，显著偏大
   - 建议: 剔除80px+的图片（面积>6.25%，边长>25%）

3. ROI160 ← 当前讨论重点:
   - 典型目标(16-32px): 占1-4%，合适
   - 48px: 占9%，边长30% ✅ 可以保留
   - 64px: 占16%，边长40% ⚠️  边界情况
   - 80px: 占25%，边长50% ❌ 建议剔除
   - 100px+: 占39%+，边长62%+ ❌ 必须剔除

关键阈值建议：
  ROI640: 剔除 > 100px 的目标
  ROI320: 剔除 > 80px 的目标
  ROI160: 剔除 > 64px 的目标 (或 > 80px 更宽松)

理由：
  1. 面积占比 > 25%: 目标太大，背景上下文不足
  2. 边长占比 > 50%: 容易边缘截断，检测头感受野不足
  3. P2/P3检测头设计: 适合4-40px，超过64px偏离设计目标
"""

print(conclusions)
