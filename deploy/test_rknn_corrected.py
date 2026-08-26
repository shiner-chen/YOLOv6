#!/usr/bin/env python3
"""
修正版检测脚本 - 使用正确的 dist2bbox decode

问题：之前使用了错误的 decode 方法（中心点+宽高）
正确：YOLOv6 输出是 distance (ltrb) 格式
"""
import os
import cv2
import time
import numpy as np
from rknnlite.api import RKNNLite

RKNN_MODEL = 'yolov6n_roi160_p2p3_bs4_int8.rknn'
TEST_DIR = 'test_images'
OUTPUT_DIR = 'demo_results_corrected'
IMG_SIZE = 160
CONF_THRESH = 0.35

os.makedirs(OUTPUT_DIR, exist_ok=True)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def generate_anchors(h, w, stride):
    """生成 anchor points"""
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    anchors = np.stack([x, y], axis=-1).reshape(-1, 2).astype(np.float32)
    # 中心点坐标（像素空间）
    anchors = anchors * stride + stride / 2
    return anchors

def dist2bbox_corrected(distance, anchor_points, stride):
    """
    正确的 YOLOv6 decode 方法

    distance: [N, 4] 格式为 [left, top, right, bottom] 距离（需要乘以 stride）
    anchor_points: [N, 2] anchor 中心点坐标
    stride: 特征图步长

    返回: [N, 4] 格式为 [cx, cy, w, h]
    """
    # Distance 需要乘以 stride 恢复到像素空间
    lt = distance[:, :2] * stride  # [left, top]
    rb = distance[:, 2:] * stride  # [right, bottom]

    # 计算四个边界
    x1y1 = anchor_points - lt  # 左上角
    x2y2 = anchor_points + rb  # 右下角

    # 转换为中心点+宽高
    cx = (x1y1[:, 0] + x2y2[:, 0]) / 2
    cy = (x1y1[:, 1] + x2y2[:, 1]) / 2
    w = x2y2[:, 0] - x1y1[:, 0]
    h = x2y2[:, 1] - x1y1[:, 1]

    return np.stack([cx, cy, w, h], axis=1)

def decode_boxes_corrected(reg_output, stride, h, w):
    """
    使用正确的 dist2bbox 解码
    """
    batch_size = reg_output.shape[0]
    # [B, 4, H, W] -> [B, H, W, 4] -> [B, H*W, 4]
    reg_output = reg_output.transpose(0, 2, 3, 1).reshape(batch_size, -1, 4)

    # 生成 anchor points
    anchors = generate_anchors(h, w, stride)

    # 对每个 batch 解码
    boxes_list = []
    for b in range(batch_size):
        boxes = dist2bbox_corrected(reg_output[b], anchors, stride)  # 传入 stride
        boxes_list.append(boxes)

    return np.stack(boxes_list, axis=0)

def postprocess_single_target(outputs, img_w, img_h, conf_thresh):
    """单目标优化版后处理"""
    reg_s0, cls_s0, reg_s1, cls_s1 = outputs

    # 解码 boxes（修正后的方法）
    boxes_p2 = decode_boxes_corrected(reg_s0[0:1], stride=4, h=40, w=40)[0]
    boxes_p3 = decode_boxes_corrected(reg_s1[0:1], stride=8, h=20, w=20)[0]
    boxes = np.concatenate([boxes_p2, boxes_p3], axis=0)

    # Sigmoid 置信度
    cls_p2 = sigmoid(cls_s0[0:1].transpose(0, 2, 3, 1).reshape(1, -1, 1))[0, :, 0]
    cls_p3 = sigmoid(cls_s1[0:1].transpose(0, 2, 3, 1).reshape(1, -1, 1))[0, :, 0]
    scores = np.concatenate([cls_p2, cls_p3], axis=0)

    # 找最高置信度
    max_idx = scores.argmax()
    max_conf = scores[max_idx]

    if max_conf < conf_thresh:
        return None

    # 转换到原图坐标
    scale_x = img_w / IMG_SIZE
    scale_y = img_h / IMG_SIZE

    cx, cy, w, h = boxes[max_idx]
    x1 = (cx - w / 2) * scale_x
    y1 = (cy - h / 2) * scale_y
    x2 = (cx + w / 2) * scale_x
    y2 = (cy + h / 2) * scale_y

    return [x1, y1, x2, y2, max_conf]

# 选择测试图片
selected_images = [
    'phantom03_1332__o00_s01_center.jpg',
    'phantom03_0830__o00_s01_edge.jpg',
    'phantom05_0788__o00_s01_small_offset.jpg',
    'phantom03_1540__o00_s02_partial.jpg',
]

print('='*80)
print('修正版检测 - 正确的 dist2bbox decode')
print('='*80)
print(f'置信度阈值: {CONF_THRESH}')
print('修正内容: 使用 distance (ltrb) 格式解码\n')

# 加载模型
rknn = RKNNLite()
rknn.load_rknn(RKNN_MODEL)
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)

# 预热
dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
for _ in range(3):
    rknn.inference(inputs=[dummy])

print('检测结果:\n')

postprocess_times = []
detected_count = 0

for img_name in selected_images:
    img_path = os.path.join(TEST_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]

    # 预处理
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    batch_input = np.expand_dims(img_rgb, axis=0)

    # 推理
    outputs = rknn.inference(inputs=[batch_input])

    # 后处理
    t_start = time.perf_counter()
    detection = postprocess_single_target(outputs, w, h, CONF_THRESH)
    t_post = (time.perf_counter() - t_start) * 1000

    postprocess_times.append(t_post)

    # 绘制结果
    result_img = img.copy()
    if detection is not None:
        detected_count += 1
        x1, y1, x2, y2, conf = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # 绘制框
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # 绘制标签
        label = f'UAV {conf:.3f}'
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y1_label = max(y1, label_size[1] + 10)

        cv2.rectangle(result_img, (x1, y1_label - label_size[1] - 10),
                     (x1 + label_size[0] + 10, y1_label + 5), (0, 255, 0), -1)
        cv2.putText(result_img, label, (x1 + 5, y1_label),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # 计算框尺寸
        box_w = x2 - x1
        box_h = y2 - y1

        print(f'✓ {img_name}')
        print(f'  置信度: {conf:.3f}')
        print(f'  位置: ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})')
        print(f'  尺寸: {box_w:.0f} x {box_h:.0f} 像素')
        print(f'  后处理: {t_post:.2f} ms\n')
    else:
        print(f'✗ {img_name}: 未检测到\n')

    # 保存
    output_path = os.path.join(OUTPUT_DIR, f'corrected_{img_name}')
    cv2.imwrite(output_path, result_img)

rknn.release()

avg_post = np.mean(postprocess_times)

print('='*80)
print('性能统计:')
print('='*80)
print(f'平均后处理时间: {avg_post:.2f} ms')
print(f'检测成功率: {detected_count}/{len(selected_images)}')
print(f'结果保存在: {OUTPUT_DIR}/')
print('='*80)
