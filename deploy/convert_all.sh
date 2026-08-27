#!/bin/bash
"""
YOLOv6 RKNN 一键转换脚本

Usage:
    ./convert_all.sh yolov6n_roi320.pt 320 roi320 1,4

参数：
    $1: PyTorch权重文件路径
    $2: 输入尺寸
    $3: 输出文件前缀
    $4: batch size列表（逗号分隔）
"""

set -e  # 遇到错误立即退出

# 检查参数
if [ $# -ne 4 ]; then
    echo "Usage: $0 <weights.pt> <img_size> <output_prefix> <batch_sizes>"
    echo "Example: $0 yolov6n_roi320.pt 320 roi320 1,4"
    exit 1
fi

WEIGHTS=$1
IMG_SIZE=$2
PREFIX=$3
BATCH_SIZES=$4

# 路径配置
SCRIPT_DIR="/data/workdir/et-yolov6"
WORK_DIR="/home/chenx/workdir/et-yolov6n"
PYTHON="/home/chenx/rknn-env/bin/python3"

# 数据集路径自动推断
if [ $IMG_SIZE -eq 160 ]; then
    DATASET="${SCRIPT_DIR}/rknn_calibration_roi640_list.txt"
elif [ $IMG_SIZE -eq 320 ]; then
    DATASET="${SCRIPT_DIR}/rknn_calibration_roi320_list.txt"
elif [ $IMG_SIZE -eq 640 ]; then
    DATASET="${SCRIPT_DIR}/rknn_calibration_roi640_list.txt"
else
    echo "警告: 未找到ROI${IMG_SIZE}的校验数据集，使用roi640"
    DATASET="${SCRIPT_DIR}/rknn_calibration_roi640_list.txt"
fi

# 输出文件路径
ONNX_OUTPUT="${SCRIPT_DIR}/${PREFIX}_split.onnx"

echo "=================================================="
echo "YOLOv6 RKNN 一键转换"
echo "=================================================="
echo "权重文件:     $WEIGHTS"
echo "输入尺寸:     ${IMG_SIZE}x${IMG_SIZE}"
echo "输出前缀:     $PREFIX"
echo "Batch sizes:  $BATCH_SIZES"
echo "校验数据集:   $DATASET"
echo "=================================================="
echo ""

# 步骤1: 导出ONNX
echo "步骤 1/3: 导出ONNX模型..."
cd $WORK_DIR
$PYTHON ${SCRIPT_DIR}/export_yolov6_rknn.py \
    --weights $WEIGHTS \
    --img-size $IMG_SIZE \
    --output $ONNX_OUTPUT

if [ $? -ne 0 ]; then
    echo "✗ ONNX导出失败"
    exit 1
fi
echo ""

# 步骤2: 转换RKNN (多个batch size)
echo "步骤 2/3: 转换RKNN模型..."
IFS=',' read -ra BS_ARRAY <<< "$BATCH_SIZES"
for BS in "${BS_ARRAY[@]}"; do
    RKNN_OUTPUT="${SCRIPT_DIR}/${PREFIX}_bs${BS}_int8.rknn"

    echo ""
    echo "  → 转换 batch_size=$BS ..."
    $PYTHON ${SCRIPT_DIR}/convert_yolov6_rknn.py \
        --onnx $ONNX_OUTPUT \
        --output $RKNN_OUTPUT \
        --batch-size $BS \
        --dataset $DATASET \
        --img-size $IMG_SIZE

    if [ $? -ne 0 ]; then
        echo "✗ RKNN转换失败 (batch_size=$BS)"
        exit 1
    fi
done
echo ""

# 步骤3: 显示生成的文件
echo "步骤 3/3: 生成文件汇总"
echo "=================================================="
echo "ONNX模型:"
ls -lh $ONNX_OUTPUT
echo ""
echo "RKNN模型:"
for BS in "${BS_ARRAY[@]}"; do
    RKNN_OUTPUT="${SCRIPT_DIR}/${PREFIX}_bs${BS}_int8.rknn"
    ls -lh $RKNN_OUTPUT
done
echo "=================================================="
echo ""

echo "✓ 全部转换完成！"
echo ""
echo "下一步："
echo "  1. 上传RKNN模型到RK3588设备"
echo "  2. 使用 benchmark_yolov6_rknn.py 测试性能"
echo ""
echo "性能测试命令示例:"
echo "  sshpass -p firefly scp ${PREFIX}_bs*.rknn benchmark_yolov6_rknn.py \\"
echo "      firefly@192.168.1.34:/home/firefly/workspace/test/"
echo ""
echo "  sshpass -p firefly ssh firefly@192.168.1.34 \\"
echo "      \"cd /home/firefly/workspace/test && \\"
echo "      /home/firefly/rknn-venv/bin/python3 benchmark_yolov6_rknn.py \\"
echo "          --model ${PREFIX}_bs1_int8.rknn ${PREFIX}_bs4_int8.rknn \\"
echo "          --img-size $IMG_SIZE $IMG_SIZE \\"
echo "          --batch-size 1 4\""
