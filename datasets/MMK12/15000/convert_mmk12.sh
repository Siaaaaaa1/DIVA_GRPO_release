#!/bin/bash

# ================= 可配置变量 =================
# 调试模式开关（true/false）
DEBUG_MODE=true

# 数据集路径
VARIANT_DIR="/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/merge"
DATA_DIR="/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12"

# 输入文件名
VARIANT_FILE1="mmk12_train_think_steps_first_qwen_merge.json"
VARIANT_FILE2="mmk12_train_variants_qwen.json"

# 脚本路径
CONVERT_SCRIPT="${DATA_DIR}/convert_id_key_variant.py"
FINAL_SCRIPT="${DATA_DIR}/add_difficult.py"
# =================================================

# 设置调试模式
if [ "$DEBUG_MODE" = true ]; then
    set -x
fi

# 检查关键目录是否存在
echo "[DEBUG] 检查目录是否存在..."
if [ ! -d "$VARIANT_DIR" ]; then
    echo "[ERROR] Variant目录不存在: $VARIANT_DIR" >&2
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "[ERROR] Data目录不存在: $DATA_DIR" >&2
    exit 1
fi

# 处理第一个JSON文件
input_file1="${VARIANT_DIR}/${VARIANT_FILE1}"
echo "[INFO] 正在处理文件: $input_file1"
if [ ! -f "$input_file1" ]; then
    echo "[ERROR] 输入文件不存在: $input_file1" >&2
    exit 1
fi

if ! python "$CONVERT_SCRIPT" "$input_file1"; then
    echo "[ERROR] 处理文件失败: $input_file1" >&2
    exit 1
fi

# 处理第二个JSON文件
input_file2="${VARIANT_DIR}/${VARIANT_FILE2}"
echo "[INFO] 正在处理文件: $input_file2"
if [ ! -f "$input_file2" ]; then
    echo "[ERROR] 输入文件不存在: $input_file2" >&2
    exit 1
fi

if ! python "$CONVERT_SCRIPT" "$input_file2"; then
    echo "[ERROR] 处理文件失败: $input_file2" >&2
    exit 1
fi

# 运行最后的Python脚本
echo "[INFO] 正在运行脚本: $FINAL_SCRIPT"
if [ ! -f "$FINAL_SCRIPT" ]; then
    echo "[ERROR] Python脚本不存在: $FINAL_SCRIPT" >&2
    exit 1
fi

if ! python "$FINAL_SCRIPT"; then
    echo "[ERROR] 执行脚本失败: $FINAL_SCRIPT" >&2
    exit 1
fi

echo "[SUCCESS] 所有任务完成!"
exit 0
