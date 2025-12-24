#!/bin/bash

# 设置调试模式（显示执行的命令）
set -x

# 定义路径变量
Variant_Path="/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation"
Data_Path="/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12"

# 检查关键目录是否存在
echo "[DEBUG] 检查目录是否存在..."
if [ ! -d "$Variant_Path" ]; then
    echo "[ERROR] Variant目录不存在: $Variant_Path" >&2
    exit 1
fi

if [ ! -d "$Data_Path" ]; then
    echo "[ERROR] Data目录不存在: $Data_Path" >&2
    exit 1
fi

# 处理第一个JSON文件
input_file1="${Variant_Path}/mmk12_train_think_steps_gpto3.json"
echo "[INFO] 正在处理文件: $input_file1"
if [ ! -f "$input_file1" ]; then
    echo "[ERROR] 输入文件不存在: $input_file1" >&2
    exit 1
fi

if ! python convert_id_key_variant.py "$input_file1"; then
    echo "[ERROR] 处理文件失败: $input_file1" >&2
    exit 1
fi

# 处理第二个JSON文件
input_file2="${Variant_Path}/mmk12_train_variants_gpto3.json"
echo "[INFO] 正在处理文件: $input_file2"
if [ ! -f "$input_file2" ]; then
    echo "[ERROR] 输入文件不存在: $input_file2" >&2
    exit 1
fi

if ! python convert_id_key_variant.py "$input_file2"; then
    echo "[ERROR] 处理文件失败: $input_file2" >&2
    exit 1
fi

# 运行最后的Python脚本
script_path="${Data_Path}/add_difficult.py"
echo "[INFO] 正在运行脚本: $script_path"
if [ ! -f "$script_path" ]; then
    echo "[ERROR] Python脚本不存在: $script_path" >&2
    exit 1
fi

if ! python "$script_path"; then
    echo "[ERROR] 执行脚本失败: $script_path" >&2
    exit 1
fi

echo "[SUCCESS] 所有任务完成!"
exit 0