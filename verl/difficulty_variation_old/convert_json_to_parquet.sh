#!/bin/bash

# 脚本：run_python_scripts.sh
# 功能：顺序执行多个Python脚本
# 作者：<你的名字>
# 日期：2025-07-29

# 设置工作目录
WORKSPACE="/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting"

# 定义日志文件路径
LOG_FILE="${WORKSPACE}/script_execution.log"
echo "脚本执行日志 - $(date)" > $LOG_FILE

# 定义要执行的Python脚本数组
SCRIPTS=(
    # "verl/difficulty_variation/merge_variant.py"
    "verl/difficulty_variation/convert_id_key_variant.py"
    "datasets/MMK12/add_difficult.py"
)

# 函数：执行Python脚本并记录结果
execute_python_script() {
    local script_path="$1"
    echo "正在执行: ${script_path}" | tee -a $LOG_FILE
    echo "开始时间: $(date)" | tee -a $LOG_FILE
    
    if python "${WORKSPACE}/${script_path}"; then
        echo "执行成功: ${script_path}" | tee -a $LOG_FILE
    else
        echo "执行失败: ${script_path}" | tee -a $LOG_FILE
        echo "错误详情: 请检查 ${LOG_FILE} 获取更多信息" | tee -a $LOG_FILE
        exit 1
    fi
    
    echo "结束时间: $(date)" | tee -a $LOG_FILE
    echo "----------------------------------------" | tee -a $LOG_FILE
}

# 主执行逻辑
for script in "${SCRIPTS[@]}"; do
    execute_python_script "$script"
done

echo "所有脚本执行完成" | tee -a $LOG_FILE
exit 0