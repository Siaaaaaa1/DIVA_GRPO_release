#!/bin/bash

set -x

idle_count=0
max_idle_count=3

is_gpu_idle() {
    # 获取 GPU 使用率
    gpu_utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)

    # 去掉输出的任何空格或换行
    gpu_utilization=$(echo "$gpu_utilization" | tr -d '[:space:]')

    # 如果 GPU 使用率为空，认为是空闲的
    if [[ -z "$gpu_utilization" ]]; then
        return 0  # GPU 空闲
    fi

    # 如果输出包含 1-9 的任何一个数字，则认为 GPU 正在使用
    if [[ "$gpu_utilization" =~ [1-9] ]]; then
        return 1  # GPU 正在使用
    fi

    # 否则（只包含 0 或没有数字），认为 GPU 空闲
    return 0  # GPU 空闲
}

# 使用示例
while true; do
   if is_gpu_idle; then
       ((idle_count++))  # 增加空闲计数
       echo "GPU 空闲 ($idle_count/$max_idle_count)，等待继续..."

       # 如果连续 3 次检查空闲，开始运行
       if [ "$idle_count" -ge "$max_idle_count" ]; then
           echo "GPU 连续空闲 3 次，开始训练..."
           break
       fi
   else
       echo "GPU 正在使用中，等待空闲..."
       idle_count=0  # 如果不是空闲，重置计数器
   fi
   sleep 10  # 等待 10 秒钟后再次检查
done

SCRIPT_NAME=$(basename "$0" .sh)
EXP_NAME="${SCRIPT_NAME}_$(date +%m%d-%H%M)"

export PYTHONUNBUFFERED=1
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_PROJECT=Easy-R1
export WANDB_API_KEY="e0a53bc21f8007dfb8dc043a7a44a591a9235f7f"
export WANDB_RUN_NAME=${EXP_NAME}--$(date +%Y-%m-%d-%H-%M-%S)
wandb login $WANDB_API_KEY


MODEL_PATH="/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1/models/Qwen2.5-VL-7B-Instruct" # replace it with your local file path
LOG_PATH="logs/"

if [ ! -d "$LOG_PATH" ]; then
    mkdir -p "$LOG_PATH"
fi

# 定义路径和SCRIPT_NAME
All_Log_Path="/mmu_cd_ssd/zhangzhenyu06/workspace/Rebuttal/all_log_path"
All_Data_Path="/mmu_cd_ssd/zhangzhenyu06/workspace/Rebuttal/all_data_path"
All_Log_Path_File="${SCRIPT_NAME}_$(date +%m%d-%H%M)"
mkdir -p "${All_Log_Path}/${All_Log_Path_File}"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/mmu_cd_ssd/zhangzhenyu06/workspace/Rebuttal/EasyR1/datasets/MMK12/R1-Share-VL/GRPO_45117_random5000.parquet \
    data.val_files=/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/data/MMK12_test.parquet \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.n_gpus_per_node=8 \
    worker.rollout.gpu_memory_utilization=0.6 \
    trainer.experiment_name=${EXP_NAME} \
    trainer.total_epochs=15 \
    trainer.val_freq=1 \
    trainer.val_generations_to_log=1 \
    trainer.save_limit=10 \
    trainer.save_freq=1 \
    trainer.val_before_train=false \
    trainer.find_last_checkpoint=false \
    data.rollout_batch_size=512 \
    worker.actor.global_batch_size=128 \
    worker.rollout.n=5 \
    trainer.All_Log_Path="${All_Log_Path}/${All_Log_Path_File}" \
    data.max_pixels=1048576 \
    worker.rollout.disable_tqdm=false \
    data.shuffle=true \
    algorithm.disable_kl=true \
    algorithm.use_kl_loss=false \
    trainer.val_before_train=true \
    trainer.DIVA_GRPO=false \
    trainer.Share_VL=false \
    trainer.Variant_Num=1 \
    trainer.Dataset_Mode=None \
    trainer.Save_Data=false \
    algorithm.weight_mode="${SCRIPT_NAME}$" \
    trainer.score_ranges='[[0.0,0.05],[0.05,0.3],[0.3,0.5],[0.5,0.7],[0.7,1.0]]' \
    trainer.difficulty_changes='[2,1,0,-1,-2]' \
    trainer.weighted_advantage_k=0.1 \
    worker.reward.reward_save_path="${All_Log_Path}/${All_Log_Path_File}" \
    2>&1 | tee -a "${LOG_PATH}/${EXP_NAME}_training_log.log"
wait