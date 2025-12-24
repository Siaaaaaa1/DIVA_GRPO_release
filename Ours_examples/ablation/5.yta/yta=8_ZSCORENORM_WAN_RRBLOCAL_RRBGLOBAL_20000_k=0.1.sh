#!/bin/bash

set -x

# 定义一个变量用于计数连续的空闲检查次数
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
mkdir -p "${All_Log_Path}/${SCRIPT_NAME}"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/mmu_cd_ssd/zhangzhenyu06/workspace/Rebuttal/EasyR1/datasets/DIVA_GRPO/GRPO_18000.parquet \
    data.val_files=/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/data/MMK12_test.parquet \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.n_gpus_per_node=8 \
    worker.rollout.gpu_memory_utilization=0.6 \
    trainer.experiment_name=${EXP_NAME} \
    trainer.total_epochs=5 \
    trainer.val_freq=1 \
    trainer.val_generations_to_log=1 \
    trainer.save_limit=10 \
    trainer.save_freq=1 \
    trainer.val_before_train=false \
    trainer.find_last_checkpoint=false \
    data.rollout_batch_size=512 \
    worker.actor.global_batch_size=256 \
    worker.rollout.n=5 \
    trainer.Variant_Num=1 \
    trainer.All_Log_Path="${All_Log_Path}/${SCRIPT_NAME}" \
    data.max_pixels=1048576 \
    worker.rollout.disable_tqdm=false \
    data.shuffle=true \
    algorithm.disable_kl=true \
    algorithm.use_kl_loss=false \
    trainer.val_before_train=true \
    trainer.DIVA_GRPO=true \
    trainer.Share_VL=false \
    trainer.Dataset_Mode=None \
    trainer.Save_Data=false \
    algorithm.weight_mode="${SCRIPT_NAME}$" \
    trainer.score_ranges='[[0.0,0.2],[0.2,0.4],[0.4,0.6],[0.6,0.8],[0.8,1.0]]' \
    trainer.difficulty_changes='[4,2,0,-2,-4]' \
    trainer.weighted_advantage_k=0.1 \
    worker.reward.reward_save_path="${All_Log_Path}/${All_Log_Path_File}" \
    2>&1 | tee -a "${LOG_PATH}/${EXP_NAME}_training_log.log"
wait

### 算法配置： 在weight_mode处设置如下的字符串，使用_拼起来，每一行至多一个，或者None
### 自动获取sh文件的名称作为weight_mode的配置
### KLCOV
### WBN
### RMSNORM / MINMAXNORM / ZSCORENORM
### WAN
### RRBLOCAL
### RRBGLOBAL

### 只在trainer和algorithm中加参数
### trainer.Difficulty_Adaptation=true 
### trainer.Variant_Num=1
### trainer.Full_Vector_Data_Path
### trainer.Diffculty_Updates_Path
### algorithm.Global_Local = true
### algorithm.weight_mode = "str"
### 
    # trainer.Difficulty_Adaptation=true \
    # trainer.Variant_Num=1 \
    # trainer.Difficulty_Change=true \
    # trainer.Full_Vector_Data_Path="/mmu_cd_ssd/zhangzhenyu06/workspace/logging_0830/full_vector_${EXP_NAME}.json" \
    # trainer.Diffculty_Updates_Path="/mmu_cd_ssd/zhangzhenyu06/workspace/logging_0830/difficulty_updates_${EXP_NAME}.jsonl" \
    # algorithm.Adjust_Low_Reward_Local=true \
    # algorithm.Adjust_Low_Reward_Global=true \
    # algorithm.weight_mode="weightafter1-5_zscore_norm" \
    # trainer.Dataset_Mode="only_text_thinking" \

    # trainer.save_freq=1 \
#only_text_thinking
    # data.rollout_batch_size=16 \
    # worker.actor.global_batch_size=8 \
    # worker.actor.micro_batch_size_per_device_for_update=1 \
    # worker.actor.micro_batch_size_per_device_for_experience=1 
    
#用于测试
