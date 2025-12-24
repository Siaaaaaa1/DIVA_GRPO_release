#!/bin/bash

set -x

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
    data.train_files=/mmu_cd_ssd/zhangzhenyu06/workspace/Rebuttal/EasyR1/datasets/MMK12/data/MMK12_train_thinkq_variantg.parquet \
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
    trainer.find_last_checkpoint=false \
    data.rollout_batch_size=1024 \
    worker.rollout.n=5 \
    worker.actor.global_batch_size=256 \
    trainer.All_Log_Path="${All_Log_Path}/${SCRIPT_NAME}" \
    data.max_pixels=1048576 \
    worker.rollout.disable_tqdm=false \
    data.shuffle=false \
    algorithm.disable_kl=true \
    algorithm.use_kl_loss=false \
    trainer.val_before_train=true \
    trainer.DIVA_GRPO=true \
    trainer.Share_VL=false \
    trainer.Variant_Num=1 \
    trainer.Dataset_Mode=None \
    trainer.Save_Data=false \
    algorithm.weight_mode="${SCRIPT_NAME}$" \
    trainer.score_ranges='[[0.0,0.2],[0.2,0.4],[0.4,0.6],[0.6,0.8],[0.8,1.0]]' \
    trainer.difficulty_changes='[2,1,0,-1,-2]' \
    trainer.weighted_advantage_k=0.1 \
    trainer.DIVA_warmup=true \
    worker.reward.reward_save_path="${All_Log_Path}/${SCRIPT_NAME}" \
    2>&1 | tee -a "${LOG_PATH}/${EXP_NAME}_training_log.log"
wait