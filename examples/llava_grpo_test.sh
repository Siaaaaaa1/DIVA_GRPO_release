#!/bin/bash

set -x

MODEL_PATH="/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1/models/Qwen2.5-VL-7B-Instruct" # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    worker.actor.model.model_path=${MODEL_PATH}
