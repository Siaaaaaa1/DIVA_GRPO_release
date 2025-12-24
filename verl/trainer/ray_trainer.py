# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import FunctionRewardManager
from .config import PPOConfig
from ..utils.diva_utils import (
    calculate_new_difficulty, 
    calculate_difficulty_changes, 
    normalize_advantages, 
    minmax_normalize_advantages, 
    weighted_advantage, 
    append_to_json_log, 
    save_full_vectors_to_json,
    adjust_low_reward_advantages,
    save_full_vectors_to_json_origin_grpo
)
from .core_algos import (
    compute_grpo_outcome_advantage, 
    compute_grpo_outcome_advantage_kl_cov,
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from collections import defaultdict


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create ray resource pools for distributed training."""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards."""
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(
    data: DataProto, 
    adv_estimator: AdvantageEstimator, 
    gamma: float = 1.0, 
    lam: float = 1.0, 
    all_log_path = ""):
    """Compute advantage estimates for policy optimization."""
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns

    token_level_rewards = data.batch["token_level_rewards"]
    vector_data = {
        "advantages": advantages.cpu().numpy()[:, 0],  # 取第一列
        "token_rewards": token_level_rewards.cpu().numpy().sum(axis=-1),  # 计算token rewards总和
    }
    log_path = os.path.join(all_log_path,'full_vector.log')
    save_full_vectors_to_json_origin_grpo(data, vector_data, log_path) ## path

    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[FunctionRewardManager] = None,
        val_reward_fn: Optional[FunctionRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        # TODO: 如果开启Difficulty_Adaptation开关，根据rollout次数进行修正，与之前代码有区别！
        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor, rollout and ref
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    # TODO: 将训练数据集保存为与检查点同目录的parquet文件（仅在GPU 0上执行）
    def _save_dataset(self) -> None:
        
        try:
            import torch
            # 检查当前是否在GPU 0上，如果不是则直接返回
            if torch.cuda.is_available() and torch.cuda.current_device() != 0:
                print("当前不在GPU 0上，跳过数据集保存")
                return

            import pyarrow.parquet as pq
            import pandas as pd
            
            # 从dataloader获取数据集
            dataset = self.train_dataloader.dataset
            
            # 处理不同的数据集类型
            if hasattr(dataset, 'dataset') and isinstance(dataset.dataset, (Dataset, DatasetDict)):  
                # 处理RLHFDataset类型（内部包含HuggingFace数据集）
                hf_dataset = dataset.dataset
                if isinstance(hf_dataset, DatasetDict):
                    # 如果是DatasetDict，默认取第一个split
                    hf_dataset = next(iter(hf_dataset.values()))
                df = hf_dataset.to_pandas()
                
            elif hasattr(dataset, 'to_pandas'):  # 数据集自带to_pandas方法
                df = dataset.to_pandas()
                
            elif hasattr(dataset, '__array__'):  # numpy数组类型
                df = pd.DataFrame(dataset)
                
            elif hasattr(dataset, '__iter__'):  # 可迭代数据集
                # 先转换为列表再转为DataFrame
                data_list = list(dataset)
                if data_list and isinstance(data_list[0], dict):  # 元素是字典类型
                    df = pd.DataFrame(data_list)
                else:  # 其他可迭代类型
                    df = pd.DataFrame({'data': data_list})
                    
            else:
                raise ValueError("不支持的数据集类型 - 无法转换为DataFrame")
                
            # 创建检查点目录（如果不存在）
            folder_path = self.config.trainer.save_checkpoint_path
            os.makedirs(folder_path, exist_ok=True)
            # 定义parquet文件路径
            parquet_path = os.path.join(folder_path, f"MMK12_Adapter.parquet")
            
            # 保存为parquet文件
            df.to_parquet(parquet_path)
            
            print(f"成功保存数据集到 {parquet_path}")
            
        except Exception as e:
            print(f"保存数据集失败: {str(e)}")
            raise

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        #TODO: 添加保存数据相关设置
        if self.config.trainer.Save_Data == True:
            self._save_dataset()
            
        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _maybe_log_val_generations(
        self, inputs: list[str], outputs: list[str], labels: list[str], scores: list[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # store generations
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        print("Finish validation.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics, **val_length_metrics}

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    # def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
    #     # ==================== DEBUG SETUP ====================
    #     import os
    #     import torch
    #     from PIL import Image
    #     import json
    #     from datetime import datetime
        
    #     debug_dir = "/mmu_cd_ssd/zhangzhenyu06/workspace/Rebuttal/Help"
    #     os.makedirs(debug_dir, exist_ok=True)
    #     debug_log_path = os.path.join(debug_dir, "debug_log.txt")
        
    #     def debug_print(msg, data=None):
    #         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    #         with open(debug_log_path, "a") as f:
    #             if data is not None:
    #                 f.write(f"[{timestamp}] {msg}\n")
    #                 f.write(f"  Type: {type(data)}\n")
    #                 if isinstance(data, (torch.Tensor, np.ndarray)):
    #                     f.write(f"  Shape: {data.shape}, Dtype: {data.dtype}, Min: {data.min()}, Max: {data.max()}\n") #, Mean: {data.mean()}
    #                 elif isinstance(data, (list, dict, tuple)):
    #                     f.write(f"  Length: {len(data)}, Value: {str(data)[:50]}\n")
    #                 else:
    #                     f.write(f"  Value: {str(data)[:50]}\n")
    #                 f.write("-" * 80 + "\n")
    #             else:
    #                 f.write(f"[{timestamp}] {msg}\n")
    #                 f.write("-" * 80 + "\n")
        
    #     def save_images_from_batch(batch, step, prefix=""):
    #         """Save images from multi_modal_data if present"""
    #         if "multi_modal_data" in batch.non_tensor_batch:
    #             mm_data = batch.non_tensor_batch["multi_modal_data"]
    #             if "image" in mm_data:
    #                 images = mm_data["image"]
    #                 if not isinstance(images, list):
    #                     images = [images]
                    
    #                 for idx, img in enumerate(images):
    #                     if isinstance(img, (Image.Image, torch.Tensor, np.ndarray)):
    #                         save_path = os.path.join(debug_dir, f"{prefix}_step{step}_img{idx}.png")
    #                         if isinstance(img, torch.Tensor):
    #                             img = img.cpu().numpy()
    #                         if isinstance(img, np.ndarray):
    #                             if img.dtype != np.uint8:
    #                                 img = (img * 255).astype(np.uint8)
    #                             # Handle CHW format
    #                             if img.ndim == 3 and img.shape[0] in [1, 3]:
    #                                 img = img.transpose(1, 2, 0)
    #                             img = Image.fromarray(img.squeeze())
    #                         img.save(save_path)
    #                         debug_print(f"Saved image: {save_path}")
        
    #     debug_print("=" * 50)
    #     debug_print(f"START _make_batch_data - Config: rollout_n={self.config.worker.rollout.n}, "
    #                 f"rollout_batch_size={self.config.data.rollout_batch_size}")
    #     # =====================================================
        
    #     batch = None
    #     all_metrics = defaultdict(list)
    #     num_try_make_batch = 0
    #     print("Start generating batch...")
    #     debug_print("Start generating batch...")
        
    #     while True:
    #         num_try_make_batch += 1
    #         debug_print(f"=== Loop iteration {num_try_make_batch} ===")
            
    #         try:
    #             batch_dict = next(self.data_iterator)
    #             debug_print(f"batch_dict keys: {list(batch_dict.keys())}", batch_dict)
    #         except StopIteration:
    #             self.data_iterator = iter(self.train_dataloader)
    #             batch_dict = next(self.data_iterator)
    #             debug_print("Data iterator exhausted, reinitialized")
            
    #         meta_info = {
    #             "min_pixels": self.config.data.min_pixels,
    #             "max_pixels": self.config.data.max_pixels,
    #             "video_fps": self.config.data.video_fps,
    #         }
            
    #         new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
    #         debug_print("new_batch created", new_batch)
    #         debug_print("new_batch.batch keys", list(new_batch.batch.keys()))
    #         debug_print("new_batch.non_tensor_batch keys", list(new_batch.non_tensor_batch.keys()))
            
    #         new_batch.non_tensor_batch["uid"] = np.array(
    #             [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
    #         )
    #         debug_print("Generated UIDs", new_batch.non_tensor_batch["uid"])
            
    #         # Save images from original batch
    #         save_images_from_batch(new_batch, num_try_make_batch, "original")
            
    #         # pop those keys for generation
    #         gen_batch = new_batch.pop(
    #             batch_keys=["input_ids", "attention_mask", "position_ids"],
    #             non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
    #             meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
    #         )
    #         debug_print("gen_batch created after pop", gen_batch)
            
    #         # generate a batch
    #         gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)
    #         debug_print("gen_batch_output received", gen_batch_output)
            
    #         if self.config.algorithm.adv_estimator == "remax":
    #             debug_print("Using ReMax estimator")
    #             gen_baseline_batch = deepcopy(gen_batch)
    #             gen_baseline_batch.meta_info["temperature"] = 0
    #             gen_baseline_batch.meta_info["n"] = 1
    #             gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)
    #             debug_print("gen_baseline_output received", gen_baseline_output)
                
    #             new_batch = new_batch.union(gen_baseline_output)
    #             debug_print("new_batch after union with baseline", new_batch)
                
    #             reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
    #             debug_print("reward_baseline_tensor computed", reward_baseline_tensor)
                
    #             reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
    #             debug_print("reward_baseline_tensor after sum", reward_baseline_tensor)
                
    #             new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
    #             new_batch.batch["reward_baselines"] = reward_baseline_tensor
    #             debug_print("reward_baselines added to batch", new_batch.batch["reward_baselines"])
                
    #             del gen_baseline_batch, gen_baseline_output
            
    #         #######################################################################
    #         #TODO: 当开启设置时，为每个唯一id生成一个global_uid
    #         if self.config.trainer.DIVA_GRPO:
    #             debug_print("DIVA_GRPO mode enabled")
                
    #             # 获取 id 数组和问题数组
    #             ids = new_batch.non_tensor_batch["id"]
    #             debug_print("Original IDs from batch", ids)
                
    #             id_counts = {} 
    #             for id_val in ids:
    #                 id_counts[id_val] = id_counts.get(id_val, 0) + 1
    #             debug_print("ID counts", id_counts)
                
    #             # 为每个唯一id生成一个global_uid
    #             unique_ids = list(id_counts.keys())
    #             debug_print(f"Found {len(unique_ids)} unique IDs", unique_ids)
                
    #             id_to_uid = {id_val: str(uuid.uuid4()) for id_val in unique_ids}
    #             debug_print("ID to Global UID mapping", id_to_uid)
                
    #             # 创建global_uids数组，确保相同id对应相同global_uid
    #             global_uids = np.array([id_to_uid[id_val] for id_val in ids], dtype=object)
    #             debug_print("Generated global_uids array", global_uids)
                
    #             # 将global_uids添加到batch中
    #             new_batch.non_tensor_batch["global_uid"] = global_uids
    #             debug_print("global_uid added to batch", new_batch.non_tensor_batch["global_uid"])
                
    #             # Save visualization of ID mapping
    #             mapping_info = {
    #                 "iteration": num_try_make_batch,
    #                 "original_ids": ids.tolist() if hasattr(ids, 'tolist') else list(ids),
    #                 "global_uids": global_uids.tolist() if hasattr(global_uids, 'tolist') else list(global_uids),
    #                 "id_counts": id_counts,
    #                 "id_to_uid_mapping": id_to_uid
    #             }
    #             mapping_path = os.path.join(debug_dir, f"id_mapping_step_{num_try_make_batch}.json")
    #             with open(mapping_path, "w") as f:
    #                 json.dump(mapping_info, f, indent=2)
    #             debug_print(f"Saved ID mapping to {mapping_path}")
                
    #         #######################################################################
            
    #         # repeat to align with repeated responses in rollout
    #         debug_print(f"Repeating batch {self.config.worker.rollout.n} times (interleave=True)")
    #         new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
    #         debug_print("new_batch after repeat", new_batch)
            
    #         new_batch = new_batch.union(gen_batch_output)
    #         debug_print("new_batch after union with gen_batch_output", new_batch)
    #         debug_print("new_batch.batch keys after union", list(new_batch.batch.keys()))
    #         debug_print("new_batch.non_tensor_batch keys after union", list(new_batch.non_tensor_batch.keys()))
            
    #         # filter group
    #         if self.config.algorithm.online_filtering:
    #             debug_print("Applying online filtering")
    #             reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
    #             debug_print("Computed reward_tensor", reward_tensor)
    #             debug_print("Received reward_metrics", dict(reward_metrics))
                
    #             new_batch.batch["token_level_scores"] = reward_tensor
    #             debug_print("Added token_level_scores to batch", new_batch.batch["token_level_scores"])
                
    #             for k, v in reward_metrics.items():
    #                 all_metrics[k].extend(v)
                
    #             filter_scores = reward_metrics[self.config.algorithm.filter_key]
    #             debug_print("filter_scores", filter_scores)
    #             debug_print(f"Filter key: {self.config.algorithm.filter_key}, "
    #                     f"Range: ({self.config.algorithm.filter_low}, {self.config.algorithm.filter_high})")
                
    #             uids = new_batch.non_tensor_batch["uid"]
    #             debug_print("UIDs for filtering", uids)
                
    #             uid2scores = defaultdict(list)
    #             for uid, score in zip(uids, filter_scores):
    #                 uid2scores[uid].append(score)
    #             debug_print("uid2scores mapping", dict(uid2scores))
                
    #             uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
    #             debug_print("uid2mean (average scores)", uid2mean)
                
    #             kept_uids = [
    #                 uid
    #                 for uid, avg_score in uid2mean.items()
    #                 if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
    #             ]
    #             debug_print("kept_uids after filtering", kept_uids)
    #             debug_print(f"Kept {len(kept_uids)} out of {len(uid2mean)} UIDs")
                
    #             kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
    #             debug_print("kept_sample_idxs", kept_sample_idxs)
                
    #             if len(kept_sample_idxs) == 0:
    #                 debug_print("ERROR: No samples kept after filtering!")
    #                 raise RuntimeError("No sample is kept after filtering. Please check your data.")
                
    #             new_batch = new_batch[kept_sample_idxs]
    #             debug_print("new_batch after filtering", new_batch)
                
    #             # Save filtered images
    #             save_images_from_batch(new_batch, num_try_make_batch, "filtered")
            
    #         batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
    #         debug_print("Current combined batch", batch)
            
    #         current_batch_size = len(batch) // self.config.worker.rollout.n
    #         rollout_batch_size = self.config.data.rollout_batch_size
            
    #         debug_print(f"Batch size check: current_batch_size={current_batch_size}, "
    #                 f"rollout_batch_size={rollout_batch_size}, "
    #                 f"raw_batch_len={len(batch)}, "
    #                 f"rollout_n={self.config.worker.rollout.n}")
            
    #         if current_batch_size < rollout_batch_size:
    #             print(f"{current_batch_size=} < {rollout_batch_size=}")
    #             debug_print(f"Batch too small, continuing... {current_batch_size} < {rollout_batch_size}")
                
    #             max_try_make_batch = self.config.trainer.max_try_make_batch
    #             if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
    #                 print(f"{num_try_make_batch=}. Continue generating...")
    #             else:
    #                 debug_print(f"Max attempts reached: {num_try_make_batch} >= {max_try_make_batch}")
    #                 raise RuntimeError(
    #                     f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
    #                 )
    #         else:
    #             print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
    #             debug_print(f"Batch ready: {current_batch_size} >= {rollout_batch_size}")
                
    #             if self.config.algorithm.online_filtering:
    #                 reduced_metrics = reduce_metrics(all_metrics)
    #                 debug_print("Reduced metrics after filtering", reduced_metrics)
    #                 metrics.update({f"reward/{k}": v for k, v in reduced_metrics.items()})
                
    #             final_batch_size = self.config.data.rollout_batch_size * self.config.worker.rollout.n
    #             result_batch = batch[:final_batch_size]
    #             debug_print("Returning final batch", result_batch)
    #             debug_print("=" * 50)
                
    #             return result_batch

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)


            # pop those keys for generation
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )

            # generate a batch
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )
            #######################################################################
            #TODO: 当开启设置时，为每个唯一id生成一个global_uid
            if self.config.trainer.DIVA_GRPO:
                # 获取 id 数组和问题数组
                ids = new_batch.non_tensor_batch["id"]

                id_counts = {} 
                for id_val in ids:
                    id_counts[id_val] = id_counts.get(id_val, 0) + 1

                # 为每个唯一id生成一个global_uid
                unique_ids = list(id_counts.keys())
                id_to_uid = {id_val: str(uuid.uuid4()) for id_val in unique_ids}
                
                # 创建global_uids数组，确保相同id对应相同global_uid
                global_uids = np.array([id_to_uid[id_val] for id_val in ids], dtype=object)
                
                # 将global_uids添加到batch中
                new_batch.non_tensor_batch["global_uid"] = global_uids
            #######################################################################

            # repeat to align with repeated responses in rollout
            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            # filter group
            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) == 0:
                    raise RuntimeError("No sample is kept after filtering. Please check your data.")

                new_batch = new_batch[kept_sample_idxs]
                # Remove step1 and step2
                for key in ["step1", "step2", "variant", "varient_problem"]:
                    if key in new_batch.non_tensor_batch:
                        new_batch.non_tensor_batch.pop(key)

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!! batch lenght is === {len(batch)}")
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]

    # TODO:DIVAGRPO的自适应优势计算方法
    # TODO:weight_mode代表在何时使用Norm
    # TODO:Adjust_Low_Reward_Local、Adjust_Low_Reward_Global代表在何处使用RRB
    # TODO:alpfa和beta控制global和local的奖励比率
    def _compute_difficult_adaption_advantage_difficulty(
        self,
        data: DataProto,                  # 输入数据对象，包含批次数据和元数据 | Input data object containing batch data and metadata
        adv_estimator: AdvantageEstimator, # 优势估计器类型（GAE/GRPO/RLOO等） | Type of advantage estimator
        gamma: float = 1.0,               # 折扣因子（默认1.0） | Discount factor (default 1.0)
        lam: float = 1.0,                  # GAE的lambda参数（默认1.0） | Lambda parameter for GAE (default 1.0)
        weight_mode: str = "weight_before_norm",
        Adjust_Low_Reward_Local: bool = False,
        Adjust_Low_Reward_Global: bool = False, 
        alpfa: float = 1.0,
        beta: float = 1.0
    ):
        token_level_rewards = data.batch["token_level_rewards"]  # 每个token的奖励 | Per-token rewards
        response_mask = data.batch["response_mask"]              # 响应掩码（1=有效token）| Response mask (1=valid token)
        index = data.non_tensor_batch["uid"]                     # 样本唯一标识符 | Sample unique identifiers
        global_index = data.non_tensor_batch["global_uid"]
        difficult = data.non_tensor_batch["difficulty"]           # 提取difficult值
        category = data.non_tensor_batch["category"]
        id = data.non_tensor_batch["id"]
        old_log_probs = data.batch["old_log_probs"]
        log_data = []
        weight_mode = weight_mode.split('_')
        
        if "WBN" in weight_mode or "WAN" in weight_mode:
            # 计算每个global_id组的difficult均值
            global_difficult_means = defaultdict(list)
            for gid, diff in zip(global_index, difficult):
                global_difficult_means[gid].append(diff)
            # 计算每组的均值
            global_difficult_means = {gid: sum(diff_list) / len(diff_list) 
                                    for gid, diff_list in global_difficult_means.items()}
            # 计算每个uid的difficult与组均值的差值
            difficult_diffs = [diff - global_difficult_means[gid] 
                            for gid, diff in zip(global_index, difficult)]
        elif "ABSWBN" in weight_mode or "ABSWAN" in weight_mode:
            # 计算所有样本的difficult整体均值
            global_overall_mean = sum(difficult) / len(difficult)
            # 计算每个样本与整体均值的差值（保持zip结构，但实际不需要gid）
            difficult_diffs = [diff - global_overall_mean for gid, diff in zip(global_index, difficult)]

        if adv_estimator == AdvantageEstimator.GRPO:
            
            def to_numpy(x):
                return x.cpu().numpy() if hasattr(x, 'cpu') else np.array(x)
            # 计算每个句子的reward
            token_rewards = to_numpy(token_level_rewards)
            scores = token_rewards.sum(axis=-1)
            difficult_map = {}
            
            # 获取每个样本的难度
            for sample_id, diff, cat in zip(id, difficult, category):
                if cat != "origin_problem":
                    continue
                if sample_id not in difficult_map:
                    difficult_map[sample_id] = diff
                elif difficult_map[sample_id] != diff:
                    raise ValueError(f"冲突: id={sample_id} 对应多个 difficulty: "
                                    f"{difficult_map[sample_id]} vs {diff}")

            score_ranges = self.config.trainer.score_ranges
            difficulty_changes = self.config.trainer.difficulty_changes
            min_diff = self.config.trainer.min_diff
            max_diff = self.config.trainer.max_diff
            weighted_advantage_k = self.config.trainer.weighted_advantage_k
            # 按 sample_id 收集 scores
            id_to_allscores = {}
            for cat, sample_id, score in zip(category, id, scores):
                id_to_allscores.setdefault(sample_id, []).append(score)

            # 收集所有需要更新的样本及其新难度值
            updates = set()
            updates_log = set()
            for sample_id, scores in id_to_allscores.items():
                avg_score = sum(scores) / len(scores)  # 计算平均分
                old_diff = difficult_map[sample_id]  # 获取旧难度值
                # 调用抽象函数来计算新的难度值
                new_diff = calculate_new_difficulty(avg_score, old_diff, score_ranges, difficulty_changes, min_diff, max_diff)
                # 记录更新信息
                updates_log.add((sample_id, new_diff, old_diff, avg_score))
                updates.add((sample_id, new_diff))
            updates = list(set(updates))
            updates_log = list(set(updates_log))
            # 批量更新 dataset 中的 difficulty
            self.train_dataloader.dataset.update_difficulty(updates)

            stats = calculate_difficulty_changes(updates_log)
            # 保存统计结果
            all_log_path = self.config.trainer.All_Log_Path  ## path
            log_path = os.path.join(all_log_path,'statistics.log')
            append_to_json_log(log_path, stats)  ## path
            # # 标准化
            # normalized_advantages = normalize_advantages(advantages)
            # # 最小最大归一化
            # minmax_normalized_advantages = minmax_normalize_advantages(advantages)
            
            save_origin_global_advantages = []
            save_origin_local_advantages = []
            save_WBN_global_advantages = []
            save_NORM_global_advantages = []
            save_NORM_local_advantages = []
            save_WAN_global_advantages = []
            save_RRB_global_advantages = []
            save_RRB_local_advantages = []
            if "KLCOV" in weight_mode:
                ref_log_probs = data.batch["ref_log_probs"]
                local_advantages, local_returns = compute_grpo_outcome_advantage_kl_cov(
                    token_level_rewards, response_mask, index, 1e-6, old_log_probs, ref_log_probs, True, 5e-4, 0.3
                )
                global_advantages, global_returns = compute_grpo_outcome_advantage_kl_cov(
                    token_level_rewards, response_mask, global_index, 1e-6, old_log_probs, ref_log_probs, True, 5e-4, 0.3
                )
            else:
                local_advantages, local_returns = compute_grpo_outcome_advantage(
                    token_level_rewards, response_mask, index
                )
                global_advantages, global_returns = compute_grpo_outcome_advantage(
                    token_level_rewards, response_mask, global_index
                )
            save_global_advantages = global_advantages
            save_local_advantages = local_advantages
            save_origin_global_advantages = global_advantages
            save_origin_local_advantages = local_advantages

            if "WBN" in weight_mode or "ABSWBN" in weight_mode:
                weighted_global_advantages = []
                for diff_diff, g_adv in zip(difficult_diffs, global_advantages):
                    weighted_global_advantages.append(weighted_advantage(diff_diff, g_adv, weighted_advantage_k))
                weighted_global_advantages = torch.stack(weighted_global_advantages)
                save_WBN_global_advantages = weighted_global_advantages
                global_advantages = weighted_global_advantages

            if 'RMSNORM' in weight_mode:
                local_advantages = rms_normalize_advantages(local_advantages)
                global_advantages = rms_normalize_advantages(global_advantages)
            elif 'MINMAXNORM' in weight_mode:
                local_advantages = minmax_normalize_advantages(local_advantages)
                global_advantages = minmax_normalize_advantages(global_advantages)
            elif 'ZSCORENORM' in weight_mode:
                local_advantages = normalize_advantages(local_advantages)
                global_advantages = normalize_advantages(global_advantages)

            if 'RMSNORM' in weight_mode or 'MINMAXNORM' in weight_mode or 'ZSCORENORM' in weight_mode:
                save_NORM_local_advantages = local_advantages
                save_NORM_global_advantages = global_advantages
    
            if "WAN" in weight_mode or "ABSWAN" in weight_mode:
                weighted_global_advantages = []
                for diff_diff, g_adv in zip(difficult_diffs, global_advantages):
                    weighted_global_advantages.append(weighted_advantage(diff_diff, g_adv, weighted_advantage_k))
                weighted_global_advantages = torch.stack(weighted_global_advantages)
                save_WAN_global_advantages = weighted_global_advantages
                global_advantages = weighted_global_advantages

            if "RRBLOCAL" in weight_mode:
                local_advantages = adjust_low_reward_advantages(
                    local_advantages, 
                    index, 
                    token_level_rewards
                )
                save_RRB_local_advantages = local_advantages

            if "RRBGLOBAL" in weight_mode:
                global_advantages = adjust_low_reward_advantages(
                    global_advantages, 
                    global_index, 
                    token_level_rewards
                )
                save_RRB_global_advantages = global_advantages

            advantages = alpfa * local_advantages + beta * global_advantages
            returns = alpfa * local_advantages + beta * global_advantages
            # advantages = global_advantages
            # returns = global_advantages
            vector_data = {
                "local_advantages": local_advantages.cpu().numpy()[:, 0],  # 转为numpy并只取第一列
                "global_advantages": global_advantages.cpu().numpy()[:, 0],
                # "adj_global_advantages": adj_global_advantages.cpu().numpy()[:, 0],
                "advantages": advantages.cpu().numpy()[:, 0],  # 取第一列
                "token_rewards": token_level_rewards.cpu().numpy().sum(axis=-1),  # 计算token rewards总和
                "save_origin_global_advantages": save_origin_global_advantages,
                "save_origin_local_advantages": save_origin_local_advantages,
                "save_WBN_global_advantages": save_WBN_global_advantages,
                "save_NORM_global_advantages": save_NORM_global_advantages,
                "save_NORM_local_advantages": save_NORM_local_advantages,
                "save_WAN_global_advantages": save_WAN_global_advantages,
                "save_RRB_global_advantages": save_RRB_global_advantages,
                "save_RRB_local_advantages": save_RRB_local_advantages
            }

            # 调用保存函数
            all_log_path = self.config.trainer.All_Log_Path  ## path
            log_path = os.path.join(all_log_path,'full_vector.log')
            save_full_vectors_to_json(data, vector_data, log_path) ## path
        
        # 将计算结果存入原始数据对象 | Store results in original data object
        data.batch["advantages"] = advantages  # 优势函数值 | Advantage values
        data.batch["returns"] = returns        # 回报值 | Return values
        return data

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0

        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # make a batch of data
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                # balance the number of valid tokens on each dp rank.
                # NOTE: this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                self._balance_batch(batch, metrics=metrics)

                # compute global valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # compute reward
                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        reward_ref = self.reward_fn.compute_reward.remote(batch)

                # recompute old_log_probs
                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)

                # compute ref_log_probs
                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)

                # compute values
                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        # get token level scores asynchronously
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)

                    # apply kl penalty if available
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        # apply kl penalty to reward
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    #TODO: 1.如果开启DIVA_GRPO, 则使用动态变化的难度
                    #TODO: 2.如果开启Share_VL，则使用Global+local
                    #TODO: 3.如果都关闭则使用原始的优势计算
                    # compute advantages, executed on the driver process
                    if self.config.trainer.DIVA_GRPO and (self.config.trainer.DIVA_warmup != True or self.global_step > 30):
                        print("使用DIVA_GRPO优势估计方法")
                        batch = self._compute_difficult_adaption_advantage_difficulty(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,  # 优势估计方法（如GAE）
                            gamma=self.config.algorithm.gamma,  # 折扣因子
                            lam=self.config.algorithm.lam,  # GAE参数
                            weight_mode=self.config.algorithm.weight_mode,
                        )
                    elif self.config.trainer.Share_VL:
                        print("使用Share_VL优势估计方法")
                        batch = self._compute_difficult_adaption_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,  # 优势估计方法（如GAE）
                            gamma=self.config.algorithm.gamma,  # 折扣因子
                            lam=self.config.algorithm.lam,  # GAE参数
                        )
                    else:
                        print("使用普通优势估计方法")
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            all_log_path = self.config.trainer.All_Log_Path  ## path
                        )

                # update critic
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                # update actor
                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
