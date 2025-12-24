import torch
import os
import random
import re
from PIL import Image, ImageDraw, ImageFont
import io
import fcntl
import json

def calculate_difficulty_changes(updates_log):
    """
    统计difficulty变化，并计算平均分。
    :param updates_log: 包含更新日志的列表，格式为 (uid, new_diff, old_diff, avg_score)
    :return: 包含统计信息的字典
    """
    stats = {
        'new_diff': {},
        'old_diff': {},
        'avg_score': 0.0
    }

    total_entries = 0
    total_score = 0.0

    for uid, new_diff, old_diff, avg_score in updates_log:
        # 记录 new_diff 和 old_diff 的变化
        if new_diff != 0:
            stats['new_diff'][new_diff] = stats['new_diff'].get(new_diff, 0) + 1
        if old_diff != 0:
            stats['old_diff'][old_diff] = stats['old_diff'].get(old_diff, 0) + 1

        # 累加avg_score用于计算平均值
        total_score += avg_score
        total_entries += 1

    # 计算平均分
    if total_entries > 0:
        stats['avg_avg_score'] = float(total_score / total_entries)

    return stats

def normalize_advantages(advantages):
    """
    归一化优势值（标准化）。
    :param advantages: PyTorch 张量
    :return: 归一化后的张量
    """
    mean = advantages.mean()
    std = advantages.std()

    # 避免除以0
    std = std if std > 0 else 1.0

    # 归一化
    normalized = (advantages - mean) / std
    return normalized

def minmax_normalize_advantages(advantages):
    """
    使用最大最小缩放对优势值进行归一化。
    :param advantages: PyTorch 张量
    :return: 归一化后的张量
    """
    max_abs = torch.max(torch.abs(advantages))

    # 如果所有优势值都是0，直接返回原数组（避免除以0）
    if max_abs == 0:
        return advantages

    # 缩放到 [-1, 1] 区间
    normalized = advantages / max_abs
    return normalized

def calculate_new_difficulty(avg_score, old_diff, score_ranges, difficulty_changes, min_diff, max_diff):
    for (min_score, max_score), difficulty_change in zip(score_ranges, difficulty_changes):
        if min_score <= avg_score <= max_score:
            new_diff = old_diff + difficulty_change
            break
    else:
        new_diff = old_diff  # 如果没有匹配的区间，保持原难度值

    # 限制难度值在 min_diff 和 max_diff 之间
    return max(min_diff, min(new_diff, max_diff))

def multiplier(difficult, advantage, weighted_advantage_k):
    # 使用逐元素操作替代标量判断
    k = weighted_advantage_k  # 可调整的敏感度参数 （）
    
    # 创建与advantage相同形状的初始乘数
    multiplier = torch.ones_like(advantage)
    
    # 找到非零元素的索引
    nonzero_mask = (advantage != 0)
    
    # 对非零元素计算乘数
    sign = torch.where(advantage > 0, 1.0, -1.0)
    multiplier[nonzero_mask] = torch.exp(k * difficult * sign[nonzero_mask])
    
    return multiplier

def weighted_advantage(difficult, advantage, weighted_advantage_k):
    return advantage * multiplier(difficult, advantage, weighted_advantage_k)

def log_difficulty_update(id_, old_diff, new_diff, math_reward, path):
    """记录difficult更新日志"""
    log_entry = {
        "id": id_,
        "old_difficulty": old_diff,
        "new_difficulty": new_diff,
        "math_reward": math_reward,
        "timestamp": datetime.datetime.now().isoformat()
    }
    try:
        with open(path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Error logging difficulty update: {e}")

def append_to_json_log(filename, data):
    """改进的JSON日志追加方法，使用JSONL格式（每行一个JSON对象）"""
    try:
        with open(filename, "a+") as f:
            # 获取文件锁
            fcntl.flock(f, fcntl.LOCK_EX)
            # 移动到文件末尾
            f.seek(0, 2)
            # 写入数据（JSONL格式不需要逗号或方括号）
            json.dump(data, f)
            f.write("\n")
            # 在文件关闭前释放锁
            fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        print(f"Error writing to {filename}: {str(e)}")

def normalize_advantages(advantages):
    # 计算均值和标准差
    mean = advantages.mean()
    std = advantages.std()
    # 避免除以0
    std = std if std > 0 else 1.0
    # 归一化
    normalized = (advantages - mean) / std
    return normalized

def minmax_normalize_advantages(advantages):
    # 使用 PyTorch 的方法
    max_abs = torch.max(torch.abs(advantages))
    # 如果所有优势值都是0，直接返回原数组（避免除以0）
    if max_abs == 0:
        return advantages
    # 缩放到 [-1, 1] 区间
    normalized = advantages / max_abs
    return normalized

def rms_normalize_advantages(advantages, epsilon=1e-6):
    # 计算每个样本的均方根
    rms = torch.sqrt(torch.mean(torch.square(advantages), dim=-1, keepdim=True) + epsilon)
    # 归一化
    normalized = advantages / rms
    return normalized

def adjust_low_reward_advantages(
    global_advantages: torch.Tensor,
    global_index: list,
    token_level_rewards: torch.Tensor,
    threshold: float = 0.15,
    scale_factor: float = 0.1
) -> torch.Tensor:
    """
    调整低奖励global_id对应的advantages
    
    参数:
        global_advantages: 原始计算的global advantages张量
        global_index: global_id列表，与global_advantages一一对应
        token_level_rewards: 每个token的奖励张量
        threshold: 判断为低奖励的阈值(默认0.15)
        scale_factor: 低奖励advantages的缩放因子(默认0.1)
    
    返回:
        调整后的global_advantages张量
    """
    # 创建字典记录每个global_id的总奖励
    global_id_to_rewards = {}
    for gid, rewards in zip(global_index, token_level_rewards):
        if gid not in global_id_to_rewards:
            global_id_to_rewards[gid] = []
        global_id_to_rewards[gid].append(rewards.sum().item())
    
    # 找出所有总奖励都低于threshold的global_id
    low_reward_global_ids = {
        gid for gid, rewards in global_id_to_rewards.items()
        if all(r < threshold for r in rewards)
    }
    
    # 复制原始advantages以避免原地修改
    adjusted_advantages = global_advantages.clone()
    
    # 调整低奖励global_id对应的advantages
    for i, gid in enumerate(global_index):
        if gid in low_reward_global_ids:
            adjusted_advantages[i] = adjusted_advantages[i] * scale_factor
    
    return adjusted_advantages

def save_full_vectors_to_json(data, vector_data, output_path):
    """
    保存完整向量到JSON文件（包括所有相关的向量和原始数据）
    
    参数:
        data: DataProto 包含样本数据
        vector_data: 包含所有需要保存的向量数据及原始数据的字典
    """
    existing_lines = []
    
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing_lines = [line.strip() for line in f if line.strip()]

    # 准备数据（保存处理后的向量）
    jsonl_data = [
        json.dumps({
            "id": str(data.non_tensor_batch["id"][i]),
            "local_idx": str(data.non_tensor_batch["uid"][i]),
            "global_idx": str(data.non_tensor_batch["global_uid"][i]),
            "problem": str(data.non_tensor_batch["problem"][i]),
            "index": i,
            "category": data.non_tensor_batch["category"][i],
            "difficulty": data.non_tensor_batch["difficulty"][i],
            # 判断 `save_origin_global_advantages` 是否为空，并做相应处理
            "local_advantages": 
                None if len(vector_data["local_advantages"]) == 0 else (
                    vector_data["local_advantages"][i].tolist()
                    if isinstance(vector_data["local_advantages"][i], torch.Tensor) 
                    else float(vector_data["local_advantages"][i])
                ) if i < len(vector_data["local_advantages"]) else None,
            
            "global_advantages": 
                None if len(vector_data["global_advantages"]) == 0 else (
                    vector_data["global_advantages"][i].tolist()
                    if isinstance(vector_data["global_advantages"][i], torch.Tensor) 
                    else float(vector_data["global_advantages"][i])
                ) if i < len(vector_data["global_advantages"]) else None,

            "advantages": 
                None if len(vector_data["advantages"]) == 0 else (
                    vector_data["advantages"][i].tolist()
                    if isinstance(vector_data["advantages"][i], torch.Tensor) 
                    else float(vector_data["advantages"][i])
                ) if i < len(vector_data["advantages"]) else None,

            "token_rewards_score": 
                None if len(vector_data["token_rewards"]) == 0 else (
                    vector_data["token_rewards"][i].tolist()
                    if isinstance(vector_data["token_rewards"][i], torch.Tensor) 
                    else float(vector_data["token_rewards"][i])
                ) if i < len(vector_data["token_rewards"]) else None,

            "save_origin_global_advantages": 
                None if len(vector_data["save_origin_global_advantages"]) == 0 else (
                    vector_data["save_origin_global_advantages"][i].tolist()[:5]
                    if isinstance(vector_data["save_origin_global_advantages"][i], torch.Tensor) 
                    else vector_data["save_origin_global_advantages"][i][:5]
                ) if i < len(vector_data["save_origin_global_advantages"]) else None,
            
            "save_origin_local_advantages": 
                None if len(vector_data["save_origin_local_advantages"]) == 0 else (
                    vector_data["save_origin_local_advantages"][i].tolist()[:5]
                    if isinstance(vector_data["save_origin_local_advantages"][i], torch.Tensor) 
                    else vector_data["save_origin_local_advantages"][i][:5]
                ) if i < len(vector_data["save_origin_local_advantages"]) else None,

            "save_WBN_global_advantages": 
                None if len(vector_data["save_WBN_global_advantages"]) == 0 else (
                    vector_data["save_WBN_global_advantages"][i].tolist()[:5]
                    if isinstance(vector_data["save_WBN_global_advantages"][i], torch.Tensor) 
                    else vector_data["save_WBN_global_advantages"][i][:5]
                ) if i < len(vector_data["save_WBN_global_advantages"]) else None,

            "save_NORM_global_advantages": 
                None if len(vector_data["save_NORM_global_advantages"]) == 0 else (
                    vector_data["save_NORM_global_advantages"][i].tolist()[:5]
                    if isinstance(vector_data["save_NORM_global_advantages"][i], torch.Tensor) 
                    else vector_data["save_NORM_global_advantages"][i][:5]
                ) if i < len(vector_data["save_NORM_global_advantages"]) else None,

            "save_NORM_local_advantages": 
                None if len(vector_data["save_NORM_local_advantages"]) == 0 else (
                    vector_data["save_NORM_local_advantages"][i].tolist()[:5]
                    if isinstance(vector_data["save_NORM_local_advantages"][i], torch.Tensor) 
                    else vector_data["save_NORM_local_advantages"][i][:5]
                ) if i < len(vector_data["save_NORM_local_advantages"]) else None,

            "save_WAN_global_advantages": 
                None if len(vector_data["save_WAN_global_advantages"]) == 0 else (
                    vector_data["save_WAN_global_advantages"][i].tolist()[:5]
                    if isinstance(vector_data["save_WAN_global_advantages"][i], torch.Tensor) 
                    else vector_data["save_WAN_global_advantages"][i][:5]
                ) if i < len(vector_data["save_WAN_global_advantages"]) else None,

            "save_RRB_global_advantages": 
                None if len(vector_data["save_RRB_global_advantages"]) == 0 else (
                    vector_data["save_RRB_global_advantages"][i].tolist()[:5]
                    if isinstance(vector_data["save_RRB_global_advantages"][i], torch.Tensor) 
                    else vector_data["save_RRB_global_advantages"][i][:5]
                ) if i < len(vector_data["save_RRB_global_advantages"]) else None,

            "save_RRB_local_advantages": 
                None if len(vector_data["save_RRB_local_advantages"]) == 0 else (
                    vector_data["save_RRB_local_advantages"][i].tolist()[:5]
                    if isinstance(vector_data["save_RRB_local_advantages"][i], torch.Tensor) 
                    else vector_data["save_RRB_local_advantages"][i][:5]
                ) if i < len(vector_data["save_RRB_local_advantages"]) else None,
        })
        for i in range(len(data.non_tensor_batch["id"]))
    ]



    # 合并数据并写入文件（追加模式）
    with open(output_path, "a", encoding="utf-8") as f:
        for line in jsonl_data:
            f.write(line + "\n")

    print(f"已保存完整向量数据，新增样本数：{len(jsonl_data)}，总样本数：{len(existing_lines) + len(jsonl_data)}")


def save_full_vectors_to_json_origin_grpo(data, vector_data, output_path):
    """
    保存完整向量到JSON文件（包括所有相关的向量和原始数据）
    
    参数:
        data: DataProto 包含样本数据
        vector_data: 包含所有需要保存的向量数据及原始数据的字典
    """
    existing_lines = []
    
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing_lines = [line.strip() for line in f if line.strip()]

    # 准备数据（保存处理后的向量）
    jsonl_data = [
        json.dumps({
            "id": str(data.non_tensor_batch["id"][i]),
            "local_idx": str(data.non_tensor_batch["uid"][i]),
            "problem": str(data.non_tensor_batch["problem"][i]),
            "index": i,
            "category": data.non_tensor_batch["category"][i],
            # 判断 `save_origin_global_advantages` 是否为空，并做相应处理
            "advantages": 
                None if len(vector_data["advantages"]) == 0 else (
                    vector_data["advantages"][i].tolist()
                    if isinstance(vector_data["advantages"][i], torch.Tensor) 
                    else float(vector_data["advantages"][i])
                ) if i < len(vector_data["advantages"]) else None,

             "token_rewards_score": 
                None if len(vector_data["token_rewards"]) == 0 else (
                    vector_data["token_rewards"][i].tolist()
                    if isinstance(vector_data["token_rewards"][i], torch.Tensor) 
                    else float(vector_data["token_rewards"][i])
                ) if i < len(vector_data["token_rewards"]) else None
        })
        for i in range(len(data.non_tensor_batch["id"]))
    ]



    # 合并数据并写入文件（追加模式）
    with open(output_path, "a", encoding="utf-8") as f:
        for line in jsonl_data:
            f.write(line + "\n")

    print(f"已保存完整向量数据，新增样本数：{len(jsonl_data)}，总样本数：{len(existing_lines) + len(jsonl_data)}")