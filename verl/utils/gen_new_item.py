import random
import copy
from typing import Dict, List, Callable, Any, Optional
from functools import partial
import numpy as np

from ..difficulty_variation.difficulty_utils import (
    rotate_image,
    add_gaussian_noise,
    add_salt_pepper_noise,
    add_speckle_noise,
    add_blur,
    add_low_resolution,
    add_text_to_image_with_space
    # ,get_variant_text
)

# ==================== 常量和配置 ====================

# 难度调整参数
DIFFICULTY_ADJUSTMENT = {
    "low_threshold": -7,
    "medium_threshold": -5,
    "high_threshold": -3
}

# 噪声强度阈值
NOISE_INTENSITY_THRESHOLDS = {
    "low": 0.25,
    "medium": 0.45
}

# 图像变换函数映射
IMAGE_TRANSFORMS = {
    "rotate": rotate_image,
    "gaussian": add_gaussian_noise,
}

# 难度采样映射 (用于 generate_varent_difficulty_samples)
DIFFICULTY_MAPPING = {
    1: [9], 2: [8], 3: [7], 4: [6], 5: [6],
    6: [5], 7: [5], 8: [5], 9: [4]
}

# 难度均值 (用于正态分布采样)
DIFFICULTY_MEAN = {
    1: 8, 2: 7.5, 3: 7, 4: 6.5, 5: 6,
    6: 5.5, 7: 4.5, 8: 3.5, 9: 3
}

# ==================== 基础工具函数 ====================

def update_item_metadata(
    item: Dict[str, Any],
    difficulty_delta: int,
    category: str
) -> Dict[str, Any]:
    """统一更新题目元数据"""
    item['difficulty'] += difficulty_delta
    item['category'] = category
    return item

def select_variant_text(item: Dict[str, Any], probability: float = 0.5) -> str:
    """随机选择变体文本或原问题"""
    if random.random() < probability and item.get('variant') is not None:
        return random.choice(item['variant'])
    return item['problem']

# ==================== 图像处理函数 ====================

def gen_only_vision(
    noise_intensity: float,
    difficulty_delta: int,
    origin_item: Dict[str, Any]
) -> Dict[str, Any]:
    """生成纯视觉问题：将文本添加到图像中"""
    if random.random() < 0.5:
        origin_item['problem'] = select_variant_text(origin_item)
    
    origin_item['images']['bytes'] = add_text_to_image_with_space(
        origin_item['images']['bytes'],
        origin_item['problem'],
        text_height_ratio=0.22,
        position=None
    )
    origin_item['problem'] = "As shown in the <image>."
    
    return update_item_metadata(
        origin_item,
        difficulty_delta,
        f'only_vision_{noise_intensity}'
    )

def gen_speckle_noise(
    noise_intensity: float,
    difficulty_delta: int,
    origin_item: Dict[str, Any]
) -> Dict[str, Any]:
    """添加斑点噪声"""
    origin_item['images']['bytes'] = add_speckle_noise(
        origin_item['images']['bytes'],
        noise_intensity
    )
    return update_item_metadata(
        origin_item,
        difficulty_delta,
        f'speckle_{noise_intensity}'
    )

def gen_gaussian_noise(
    noise_intensity: float,
    difficulty_delta: int,
    origin_item: Dict[str, Any]
) -> Dict[str, Any]:
    """添加高斯噪声"""
    origin_item['images']['bytes'] = add_gaussian_noise(
        origin_item['images']['bytes'],
        noise_intensity
    )
    origin_item['problem'] = select_variant_text(origin_item)
    return update_item_metadata(
        origin_item,
        difficulty_delta,
        f'gauss_{noise_intensity}'
    )

def gen_salt_pepper_noise(
    noise_intensity: float,
    difficulty_delta: int,
    origin_item: Dict[str, Any]
) -> Dict[str, Any]:
    """添加椒盐噪声"""
    origin_item['images']['bytes'] = add_salt_pepper_noise(
        origin_item['images']['bytes'],
        noise_intensity
    )
    origin_item['problem'] = select_variant_text(origin_item)
    return update_item_metadata(
        origin_item,
        difficulty_delta,
        f'salt_{noise_intensity}'
    )

def gen_blur(
    noise_intensity: float,
    difficulty_delta: int,
    origin_item: Dict[str, Any]
) -> Dict[str, Any]:
    """添加模糊效果"""
    origin_item['images']['bytes'] = add_blur(
        origin_item['images']['bytes'],
        noise_intensity
    )
    return update_item_metadata(
        origin_item,
        difficulty_delta,
        f'blur_{noise_intensity}'
    )

def gen_low_resolution(
    noise_intensity: float,
    difficulty_delta: int,
    origin_item: Dict[str, Any]
) -> Dict[str, Any]:
    """降低图像分辨率"""
    origin_item['images']['bytes'] = add_low_resolution(
        origin_item['images']['bytes'],
        noise_intensity
    )
    return update_item_metadata(
        origin_item,
        difficulty_delta,
        f'low_resolution_{noise_intensity}'
    )

def gen_rotate(
    noise_intensity: float,
    difficulty_delta: int,
    origin_item: Dict[str, Any]
) -> Dict[str, Any]:
    """旋转图像（角度根据噪声强度动态计算）"""
    # 根据噪声强度确定旋转步长
    if noise_intensity < NOISE_INTENSITY_THRESHOLDS["low"]:
        angle_step = 45
    elif noise_intensity < NOISE_INTENSITY_THRESHOLDS["medium"]:
        angle_step = 30
    else:
        angle_step = 1

    # 计算可能的旋转角度（1-359度之间能被angle_step整除的角度）
    max_multiple = (359) // angle_step
    chosen_multiple = random.randint(1, max(1, max_multiple))
    rotation_angle = chosen_multiple * angle_step

    # 应用旋转
    origin_item['images']['bytes'] = rotate_image(
        origin_item['images']['bytes'],
        rotation_angle
    )
    
    # 更新问题描述
    variant_text = select_variant_text(origin_item)
    origin_item['problem'] = (
        f"{variant_text} This image has been rotated by {rotation_angle} degrees. "
        "Please mentally rotate it back and solve the problem."
    )
    
    return update_item_metadata(
        origin_item,
        difficulty_delta,
        f'rotate_{angle_step}'
    )

# ==================== 文本和问题处理函数 ====================

def gen_variant_text(
    difficulty_delta: int,
    origin_item: Dict[str, Any]
) -> Dict[str, Any]:
    """生成文本变体问题（不改变难度）"""
    origin_item['problem'] = select_variant_text(origin_item)
    origin_item['category'] = 'varient_text'
    return origin_item

def gen_origin_sample(
    difficulty_delta: int,
    origin_item: Dict[str, Any]
) -> Dict[str, Any]:
    """生成文本变体问题（不改变难度）"""
    origin_item['problem'] = origin_item['problem']
    origin_item['category'] = 'origin_item'
    return origin_item

def gen_ground_truth(
    difficulty_delta: int,
    origin_item: Dict[str, Any]
) -> Dict[str, Any]:
    """在问题中直接提供正确答案"""
    origin_item['problem'] = (
        f"{origin_item['problem']}\n"
        f"The correct answer to this question is: {origin_item['answer']}. "
        "Please provide detailed reasoning and arrive at the final result."
    )
    return update_item_metadata(
        origin_item,
        difficulty_delta,
        'ground_truth'
    )

def gen_think_step(
    difficulty_delta: int,
    origin_item: Dict[str, Any]
) -> Dict[str, Any]:
    """
    根据难度级别提供思考步骤引导
    
    Args:
        difficulty_delta: 负值，值越小提供的步骤越多
        origin_item: 包含原始题目和 step2 字段的字典
    """
    if not origin_item.get('step2'):
        origin_item['category'] = 'origin_problem_think_step'
        return origin_item

    # 根据难度调整值确定步骤数量
    total_steps = len(origin_item['step2'])
    
    if difficulty_delta <= DIFFICULTY_ADJUSTMENT["low_threshold"]:
        num_steps = total_steps
    elif difficulty_delta <= DIFFICULTY_ADJUSTMENT["medium_threshold"]:
        num_steps = min(3, total_steps)
    elif difficulty_delta <= DIFFICULTY_ADJUSTMENT["high_threshold"]:
        num_steps = min(2, total_steps)
    else:
        num_steps = 1

    # 提取并拼接思考步骤
    thinking_steps = " ".join(origin_item['step2'][:num_steps])
    
    # 增强问题描述
    origin_item['problem'] = (
        f"{origin_item['problem']} I will now provide some thinking prompts. "
        "Please output the complete thought process and answer from the beginning, "
        f"without skipping any steps. : {thinking_steps}."
    )
    
    return update_item_metadata(
        origin_item,
        difficulty_delta,
        f'guided_thinking_{num_steps}'
    )

# ==================== 复合变换函数 ====================

def apply_random_transformations(
    num_transforms: int,
    noise_intensity: float,
    difficulty_delta: int,
    origin_item: Dict[str, Any]
) -> Dict[str, Any]:
    """随机应用多种图像变换"""
    # 对于图文混合内容，只旋转
    if origin_item.get('text_in_image', False):
        selected_keys = ["rotate"]
        num_transforms = 1
    else:
        selected_keys = random.sample(
            list(IMAGE_TRANSFORMS.keys()),
            min(num_transforms, len(IMAGE_TRANSFORMS))
        )

    # 随机替换问题为变体文本
    variant_threshold = 0.2 + 0.07 * noise_intensity
    if random.random() < variant_threshold:
        origin_item['problem'] = select_variant_text(origin_item)

    # 应用选中的变换
    for key in selected_keys:
        if key == "rotate":
            origin_item['problem'] = (
                f"{origin_item['problem']} "
                "This image has been rotated. Please mentally rotate it back and solve the problem."
            )
        
        origin_item['images']['bytes'] = IMAGE_TRANSFORMS[key](
            origin_item['images']['bytes'],
            noise_intensity
        )

    # 生成类别名称
    category_name = f"random_{num_transforms}_{noise_intensity}_" + "_".join(selected_keys)
    
    return update_item_metadata(origin_item, difficulty_delta, category_name)

# ==================== 难度采样函数 ====================

def get_difficulty_sample(base_difficulty: int) -> Optional[List[int]]:
    """根据基础难度获取确定性难度样本"""
    return DIFFICULTY_MAPPING.get(base_difficulty)

def generate_variant_difficulty_samples(
    base_difficulty: int,
    num_samples: int = 2,
    sigma: float = 1.0
) -> List[int]:
    """
    生成难度样本
    
    Args:
        base_difficulty: 基础难度值 (1-9)
        num_samples: 样本数量
        sigma: 正态分布标准差
    """
    if num_samples == 1:
        sample = get_difficulty_sample(base_difficulty)
        return sample if sample else []
    else:
        # 使用正态分布生成样本
        mean = DIFFICULTY_MEAN.get(base_difficulty, 5)
        samples = np.random.normal(mean, sigma, num_samples)
        samples = np.round(samples).astype(int)
        samples = np.clip(samples, 1, 9)
        return samples.tolist()

# ==================== 主生成函数 ====================

# 难度函数映射表
DIFFICULTY_FUNCTION_MAP = {
    -8: [partial(gen_think_step, -8)],
    -7: [partial(gen_think_step, -7)],
    -6: [partial(gen_think_step, -6)],
    -5: [partial(gen_think_step, -5)],
    -4: [partial(gen_think_step, -4)],
    -3: [partial(gen_think_step, -3)],
    -2: [partial(gen_think_step, -2)],
    -1: [partial(gen_think_step, -1)],
    0: [partial(gen_variant_text, 0)],
    1: [
        partial(gen_origin_sample, 1),
        partial(gen_origin_sample, 1),
        partial(gen_variant_text, 1),
        partial(gen_gaussian_noise, 0.30, 1),
        partial(gen_rotate, 0.30, 1),
        partial(gen_variant_text, 1)
    ],
    2: [
        partial(gen_gaussian_noise, 0.35, 2),
        partial(gen_rotate, 0.35, 2),
        partial(gen_variant_text, 2),
        partial(gen_only_vision,0.50, 2),
        partial(gen_variant_text, 2)
    ],
    3: [
        partial(gen_gaussian_noise, 0.40, 3),
        partial(gen_rotate, 0.40, 3),
        partial(gen_variant_text, 3),
        partial(gen_only_vision,0.40, 3)
    ],
    4: [partial(gen_gaussian_noise, 0.45, 3), partial(gen_rotate, 0.45, 3), partial(apply_random_transformations, 2, 0.30, 4),partial(apply_random_transformations, 2, 0.30, 4),partial(gen_only_vision,0.50, 2)],
    5: [partial(apply_random_transformations, 2, 0.30, 4),partial(apply_random_transformations, 2, 0.35, 5),partial(gen_only_vision,0.50, 2)],
    6: [partial(apply_random_transformations, 2, 0.30, 4),partial(apply_random_transformations, 2, 0.40, 6),partial(gen_only_vision,0.50, 2)],
    7: [partial(apply_random_transformations, 2, 0.30, 4),partial(apply_random_transformations, 2, 0.45, 7),partial(gen_only_vision,0.50, 2)],
    8: [partial(apply_random_transformations, 2, 0.30, 4),partial(apply_random_transformations, 2, 0.50, 8),partial(gen_only_vision,0.50, 2)],
}


# 难度函数映射表
# DIFFICULTY_FUNCTION_MAP = {
#     -8: [partial(gen_think_step, -8)],
#     -7: [partial(gen_think_step, -7)],
#     -6: [partial(gen_think_step, -6)],
#     -5: [partial(gen_think_step, -5)],
#     -4: [partial(gen_think_step, -4)],
#     -3: [partial(gen_think_step, -3)],
#     -2: [partial(gen_think_step, -2)],
#     -1: [partial(gen_think_step, -1)],
#     0: [partial(gen_variant_text, 0)],
#     1: [
#         partial(gen_origin_sample, 1),
#     ],
#     2: [
#         partial(gen_gaussian_noise, 0.50, 2),
#         partial(gen_rotate, 0.50, 2),
#         partial(gen_variant_text, 2),
#         partial(gen_only_vision,0.50, 2)
#     ],
#     3: [
#         partial(gen_gaussian_noise, 0.60, 3),
#         partial(gen_rotate, 0.60, 3),
#         partial(gen_variant_text, 3),
#         partial(gen_only_vision,0.50, 3)
#     ],
#     4: [partial(apply_random_transformations, 2, 0.40, 4)],
#     5: [partial(apply_random_transformations, 2, 0.50, 5)],
#     6: [partial(apply_random_transformations, 2, 0.40, 6)],
#     7: [partial(apply_random_transformations, 2, 0.50, 7)],
#     8: [partial(apply_random_transformations, 2, 0.50, 8)],
# }

import os
import copy
import random
from datetime import datetime
from typing import Dict, Any, List
from PIL import Image as ImageModule

import math
import logging
import os
from pathlib import Path
from typing import Union, Any, Optional
from datetime import datetime
from io import BytesIO
from PIL import Image as ImageModule
from PIL.Image import Image as ImageObject

def generate_variants(
    origin_item: Dict[str, Any],
    variant_num: int,
    allow_multiple_thinking: bool = False
) -> List[Dict[str, Any]]:
    """
    生成指定数量的题目变体
    
    Args:
        origin_item: 原始题目
        variant_num: 变体数量
        allow_multiple_thinking: 是否允许多个思考步骤变体
    """
    # 创建debug日志文件路径
    debug_dir = "/mmu_cd_ssd/zhangzhenyu06/workspace/Rebuttal/Help"
    os.makedirs(debug_dir, exist_ok=True)
    debug_file = os.path.join(debug_dir, f"generate_variants_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    def log_debug(message: str, data: Any = None):
        """辅助函数：写入debug日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        with open(debug_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
            if data is not None:
                f.write(f"  Data: {str(data)[:500]}\n")  # 限制长度避免日志过大
            f.write("-" * 50 + "\n")
    
    # 记录函数入口参数
    log_debug("函数 generate_variants 被调用", {
        "origin_item_keys": list(origin_item.keys()) if origin_item else None,
        "variant_num": variant_num,
        "allow_multiple_thinking": allow_multiple_thinking,
        "origin_difficulty": origin_item.get('difficulty') if origin_item else None
    })
    
    # 生成难度样本
    difficulty_samples = generate_variant_difficulty_samples(
        origin_item['difficulty'],
        variant_num
    )

    log_debug("生成难度样本完成", {
        "difficulty_samples": difficulty_samples,
        "origin_difficulty": origin_item['difficulty']
    })
    
    # 计算难度差值
    difficulty_deltas = [d - origin_item['difficulty'] for d in difficulty_samples]
    log_debug("计算难度差值完成", {
        "difficulty_deltas": difficulty_deltas
    })
    
    # 处理思考步骤变体（只允许一个）
    if not allow_multiple_thinking:
        negative_found = False
        processed_deltas = []
        for delta in difficulty_deltas:
            if delta < 0:
                if not negative_found:
                    processed_deltas.append(delta)
                    negative_found = True
                else:
                    # 替换为随机正值
                    processed_deltas.append(random.choice([0, 1, 2, 3]))
            else:
                processed_deltas.append(delta)
        difficulty_deltas = processed_deltas
    
    # 生成变体
    variant_list = [origin_item]
    log_debug("开始生成变体", {
        "initial_variant_list_length": len(variant_list)
    })
    
    for idx, delta in enumerate(difficulty_deltas):
        log_debug(f"处理第 {idx+1}/{len(difficulty_deltas)} 个变体", {
            "delta": delta
        })
        
        new_item = copy.deepcopy(origin_item)
        
        # 获取并执行对应难度的处理函数
        func_list = DIFFICULTY_FUNCTION_MAP.get(delta, [])
        log_debug(f"Delta {delta} 对应的处理函数列表", {
            "func_list_length": len(func_list),
            "delta": delta,
            "available_deltas": list(DIFFICULTY_FUNCTION_MAP.keys())
        })
        
        if func_list:
            selected_func = random.choice(func_list)
            log_debug(f"选择处理函数", {
                "delta": delta,
                "selected_func": selected_func.__name__ if hasattr(selected_func, '__name__') else str(selected_func)
            })
            
            try:
                variant_question = selected_func(new_item)
                
                variant_list.append(variant_question)
                log_debug(f"变体 {idx+1} 生成成功", {
                    "variant_question_keys": list(variant_question.keys()) if variant_question else None
                })
                # # DEBUG: 生成唯一文件名
                # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                # output_dir = Path("/mmu_cd_ssd/zhangzhenyu06/workspace/Rebuttal/Help/")
                # output_dir.mkdir(parents=True, exist_ok=True)
                # output_path = output_dir / f"processed_image_{timestamp}_{str(selected_func)}.png"
                # image = ImageModule.open(BytesIO(variant_question['images']['bytes']))
                # image.save(output_path, format='PNG')
            except Exception as e:
                log_debug(f"变体 {idx+1} 生成失败", {
                    "error": str(e),
                    "delta": delta,
                    "func": selected_func.__name__ if hasattr(selected_func, '__name__') else str(selected_func)
                })
                raise
        else:
            log_debug(f"Warning: 未找到delta {delta} 对应的处理函数", {
                "delta": delta,
                "DIFFICULTY_FUNCTION_MAP_keys": list(DIFFICULTY_FUNCTION_MAP.keys())
            })
    
    log_debug("函数执行完成", {
        "final_variant_list_length": len(variant_list),
        "generated_variants_count": len(variant_list) - 1
    })
    
    return variant_list

# def generate_variants(
#     origin_item: Dict[str, Any],
#     variant_num: int,
#     allow_multiple_thinking: bool = True
# ) -> List[Dict[str, Any]]:
#     """
#     生成指定数量的题目变体
    
#     Args:
#         origin_item: 原始题目
#         variant_num: 变体数量
#         allow_multiple_thinking: 是否允许多个思考步骤变体
#     """
#     # 生成难度样本
#     difficulty_samples = generate_variant_difficulty_samples(
#         origin_item['difficulty'],
#         variant_num
#     )
    
    
#     # 计算难度差值
#     difficulty_deltas = [d - origin_item['difficulty'] for d in difficulty_samples]
    
#     # # 处理思考步骤变体（只允许一个）
#     # if not allow_multiple_thinking:
#     #     negative_found = False
#     #     processed_deltas = []
#     #     for delta in difficulty_deltas:
#     #         if delta < 0:
#     #             if not negative_found:
#     #                 processed_deltas.append(delta)
#     #                 negative_found = True
#     #             else:
#     #                 # 替换为随机正值
#     #                 processed_deltas.append(random.choice([1, 2, 3]))
#     #         else:
#     #             processed_deltas.append(delta)
#     #     difficulty_deltas = processed_deltas
    
#     # 生成变体
#     variant_list = [origin_item]
#     for delta in difficulty_deltas:
#         new_item = copy.deepcopy(origin_item)
        
#         # 获取并执行对应难度的处理函数
#         func_list = DIFFICULTY_FUNCTION_MAP.get(delta, [])
#         if func_list:
#             selected_func = random.choice(func_list)
#             variant_question = selected_func(new_item)
#             variant_list.append(variant_question)
    
#     return variant_list