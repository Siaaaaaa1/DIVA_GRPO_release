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

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from . import torch_functional as VF
from ..utils.gen_new_item import generate_variants
import json
import random
import warnings
import logging
from functools import partial
import copy

def collate_fn(features: List[Union[Dict[str, Any], List[Dict[str, Any]]]]) -> Dict[str, Any]:
    #TODO: 处理嵌套列表的情况#######################
    if isinstance(features[0], list):
        features = features[0]  # 解包嵌套列表
    #############################################

    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}

# TODO: 此处需不需要新的collate函数？
def collate_fn_DA(features: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    处理128次__getitem__调用的结果（每次返回8个样本，共1024个样本）
    输入features: List[List[Dict]]，外层List长度为128，每个内层List包含8个Dict
    输出: Dict[str, Any]，其中张量字段已堆叠，非张量字段转为numpy数组
    """
    # 调试信息
    print(f"输入features类型: {type(features)}")
    print(f"输入features长度（__getitem__调用次数）: {len(features)}")
    if features:
        print(f"第一个__getitem__返回的样本数: {len(features[0])}")
        print(f"第一个样本的字段示例: {list(features[0][0].keys())}")

    # 展平嵌套结构（128 * 8=1024个样本）
    flat_features = [item for sublist in features for item in sublist]
    print(f"展平后的总样本数: {len(flat_features)}")

    # 分离张量与非张量字段
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    
    for feature in flat_features:
        for key, value in feature.items():
            if torch.is_tensor(value):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)
    
    # 合并张量字段（自动处理不同形状的张量）
    for key in list(tensors.keys()):
        try:
            # 尝试堆叠（适用于大多数情况）
            tensors[key] = torch.stack(tensors[key], dim=0)  # shape: [1024, ...]
        except RuntimeError:
            # 如果张量形状不一致，改为拼接（适用于如边界框等可变长度数据）
            tensors[key] = torch.cat(tensors[key], dim=0)
            print(f"警告: 字段 {key} 使用了torch.cat而非stack")
    
    # 合并非张量字段
    for key in non_tensors:
        try:
            non_tensors[key] = np.array(non_tensors[key], dtype=object)
        except Exception as e:
            print(f"无法将字段 {key} 转为numpy数组: {str(e)}")
            non_tensors[key] = non_tensors[key]  # 保持原样
    
    return {**tensors, **non_tensors}

# import math
# import logging
# import os
# from pathlib import Path
# from typing import Union, Any, Optional
# from datetime import datetime
# from io import BytesIO
# from PIL import Image as ImageModule
# from PIL.Image import Image as ImageObject

# def process_image(
#     image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
# ) -> ImageObject:
#     """
#     处理图像并保存为PNG格式，同时记录详细日志
#     """
#     # 配置日志记录器
#     log_path = Path("/mmu_cd_ssd/zhangzhenyu06/workspace/Rebuttal/Help/image.log")
#     log_path.parent.mkdir(parents=True, exist_ok=True)
    
#     logging.basicConfig(
#         level=logging.DEBUG,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(log_path, encoding='utf-8'),
#             logging.StreamHandler()  # 同时输出到控制台
#         ]
#     )
#     logger = logging.getLogger(__name__)
    
#     # 生成唯一文件名
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#     output_dir = Path("/mmu_cd_ssd/zhangzhenyu06/workspace/Rebuttal/Help/")
#     output_dir.mkdir(parents=True, exist_ok=True)
#     output_path = output_dir / f"processed_image_{timestamp}.png"
    
#     logger.info("=" * 60)
#     logger.info(f"开始处理图像 - 时间戳: {timestamp}")
#     logger.info(f"输入参数 - min_pixels: {min_pixels}, max_pixels: {max_pixels}")
    
#     try:
#         # 步骤1: 加载图像
#         logger.debug("步骤1: 加载图像")
#         if isinstance(image, str):
#             logger.debug(f"  - 输入为字符串路径: {image}")
#             image = ImageModule.open(image)  # 使用模块的open函数
#             logger.debug(f"  - 成功从路径加载图像")
#         elif isinstance(image, dict):
#             logger.debug("  - 输入为字典格式，包含bytes数据")
#             image = ImageModule.open(BytesIO(image["bytes"]))  # 使用模块的open函数
#             logger.debug(f"  - 成功从字典bytes加载图像")
#         elif isinstance(image, bytes):
#             logger.debug("  - 输入为bytes格式")
#             image = ImageModule.open(BytesIO(image))  # 使用模块的open函数
#             logger.debug(f"  - 成功从bytes加载图像")
#         else:
#             logger.debug("  - 输入已为ImageObject对象，直接使用")
        
#         # 步骤2: 避免"Too many open files"错误
#         logger.debug("步骤2: 调用image.load()避免文件句柄问题")
#         image.load()
#         logger.debug(f"  - 图像尺寸: {image.width} x {image.height}, 模式: {image.mode}")
        
#         # 步骤3: 检查并调整最大像素限制
#         current_pixels = image.width * image.height
#         logger.debug(f"步骤3: 检查最大像素限制")
#         logger.debug(f"  - 当前像素数: {current_pixels}, 最大限制: {max_pixels}")
        
#         if max_pixels is not None and current_pixels > max_pixels:
#             logger.debug(f"  - 超过最大像素限制，开始缩小图像")
#             resize_factor = math.sqrt(max_pixels / current_pixels)
#             width, height = int(image.width * resize_factor), int(image.height * resize_factor)
#             logger.debug(f"  - 缩放因子: {resize_factor:.4f}, 新尺寸: {width} x {height}")
#             image = image.resize((width, height))
#             logger.debug(f"  - 缩小操作完成")
#         else:
#             logger.debug(f"  - 未超过最大像素限制，跳过缩小")
        
#         # 步骤4: 检查并调整最小像素限制
#         current_pixels = image.width * image.height
#         logger.debug(f"步骤4: 检查最小像素限制")
#         logger.debug(f"  - 当前像素数: {current_pixels}, 最小限制: {min_pixels}")
        
#         if min_pixels is not None and current_pixels < min_pixels:
#             logger.debug(f"  - 低于最小像素限制，开始放大图像")
#             resize_factor = math.sqrt(min_pixels / current_pixels)
#             width, height = int(image.width * resize_factor), int(image.height * resize_factor)
#             logger.debug(f"  - 缩放因子: {resize_factor:.4f}, 新尺寸: {width} x {height}")
#             image = image.resize((width, height))
#             logger.debug(f"  - 放大操作完成")
#         else:
#             logger.debug(f"  - 满足最小像素限制，跳过放大")
        
#         # 步骤5: 转换为RGB模式
#         logger.debug(f"步骤5: 检查并转换图像模式")
#         logger.debug(f"  - 当前模式: {image.mode}")
#         if image.mode != "RGB":
#             logger.debug(f"  - 非RGB模式，开始转换")
#             image = image.convert("RGB")
#             logger.debug(f"  - 模式转换完成，新模式: {image.mode}")
#         else:
#             logger.debug(f"  - 已为RGB模式，跳过转换")
        
#         # 步骤6: 保存为PNG格式
#         logger.debug(f"步骤6: 保存处理后的图像为PNG格式")
#         logger.debug(f"  - 保存路径: {output_path}")
#         logger.debug(f"  - 最终图像信息: 尺寸={image.width}x{image.height}, 模式={image.mode}")
#         image.save(output_path, format='PNG')
#         logger.info(f"  - 图像成功保存为: {output_path.name}")
        
#         logger.info("图像处理流程完成")
#         logger.info("=" * 60)
        
#         return image
        
#     except Exception as e:
#         logger.error(f"图像处理过程中发生错误: {str(e)}", exc_info=True)
#         logger.info("=" * 60)
#         raise
def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_video(
    video: str, min_pixels: Optional[int], max_pixels: Optional[int], video_fps: float, return_fps: bool = False
) -> Union[list[ImageObject], tuple[list[ImageObject], list[float]]]:
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "fps": video_fps}
    return fetch_video(vision_info, return_video_sample_fps=return_fps)


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
        DIVA_GRPO: Optional[bool] = False,
        Variant_Num: Optional[int] = 0,
        Dataset_Mode: Optional[str] = None
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.DIVA_GRPO = DIVA_GRPO
        self.Variant_Num = Variant_Num
        self.Dataset_Mode = Dataset_Mode

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        # TODO: 此处缺少让format_prompt = None的代码，需要加吗？
        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        # if self.DIVA_GRPO == False or filter_overlong_prompts:
        # if filter_overlong_prompts:
        if self.DIVA_GRPO == False or filter_overlong_prompts:
            print(f"Before overlong prompts filter datasets length is === {len(self.dataset)}")
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )
            print(f"After overlong prompts filter datasets length is === {len(self.dataset)}")

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            # TODO: 由于数据集中只有一个图片，所以只使用第一个分割##################
            parts = prompt_str.split("<image>", 1)  # 只分割一次
            if parts[0]:  # 第一个部分（<image>之前的内容）
                content_list.append({"type": "text", "text": parts[0]})
            if len(parts) > 1:  # 如果有<image>标记
                content_list.append({"type": "image"})
                if parts[1]:  # <image>之后的内容
                    content_list.append({"type": "text", "text": parts[1]})
            return [{"role": "user", "content": content_list}]
            ##########################  ORIGIN CODE  ########################
            # for i, content in enumerate(prompt_str.split("<image>")):
            #     if i != 0:
            #         content_list.append({"type": "image"})

            #     if content:
            #         content_list.append({"type": "text", "text": content})

            # return [{"role": "user", "content": content_list}]
            #################################################################
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            # TODO:添加image处理代码
            if isinstance(images, dict):
                processed_images.append(process_image(images, self.min_pixels, self.max_pixels))
            else:
                for image in images:
                    processed_images.append(process_image(image, self.min_pixels, self.max_pixels))
            # for image in images:
            #     processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(process_video(video, self.min_pixels, self.max_pixels, self.video_fps))

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def update_difficulty(self, updates):
        """
        批量更新样本的difficulty值（安全且高效版）
        
        参数:
            updates: 包含(uid, new_difficulty)元组的列表
            
        返回:
            bool: 是否成功更新所有样本
        """
        # 将updates转换为字典便于快速查找
        uid_to_diff = {str(uid): new_diff for uid, new_diff in updates}
        print(f"\n[DEBUG] Starting difficulty update for {len(updates)} samples...")

        # 更新前的统计信息
        before_diff_dist = dict(zip(*np.unique(self.dataset['difficulty'], return_counts=True)))
        
        # 定义更新函数
        def apply_update(example):
            uid = str(example.get('id'))
            if uid in uid_to_diff:
                example['difficulty'] = uid_to_diff[uid]
            return example

        # 执行批量更新（自动并行处理）
        self.dataset = self.dataset.map(
            apply_update, num_proc=4, desc="Updating difficulties"
        )

        # 检查未找到的ID
        all_ids = set(str(x) for x in self.dataset['id'])
        missing_ids = [uid for uid in uid_to_diff if uid not in all_ids]

        # 输出更新结果
        after_diff_dist = dict(zip(*np.unique(self.dataset['difficulty'], return_counts=True)))
        print(f"\n[DEBUG] Update summary:")
        print(f"  - Total updates attempted: {len(updates)}")
        print(f"  - Successfully updated: {len(updates) - len(missing_ids)}")
        print(f"  - Not found samples: {len(missing_ids)}")
        if missing_ids:
            print(f"    - Missing IDs: {', '.join(missing_ids[:5])}{'...' if len(missing_ids) > 5 else ''}")

        # 随机验证样本（可选）
        if updates and not missing_ids:
            print("\n[DEBUG] Random sample verification:")
            import random
            test_cases = random.sample(updates, min(5, len(updates)))
            for uid, expected_diff in test_cases:
                idx = self.dataset['id'].index(uid)
                actual_diff = self.dataset[idx]['difficulty']
                status = "✓" if actual_diff == expected_diff else "✗"
                print(f"  - ID {uid}: expected {expected_diff}, got {actual_diff} [{status}]")
        return not missing_ids

    def _build_variant_messages(self, example):
        """为变体示例构建消息"""
        prompt_str = example[self.prompt_key]
        
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)
        
        if self.image_key in example:
            content_list = []
            #### TODO：只分割一次image
            parts = prompt_str.split("<image>", 1)  # 只分割一次
            content_list = []
            if parts[0]:  # 第一个部分（<image>之前的内容）
                content_list.append({"type": "text", "text": parts[0]})
            if len(parts) > 1:  # 如果有<image>标记
                content_list.append({"type": "image"})
                if parts[1]:  # <image>之后的内容
                    content_list.append({"type": "text", "text": parts[1]})
            return [{"role": "user", "content": content_list}]
            #### ORIGIN CODE
            # for i, content in enumerate(prompt_str.split("<image>")):
            #     if i != 0:
            #         content_list.append({"type": "image"})
            #     if content:
            #         content_list.append({"type": "text", "text": content})
            # return [{"role": "user", "content": content_list}]
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})
                if content:
                    content_list.append({"type": "text", "text": content})
            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _process_image_data(self, example, messages):
        """处理包含图像的数据"""
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        images = example.pop(self.image_key)
        
        # 处理图像路径
        if self.image_dir is not None and images and isinstance(images[0], str):
            images = [os.path.join(self.image_dir, image) for image in images]
        
        # 处理图像数据
        processed_images = [] if len(images) != 0 else None  # text-only data
        if isinstance(images, dict):
            processed_images.append(process_image(images, self.min_pixels, self.max_pixels))
        else:
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))
        
        # 获取模型输入
        model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        
        # 保存多模态数据
        example["multi_modal_data"] = {"images": images}
        
        return self._finalize_example(example, model_inputs, input_ids, attention_mask, prompt)

    def _process_video_data(self, example, messages):
        """处理包含视频的数据"""
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        videos = example.pop(self.video_key)
        
        # 处理视频路径
        if self.image_dir is not None and videos and isinstance(videos[0], str):
            videos = [os.path.join(self.image_dir, video) for video in videos]
        
        # 处理视频数据
        processed_videos = [] if len(videos) != 0 else None
        video_fps_list = []
        for video in videos:
            processed_video, video_fps = process_video(
                video, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
            )
            processed_videos.append(processed_video)
            video_fps_list.append(video_fps)
        
        # 获取模型输入
        model_inputs = self.processor(
            videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
        )
        
        # Qwen2VL特殊处理
        if "second_per_grid_ts" in self.processor.model_input_names:
            model_inputs["second_per_grid_ts"] = [2.0 / fps for fps in video_fps_list]
        
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        
        # 保存多模态数据
        example["multi_modal_data"] = {"videos": videos}
        
        return self._finalize_example(example, model_inputs, input_ids, attention_mask, prompt)

    def _process_text_data(self, example, messages):
        """处理纯文本数据"""
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        
        return self._finalize_example(example, model_inputs, input_ids, attention_mask, prompt)
        
    # 处理单个示例的通用方法
    def _process_single_example(self, example, is_variant=False):
        copy_example = copy.deepcopy(example)
        messages = self._build_messages(example) if not is_variant else self._build_variant_messages(example)
        if self.image_key in example:
            return self._process_image_data(example, messages)
        elif self.video_key in example:
            return self._process_video_data(example, messages)
        else:
            return self._process_text_data(example, messages)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        # print(f"1. Available keys in example: {list(example.keys())}")
        # print(f"1. Expected answer_key: {self.answer_key}")
        if self.DIVA_GRPO == True:
            if self.Dataset_Mode.startswith("one_thinking"):
                varient_list = generate_variants(example, self.Variant_Num, allow_multiple_thinking=True)
                return [self._process_single_example(new_example, is_variant=True) for new_example in varient_list]
            else:
                # print("DEBUG!!!")
                # print(f"2. Available keys in example: {list(example.keys())}")
                # print(f"2. Expected answer_key: {self.answer_key}")
                varient_list = generate_variants(example, self.Variant_Num, allow_multiple_thinking=False)
                return [self._process_single_example(new_example, is_variant=True) for new_example in varient_list]
        else:
            return self._process_single_example(example)

    def _finalize_example(self, example, model_inputs, input_ids, attention_mask, prompt):
        """完成示例处理的最后步骤"""
        # 生成位置ID
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)
        
        # 处理过长的输入
        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        
        # 处理原始prompt tokens
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")
        
        # 组装最终示例
        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example[self.answer_key]
        
        return example

    # TODO: 下面是原本的getitem实现
    # def __getitem__(self, index):
    #     example: dict = self.dataset[index]
    #     messages = self._build_messages(example)
    #     example.pop(self.prompt_key, None)

    #     if self.image_key in example:
    #         prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    #         images = example.pop(self.image_key)
    #         if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
    #             images = [os.path.join(self.image_dir, image) for image in images]

    #         processed_images = [] if len(images) != 0 else None  # text-only data
    #         for image in images:
    #             processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

    #         model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
    #         input_ids = model_inputs.pop("input_ids")[0]
    #         attention_mask = model_inputs.pop("attention_mask")[0]
    #         example["multi_modal_data"] = {"images": images}
    #     elif self.video_key in example:
    #         prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    #         videos = example.pop(self.video_key)
    #         if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
    #             videos = [os.path.join(self.image_dir, video) for video in videos]

    #         processed_videos = [] if len(videos) != 0 else None  # text-only data
    #         video_fps_list = []
    #         for video in videos:
    #             processed_video, video_fps = process_video(
    #                 video, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
    #             )
    #             processed_videos.append(processed_video)
    #             video_fps_list.append(video_fps)

    #         model_inputs = self.processor(
    #             videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
    #         )
    #         if "second_per_grid_ts" in self.processor.model_input_names:
    #             model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

    #         input_ids = model_inputs.pop("input_ids")[0]
    #         attention_mask = model_inputs.pop("attention_mask")[0]
    #         example["multi_modal_data"] = {"videos": videos}
    #     else:
    #         prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    #         model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
    #         input_ids = model_inputs.pop("input_ids")[0]
    #         attention_mask = model_inputs.pop("attention_mask")[0]

    #     if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
    #         # qwen-vl mrope
    #         if "Qwen3VLProcessor" in self.processor.__class__.__name__:
    #             from ..models.transformers.qwen3_vl import get_rope_index
    #         else:
    #             from ..models.transformers.qwen2_vl import get_rope_index

    #         vision_position_ids = get_rope_index(
    #             self.processor,
    #             input_ids=input_ids,
    #             image_grid_thw=model_inputs.get("image_grid_thw", None),
    #             video_grid_thw=model_inputs.get("video_grid_thw", None),
    #             second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
    #             attention_mask=attention_mask,
    #         )  # (3, seq_length)
    #         text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
    #         position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
    #     else:
    #         position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

    #     input_ids, attention_mask, position_ids = VF.postprocess_data(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         position_ids=position_ids,
    #         max_length=self.max_prompt_length,
    #         pad_token_id=self.tokenizer.pad_token_id,
    #         left_pad=True,
    #         truncation=self.truncation,
    #     )
    #     raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
    #     if len(raw_prompt_ids) > self.max_prompt_length:
    #         if self.truncation == "left":
    #             raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
    #         elif self.truncation == "right":
    #             raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
    #         elif self.truncation == "error":
    #             raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

    #     example["input_ids"] = input_ids
    #     example["attention_mask"] = attention_mask
    #     example["position_ids"] = position_ids
    #     example["raw_prompt_ids"] = raw_prompt_ids
    #     example["ground_truth"] = example.pop(self.answer_key)
    #     return example
