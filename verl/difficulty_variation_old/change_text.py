import os
from openai import OpenAI
import pandas as pd
import re
import json
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import APITimeoutError

def extract_ids_from_json(file_path):
    """
    从JSON文件中提取所有字典的'id'值，组成列表返回
    
    参数:
        file_path (str): JSON文件路径
        
    返回:
        list: 包含所有id值的列表
    """
    try:
        # 读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # 检查数据是否是列表
        if not isinstance(data, list):
            raise ValueError("JSON文件内容应该是一个列表")
            
        # 提取所有id值
        id_list = [item['id'] for item in data if 'id' in item]
        
        return id_list
    
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
        return []
    except json.JSONDecodeError:
        print("错误：文件内容不是有效的JSON格式")
        return []
    except Exception as e:
        print(f"发生错误：{str(e)}")
        return []

def read_json_to_list(file_path):
    """
    读取 JSON 文件并将其内容赋值到 Python 列表
    
    参数:
        file_path (str): JSON 文件路径
        
    返回:
        list: 包含 JSON 数据的列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # 确保返回的是列表（如果 JSON 本身是列表）
        if not isinstance(data, list):
            data = [data]  # 如果不是列表，则转为单元素列表
        
        return data
    
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
        return []
    except json.JSONDecodeError:
        print("错误：文件内容不是有效的 JSON 格式")
        return []
    except Exception as e:
        print(f"发生错误：{str(e)}")
        return []


def request_to_qwen3(input_text):
    client = OpenAI(
        api_key="sk-**********************************",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen-max",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_text},
        ],
        extra_body={"enable_thinking": False},
    )
    response = completion.model_dump()
    return response["choices"][0]["message"]["content"]

def request_to_qwen3(input_text,input_image):
    client = OpenAI(
        api_key="sk-**********************************",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen-max",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_text},
        ],
        extra_body={"enable_thinking": False},
    )
    response = completion.model_dump()
    return response["choices"][0]["message"]["content"]


import base64
import imghdr
from openai import OpenAI

def request_to_qwen3_text_image(prompt_text, input_text, input_image):
    # 检测图像格式
    image_format = imghdr.what(None, h=input_image)
    
    # 将常见的格式名称映射到标准的MIME类型后缀
    format_mapping = {
        'jpeg': 'jpeg',
        'jpg': 'jpeg',  # jpg和jpeg实际上是同一种格式
        'png': 'png',
        'gif': 'gif',
        'bmp': 'bmp',
        # 可以添加更多格式映射
    }
    
    # 获取标准化的格式名称
    normalized_format = format_mapping.get(image_format.lower(), 'jpeg')  # 默认为jpeg
    
    # 将二进制图像数据编码为Base64
    base64_image = base64.b64encode(input_image).decode('utf-8')
    
    client = OpenAI(
        api_key="sk-**********************************",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen-vl-max-latest",
        messages=[
            {
                "role": "system", 
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": input_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{normalized_format};base64,{base64_image}"
                        }
                    },
                    {"type": "text", "text": input_text}
                    
                ]
            }
        ]
    )
    
    response = completion.model_dump()
    return response["choices"][0]["message"]["content"]


prompt_format = "Below I will provide a math problem. Please generate ONE semantically consistent variant of it without altering the correct answer. The wording should vary significantly. Keep the <image> tag in the sentences unchanged, and place the variant questions between <answer></answer> tags."
prompt_format_hard = "Below I will provide a math problem. Please generate ONE semantically consistent variant of it without altering the correct answer. Rewrite it in a more convoluted, verbose, or abstract manner—using advanced vocabulary, indirect phrasing, or intricate sentence structures—while ensuring the underlying mathematical meaning and correct answer remain unchanged. Preserve any <image> tags exactly as they appear, and enclose the modified problem between <answer></answer>."

prompt_format_5_diff_text_image='''Below I will provide a problem and a figure. Please generate 5 different versions of this problem that:
1. Maintain exactly the same correct answer
2. Use significantly different wording and sentence structure
3. Present each variant between <answer1></answer1> through <answer5></answer5> tags
Format each variant as:  
<answer1>First version</answer1>  
<answer2>Second version</answer2>
<answer3>Third version</answer3>
<answer4>Fourth version</answer4> 
<answer5>Fifth version</answer5>  
Original problem:  
'''

prompt_format_think_step="""Given the text, images, and answer, generate a step-by-step reasoning process that logically leads to the correct result in the <answer> step. Requirements:
Flexible step count (3-5 steps): Use only as many steps as needed—no forced extension. Label them clearly with <step1>, <step2>, etc.
Strict dependency on input: Base reasoning only on the provided text and images—do not reverse-engineer from the answer.
Image reference: Must incorporate details from the image, not just text.
Logical rigor: Ensure each step coherently supports the next, with no gaps or contradictions.
Format (example with 4 steps):
<step1> [Observation from text/image] </step1>
<step2> [Logical inference from step1 + image details] </step2>
<step3> [Logical inference from step2 + image details] </step3>
<step4> [Key conclusion leading to answer] </step4>
<answer> [Final answer] </answer>
Input:
"""
# 读取数据
df = pd.read_parquet("/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1/datasets/MMK12/data/MMK12_train.parquet")

# 初始化结果列表和失败记录列表
answer_list = []
failed_rows = []

# 成功和失败的文件路径
success_file = '/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1/verl/difficulty_variation/mmk12_train_text_think_step.json'
failed_file = '/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1/verl/difficulty_variation/mmk12_train_failed_rows.json'

answer_list = read_json_to_list(success_file)

# 使用tenacity库实现智能重试机制
@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((APITimeoutError, Exception)),
)

def safe_request_to_qwen3(prompt):
    try:
        return request_to_qwen3(prompt)
    except Exception as e:
        print(f"请求失败，将重试: {str(e)}")
        raise

def process_question(prompt):
    while True:
        try:
            response = safe_request_to_qwen3(prompt)
            match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
            image_matches = re.findall(r'<image>', response, re.DOTALL)
            
            if len(image_matches) != 1:
                print("prompt"+prompt)
                print("responce"+response)
                print("图片标签数量不正确，将重试...")
                continue
                
            if match:
                return match.group(1).strip()
                
        except Exception as e:
            print(f"处理问题时发生错误: {str(e)}")
            time.sleep(5)

def save_progress():
    # 保存成功处理的数据
    with open(success_file, 'w', encoding='utf-8') as f:
        json.dump(answer_list, f, ensure_ascii=False, indent=2)
    
    # 保存失败的行
    if failed_rows:
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_rows, f, ensure_ascii=False, indent=2)

ids = extract_ids_from_json('/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1/verl/difficulty_variation/mmk12_train_text_variant.json')
for index, row in df.iterrows():
    if row['id'] in ids:
        continue
    new_dict = {}
    question = prompt_format + row["problem"]
    hard_question = prompt_format_hard + row["problem"]
    
    try:
        # 处理普通问题变体
        variant = process_question(question)
        print(f"成功生成变体: {variant}")
        
        # # 处理困难问题变体
        # hard_variant = process_question(hard_question)
        # print(f"成功生成困难变体: {hard_variant}")
        
        # 保存结果
        new_dict["id"] = row["id"]
        new_dict["problem"] = row["problem"]
        new_dict["variant_problem"] = variant
        # new_dict["hard_variant_problem"] = hard_variant
        answer_list.append(new_dict)
        
        # 实时保存进度
        save_progress()
            
    except Exception as e:
        print(f"处理行 {index} 时发生严重错误: {str(e)}")
        # 记录失败的行
        failed_row = {
            "index": index,
            "id": row["id"],
            "problem": row["problem"],
            "error": str(e)
        }
        failed_rows.append(failed_row)
        
        # 保存失败记录
        save_progress()
        continue

# 最终保存所有数据
save_progress()
print("处理完成！")
print(f"成功处理 {len(answer_list)} 行，失败 {len(failed_rows)} 行")