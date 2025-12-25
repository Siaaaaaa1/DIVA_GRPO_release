import os
import re
import time
import json
import base64
import logging
import random
import imghdr
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================= Prompts =================
PROMPT_VARIANTS = '''Generate 5 distinct variants of the following problem that:  
1. Preserve the **exact same correct answer** as the original.  
2. Use **significantly different wording, sentence structure**
3. You can adjust the sentence length—either making it concise or extending it—but must ensure correctness.
4. Format of variants as, include <image> in the variants if original has it:  
<variant1>[First variant's full text]</variant1>  
<variant2>[Second variant's full text]</variant2>  
...  
<variant5>[Fifth variant's full text]</variant5>  
**Original Problem:**
'''

PROMPT_THINK_STEP_INIT = r'''You are a mathematician, statistician, and geometer. Below, I will present you with a math problem along with its accompanying diagram (if any).
Given the text, images, generate a step-by-step reasoning process that logically leads to the correct result in the \boxed{}. Requirements:
Flexible step count (3-5 steps): Use only as many steps as needed. Label them clearly with <step1>, <step2>, etc.
Strict dependency on input: Base reasoning only on the provided text and images.
Image reference: Must incorporate details from the image, not just text.
Logical rigor: Ensure each step coherently supports the next.
Please ensure that the thought process is broken down into steps and enclosed within <step></step>.
Format (example with 4 steps):
<step1> [Focus on the observation of the image, describe in detail the information present in the diagram] </step1>
<step2> [Consider what mathematical theorems are available + Logical inference from step1 + text information and image details] </step2>
<step3> [Logical inference continues] </step3>
<step4> [Key conclusion leading to answer] </step4>
\boxed{[final answer]}
'''

PROMPT_THINK_STEP_REFINE = r'''You are a mathematician. You have already provided the reasoning steps above.
Do you think your answer is correct? Please revise or improve your reasoning steps based on the correct answer provided below.
Emphasize!
- DO NOT include the correct answer in <step>.
- DO NOT reverse-engineer from the answer.
After refining, regenerate a step-by-step reasoning process that logically leads to the correct result in the \boxed{}.
Format:
<step1> ... </step1>
...
\boxed{[final answer]}
'''

# ================= Parsers =================

def add_image_tag_if_missing(text: str) -> str:
    if "<image>" not in text:
        return f"As shown in the figure <image>. {text}"
    return text

def parse_variants(response: str) -> List[str]:
    variants = []
    number = 1
    while True:
        # 尝试匹配多种格式
        patterns = [
            rf'<variant{number}>(.*?)</variant{number}>',
            rf'<variant{number}>(.*?)(?=<variant{number+1}>)',
        ]
        found = False
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                variants.append(add_image_tag_if_missing(match.group(1).strip()))
                found = True
                break
        if not found:
            break
        number += 1
    return variants

def parse_think_steps(response: str) -> List[str]:
    steps = []
    number = 1
    while True:
        patterns = [
            rf'<step{number}>(.*?)</step{number}>',
            rf'<step{number}>(.*?)(?=<step{number+1}>)',
            rf'<step{number}>(.*?)\boxed',
        ]
        found = False
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                steps.append(match.group(1).strip())
                found = True
                break
        if not found:
            break
        number += 1
    return steps

def parse_boxed_answer(response: str) -> str:
    match = re.search(r'boxed\{(.*?)\}', response) # 简化匹配，原正则较复杂
    if not match:
        match = re.search(r'boxed\s+(.*)', response)
    return match.group(1) if match else ""

# ================= API Base Class =================

class BaseLLMClient(ABC):
    """API 调用基类"""
    
    @abstractmethod
    def generate_content(self, messages: List[Dict[str, Any]]) -> str:
        pass

    def process_image(self, image_bytes: bytes) -> str:
        """Helper to convert bytes to base64 string with format detection"""
        if not image_bytes:
            return ""
        fmt = imghdr.what(None, h=image_bytes) or 'png'
        # 简单映射
        fmt_map = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png'}
        fmt = fmt_map.get(fmt, 'png')
        b64_img = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:image/{fmt};base64,{b64_img}"

# ================= Concrete Implementation =================

class AzureQwenClient(BaseLLMClient):
    def __init__(self, api_key: str, endpoint: str, deployment: str = "qwen-vl-max-latest"):
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version="2025-01-01-preview", # 根据实际情况调整
        )
        self.deployment = deployment

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_content(self, messages: List[Dict[str, Any]]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                max_completion_tokens=4000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise

    def generate_variants(self, problem_text: str) -> List[str]:
        """生成 Variants"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": PROMPT_VARIANTS + problem_text}
        ]
        response = self.generate_content(messages)
        return parse_variants(response)

    def generate_think_steps(self, problem_text: str, correct_answer: str, image_bytes: bytes = None) -> Dict[str, Any]:
        """生成 Think Steps (Initial + Refine)"""
        
        # 1. 构造初始请求
        content = [{"type": "text", "text": PROMPT_THINK_STEP_INIT}]
        
        if image_bytes:
            img_url = self.process_image(image_bytes)
            content.append({"type": "image_url", "image_url": {"url": img_url}})
            
        content.append({"type": "text", "text": "Problem: " + problem_text})
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ]
        
        # 第一次生成
        first_response = self.generate_content(messages)
        first_steps = parse_think_steps(first_response)
        first_ans = parse_boxed_answer(first_response)
        
        if not first_steps:
            return {"status": "failed", "reason": "No steps in first response"}

        # 2. 构造 Refine 请求 (Multi-turn conversation)
        messages.append({"role": "assistant", "content": first_response})
        messages.append({
            "role": "user", 
            "content": f"The correct answer to this question is: {correct_answer}\n{PROMPT_THINK_STEP_REFINE}"
        })

        # 第二次生成
        second_response = self.generate_content(messages)
        second_steps = parse_think_steps(second_response)
        second_ans = parse_boxed_answer(second_response)

        if not second_steps:
            return {"status": "failed", "reason": "No steps in second response"}

        return {
            "status": "success",
            "initial_steps": first_steps,
            "initial_answer": first_ans,
            "refined_steps": second_steps,  # 这是最终想要的高质量 CoT
            "refined_answer": second_ans
        }