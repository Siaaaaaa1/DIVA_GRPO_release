import os
from openai import OpenAI
import pandas as pd
import re
import json
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import APITimeoutError
import base64
import imghdr
from openai import OpenAI
import pandas as pd
import json
from typing import List, Dict, Any 
import os
import logging
import copy
import multiprocessing
from pathlib import Path
import re
from typing import List, Tuple
import logging
from openai import AzureOpenAI



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

# prompt_format_think_step="""You are a mathematician, statistician, and geometer. Below, I will present you with a math problem along with its accompanying diagram. Please carefully observe the details in the image.
# Given the text, images, generate a step-by-step reasoning process that logically leads to the correct result in the <answer> step. Requirements:
# Flexible step count (3-5 steps): Use only as many steps as needed—no forced extension. Label them clearly with <step1>, <step2>, etc.
# Strict dependency on input: Base reasoning only on the provided text and images—do not reverse-engineer from the answer.
# Image reference: Must incorporate details from the image, not just text.
# Logical rigor: Ensure each step coherently supports the next, with no gaps or contradictions.
# Describe in detail the information present in the diagram, including but not limited to: the positional relationships of points, the meaning of numerical values in the diagram, parallel lines...
# Do not include the prompt information in the response. Consider what mathematical theorems are available.
# Format (example with 4 steps):
# <step1> [Focus on the observation of the image, describe in detail the information present in the diagram] </step1>
# <step2> [Consider what mathematical theorems are available + Logical inference from step1 + text information and image details] </step2>
# <step3> [Consider what mathematical theorems are available + Logical inference from step2 + text information and image details] </step3>
# <step4> [Key conclusion leading to answer] </step4>
# \boxed{[final answer]}
# """
# prompt_format_think_step_2=f"""You are a mathematician, statistician, and geometer. Please carefully observe the details in the image.
# You have already provided the reasoning steps above. Do you think your answer is correct?
# Please revise or improve your reasoning steps based on the correct answer. 
# Emphasize! DO NOT include the correct answer in <step>. DO NOT any text like "because the answer is ." or "as per the correct answer".
# After refining, regenerate a step-by-step reasoning process that logically leads to the correct result in the <answer> step. Format (example with 4 steps):
# <step1> [Focus on the Observation of the image, describe in detail the information present in the diagram] </step1>
# <step2> [Consider what mathematical theorems are available + Logical inference from step1 + text information and image details] </step2>
# <step3> [Consider what mathematical theorems are available + Logical inference from step2 + text information and image details] </step3>
# <step4> [Key conclusion leading to answer] </step4>
# <answer> [Final answer] </answer>
# """

prompt_format_think_step=r"""You are a mathematician, statistician, and geometer. Below, I will present you with a math problem along with its accompanying diagram. Please carefully observe the details in the image.
Given the text, images, generate a step-by-step reasoning process that logically leads to the correct result in the \boxed{}. Requirements:
Flexible step count (3-5 steps): Use only as many steps as needed—no forced extension. Label them clearly with <step1>, <step2>, etc.
Strict dependency on input: Base reasoning only on the provided text and images.
Image reference: Must incorporate details from the image, not just text.
Logical rigor: Ensure each step coherently supports the next, with no gaps or contradictions.
Describe in detail the information present in the diagram, including but not limited to: the positional relationships of points, the meaning of numerical values in the diagram, parallel lines...
Do not include the prompt information in the response. Consider what mathematical theorems are available.
Please ensure that the thought process is broken down into steps and enclosed within <step></step>.
Format (example with 4 steps):
<step1> [Focus on the observation of the image, describe in detail the information present in the diagram] </step1>
<step2> [Consider what mathematical theorems are available + Logical inference from step1 + text information and image details] </step2>
<step3> [Consider what mathematical theorems are available + Logical inference from step2 + text information and image details] </step3>
<step4> [Key conclusion leading to answer] </step4>
\boxed{[final answer]}
"""

prompt_format_think_step_2 =r"""You are a mathematician, statistician, and geometer. Please carefully observe the details in the image.
You have already provided the reasoning steps and correct answer above.
Do you think your answer is correct? Revise or improve your reasoning steps based on the correct answer.
Emphasize!
- DO NOT include the correct answer in <step>.
- DO NOT reverse-engineer from the answer (e.g., avoid phrases like "because the answer is..." or "as per the correct answer").
After refining, regenerate a step-by-step reasoning process that logically leads to the correct result in the \boxed{}.
Please ensure that the thought process is broken down into steps and enclosed within <step></step>.
Format (example with 4 steps):
<step1> [Focus on observing the image; describe diagram details thoroughly] </step1>
<step2> [Apply relevant mathematical theorems + logical inference from Step 1 + text/image details] </step2>
<step3> [Apply relevant mathematical theorems + logical inference from Step 2 + text/image details] </step3>
<step4> [Key conclusion leading to the answer] </step4>
\boxed{[final answer]}
"""


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation/processing.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def parse_steps(response: str) -> List[str]:
    """Parse the steps from the API response by sequentially matching step tags starting from step1.
    Attempts to match in the following order for each step number:
    1. <stepN>content</stepN>
    2. <stepN>content<stepN>
    3. </stepN>content</stepN>
    4. <stepN>content<stepN+1>
    
    Returns:
        List of parsed step contents
    """
    logger.debug("Parsing steps from response")
    steps = []
    number = 1
    
    while True:
        # Try all possible patterns for current step number
        patterns = [
            # 1. Enclosed tags: <stepN>content</stepN>
            rf'<step{number}>(.*?)</step{number}>',
            # 2. Start to start: <stepN>content<stepN>
            rf'<step{number}>(.*?)<step{number}>',
            # 3. End to end: </stepN>content</stepN>
            rf'</step{number}>(.*?)</step{number}>',
            # 4. Start to next start: <stepN>content<step{number+1}>
            rf'<step{number}>(.*?)<step{number+1}>',

            rf'<step{number}>(.*?)</step{number+1}>',

            rf'</step{number}>(.*?)<step{number+1}>',

            rf'</step{number}>(.*?)</step{number+1}>',

            rf'<step{number}>(.*?)\boxed',
            
            rf'</step{number}>(.*?)\boxed',

            rf'<step{number}>(.*?)boxed',
            
            rf'</step{number}>(.*?)boxed'
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
    
    if not steps:
        print(response)
        logger.warning("No steps found in response")
    else:
        logger.info(f"Found {len(steps)} steps in response")
    
    return steps

def parse_answer(response: str) -> List[str]:
    """Parse the answer from the API response"""
    logger.debug("Parsing answer from response")
    match = re.search(r'.*boxed\{(.*?)\}[^}]*$', response)
    if match:
        extracted_content = match.group(1)
        logger.info(f"Found answer ({extracted_content}) in response")
        return extracted_content
    else:
        print(response)
        print("未找到匹配的内容")
        return ""

class GeminiClient:
    def __init__(self):
        # 使用基于密钥的身份验证初始化 Azure OpenAI 客户端
        self.client = OpenAI(
            api_key="sk-25678a0b18d24afa86d3185f736fd886",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        return

    def request_to_azure_text_image(self,prompt_text, ground_truth, 
                            input_text, input_image, 
                            prompt_text2=None, answer_step_1=None, first_request=True):
        """
        使用AzureOpenAI实现与request_to_qwen3_text_image相同的功能
        
        参数:
            prompt_text: 初始提示文本
            ground_truth: 正确答案
            input_text: 输入问题文本
            input_image: 二进制图像数据
            prompt_text2: 第二次请求的提示文本(可选)
            answer_step_1: 第一次请求的响应(可选)
            first_request: 是否为第一次请求(默认为True)
        
        返回:
            AzureOpenAI API的响应内容
        """
        try:
            logger.debug("Preparing image for QWEN3 API request")
    
            if not input_image:
                logger.warning("No image provided for text-image request")
            
            # Detect image format
            image_format = imghdr.what(None, h=input_image)
            logger.debug(f"Detected image format: {image_format}")
            
            # Map common format names to standard MIME type suffixes
            format_mapping = {
                'jpeg': 'jpeg',
                'jpg': 'jpeg',
                'png': 'png',
                'gif': 'gif',
                'bmp': 'bmp',
            }
            
            normalized_format = format_mapping.get(image_format.lower(), 'png')
            logger.info(f"Normalized image format: {normalized_format}")
            
            # Encode binary image data as Base64
            base64_image = base64.b64encode(input_image).decode('utf-8')
            logger.info(f"Base64 encoded image length: {len(base64_image)}")
            
            
            # 构建消息内容
            if first_request:
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{normalized_format};base64,{base64_image}"
                                }
                            },
                            {"type": "text", "text": "Problem: " + input_text}
                        ]
                    }
                ]
            else:
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{normalized_format};base64,{base64_image}"
                                }
                            },
                            {"type": "text", "text": "Problem: " + input_text}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": answer_step_1}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text":"The correct answer to this question is:" + ground_truth + "\n" + prompt_text2}
                        ]
                    }
                ]
            
            logger.info("Sending request to Azure OpenAI API")
            
            # 调用Azure OpenAI API
            response = self.client.chat.completions.create(
                model="qwen-vl-max-latest",
                messages=messages
            )
            # 获取响应内容
            result = response.choices[0].message.content
            logger.info("Successfully received response from Azure OpenAI API")
            print(result)
            return result
        except Exception as e:
            logger.error(f"Error in Azure OpenAI API request: {str(e)}")
            raise

    def request_to_azure_text_only(self,input_text):
        """
        Simplified function to send text-only requests to Azure API
        
        Args:
            prompt_text: The instruction/prompt text
            input_text: The input text to process
        
        Returns:
            The API response content
        """
        logger.debug("Preparing text-only request for Azure API")
        messages = [
            {
                "role": "system", 
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": input_text}
                ]
            }
        ]
        logger.debug("Sending text-only request to Azure API")
        try:
            response = self.client.chat.completions.create(
            model="qwen-vl-max-latest",
            messages=messages
            )
            result = response.choices[0].message.content
            logger.info("Successfully received response from Azure OpenAI API")
            return result
        except Exception as e:
            logger.error(f"Error in Azure OpenAI API request: {str(e)}")
            raise

def process_row(row: Dict[str, Any], gemini_client: GeminiClient, max_retries: int = 5) -> Dict[str, Any]:
    global prompt_format_think_step
    global prompt_format_5_diff_text_image
    global prompt_format_think_step_2
    """Process a single row from the DataFrame"""
    row_id = row.get('id', 'unknown')
    logger.info(f"Starting processing for row ID: {row_id}")
    
    result = {
        'id': row_id,
        'problem': row['problem'],
        'original_answer': row['answer'],
        'step1': [],
        'step2': [],
        'answer1': "",
        'answer2': "",
        'status': 'failed',
        'retries': 0
    }
    
    retries = 0
    while retries < max_retries:
        try:
            logger.debug(f"Attempt {retries + 1}/{max_retries} for row {row_id}")
            
            # Get image bytes (assuming it's stored as bytes in the 'images' column)
            image_bytes = row['images']['bytes'] if 'images' in row and 'bytes' in row['images'] else None
            logger.debug(f"Image bytes length: {len(image_bytes) if image_bytes else 0}")
            
            # Make first API request
            logger.info(f"Sending first request to Azure for row {row_id}")
            start_time = time.time()
            first_response = gemini_client.request_to_azure_text_image(
                prompt_format_think_step, 
                row['answer'], 
                row['problem'], 
                image_bytes,
                prompt_format_think_step_2,
                "",
                True,
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Received first response for row {row_id} in {elapsed_time:.2f} seconds")
            # Parse steps from first response
            first_steps = parse_steps(first_response)
            first_answer = parse_answer(first_response)
            # Validate we got at least 3 steps in first response
            if len(first_steps) >= 3 and first_answer != "":
                result['step1'] = first_steps
                result['answer1'] = first_answer
                logger.info(f"First request successful with {len(first_steps)} steps for row {row_id}")
                # Make second API request
                logger.info(f"Sending second request to Azure for row {row_id}")
                start_time = time.time()
                second_response = gemini_client.request_to_azure_text_image(
                    prompt_format_think_step, 
                    row['answer'], 
                    row['problem'], 
                    image_bytes, 
                    prompt_format_think_step_2,
                    first_response,
                    False
                )
                elapsed_time = time.time() - start_time
                logger.info(f"Received second response for row {row_id} in {elapsed_time:.2f} seconds")
                
                # Parse steps from second response
                second_steps = parse_steps(second_response)
                answer2 = parse_answer(second_response)
                # Validate we got at least 3 steps in second response
                if len(second_steps) >= 3 and answer2 != "":
                    result['step2'] = second_steps
                    result['answer2'] = answer2
                    result['status'] = 'success'
                    result['retries'] = retries
                    logger.info(f"Successfully processed row {row_id} with both requests")
                    
                    # Compare answer1 and answer2 values using a prompt
                    comparison_prompt = f"""
                    Compare these two answers numerically, ignoring any formatting differences:
                    Answer 1: {result['answer1']}
                    Answer 2: {result['answer2']}
                    
                    Extract just the numerical values from each answer and compare them. 
                    If the numerical values are the same, return True. Otherwise return False.
                    Reply with <answer>True</answer> or <answer>False</answer>
                    """
                    
                    # Call the model to compare answers
                    comparison_response = gemini_client.request_to_azure_text_only(comparison_prompt)
                    
                    def extract_last_answer(text):
                        pattern = r'<answer>(.*?)</answer>'
                        matches = re.findall(pattern, text)
                        return matches[-1] if matches else None

                    # Set the comparison result
                    result['correct_think'] = extract_last_answer(comparison_response.strip().lower()) == "true"
                    break
                else:
                    logger.warning(f"Insufficient steps in second response ({len(second_steps)}) for row {row_id}, retrying...")
                    logger.warning(f"second_response:({second_response})")
                    retries += 1
                    continue
            else:
                logger.warning(f"Insufficient steps in first response ({len(first_steps)}) for row {row_id}, retrying...")
                logger.warning(f"first_response:({first_response})")
                retries += 1
                continue
                
        except Exception as e:
            logger.error(f"Error processing row {row_id} (attempt {retries + 1}): {str(e)}", exc_info=True)
            retries += 1
            continue
    
    logger.info(f"Successfully processed row {row_id}")

    return result

def process_chunk(chunk: pd.DataFrame, output_path: str, gemini_client: GeminiClient) -> None:
    """Process a chunk of data and save results to specified output path"""
    logger.info(f"Starting to process chunk with {len(chunk)} rows, saving to {output_path}")
    
    results = []
    for _, row in chunk.iterrows():
        try:
            result = process_row(row.to_dict(), gemini_client)
            results.append(result)
            
            # Save after each successful processing
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved results to {output_path} after processing row {row.get('id', 'unknown')}")
                
        except Exception as e:
            row_id = row.get('id', 'unknown')
            logger.error(f"Error processing row {row_id}: {str(e)}", exc_info=True)
            continue
    
    success_count = len([r for r in results if r.get('status') == 'success'])
    logger.info(f"Chunk processing complete. Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")

def get_processed_ids(output_file: str) -> set:
    """Get set of already processed IDs from output file that have 'correct_think' key"""
    processed_ids = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            # 只包含有id且有correct_think键的项
            processed_ids = {r['id'] for r in results if 'id' in r and r['status']=='success'}
            logger.info(f"Loaded {len(processed_ids)} already processed IDs (with 'correct_think') from {output_file}")
        except Exception as e:
            logger.error(f"Error reading existing output file: {str(e)}", exc_info=True)
    return processed_ids

def split_dataframe(df: pd.DataFrame, num_chunks: int) -> List[pd.DataFrame]:
    """Split DataFrame into approximately equal chunks"""
    chunk_size = len(df) // num_chunks
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    logger.info(f"Split DataFrame into {len(chunks)} chunks")
    return chunks

def main():
    input_file = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/data/MMK12_train.parquet"
    output_file = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation/mmk12_train_think_steps_gpto3.json"
    
    logger.info("Starting main processing")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    
    # Create GeminiClient instance
    gemini_client = GeminiClient()
    
    # Get already processed IDs
    processed_ids = get_processed_ids(output_file)
    
    # Load input data and filter out processed rows
    df = pd.read_parquet(input_file)
    logger.info(f"Loaded parquet file with {len(df)} rows")
    
    # Filter out already processed rows
    df = df[~df['id'].isin(processed_ids)]
    logger.info(f"After filtering, {len(df)} rows remain to be processed")

    existing_results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                # 先加载所有数据
                all_results = json.load(f)
                # 过滤出包含'correct_think'的项
                existing_results = [r for r in all_results if r['status']=='success']
            logger.info(f"Loaded {len(existing_results)} existing results (with 'correct_think') from {output_file}")
        except Exception as e:
            logger.error(f"Error loading existing results from {output_file}: {str(e)}", exc_info=True)
    
    if len(df) == 0:
        logger.info("No new rows to process")
        return
    
    # Determine number of workers (processes)
    num_workers = 20
    output_dir = Path(output_file).parent
    output_prefix = Path(output_file).stem
    
    # Split data into chunks
    chunks = split_dataframe(df, num_workers)
    
    # Create output paths for each worker
    output_paths = [str(output_dir / f"{output_prefix}_worker_{i+1}.json") for i in range(len(chunks))]
    
    # Create and start processes
    processes = []
    for i, (chunk, path) in enumerate(zip(chunks, output_paths)):
        # Each process gets its own GeminiClient instance
        p = multiprocessing.Process(target=process_chunk, args=(chunk, path, GeminiClient()))
        processes.append(p)
        p.start()
        logger.info(f"Started worker {i+1} processing {len(chunk)} rows")
    
    # Wait for all processes to complete
    for i, p in enumerate(processes):
        p.join()
        logger.info(f"Worker {i+1} completed processing")
    
    # Combine results from all workers
    combined_results = []
    for path in output_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            combined_results.extend(results)
            logger.info(f"Loaded {len(results)} results from {path}")
        except Exception as e:
            logger.error(f"Error loading results from {path}: {str(e)}", exc_info=True)
    
    final_results = existing_results + combined_results

    # Save combined results to main output file
    if final_results:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved combined results ({len(final_results)} rows, including {len(existing_results)} existing and {len(combined_results)} new) to {output_file}")
    
    logger.info("Processing completed successfully")

if __name__ == "__main__":
    main()