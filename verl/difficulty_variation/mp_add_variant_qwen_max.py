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
import random
from openai import AzureOpenAI

prompt_format_5_variant_prompt_v1 = '''Generate 5 distinct variants of the following problem that:
1. Preserve the exact same correct answer as the original.
2. Use significantly different wording, sentence structure
3. You can vary the sentence length or use language to explain the content of the question—either simplifying it into an easier variant or complicating it with advanced language to create a harder variant—but ensure correctness.
4. Format of variants as, include <image> in the variants:
<variant1>[First variant's full text]</variant1>
<variant2>[Second variant's full text]</variant2>
...
<variant5>[Fifth variant's full text]</variant5>
Original Problem:
'''

prompt_format_5_variant_prompt_v2 = '''Generate 5 distinct variants of the following problem that:  
1. Preserve the **exact same correct answer** as the original.  
2. Use **significantly different wording, sentence structure**
3. You can adjust the sentence length—either making it concise (using streamlined language) or extending it (by explaining the question content in detail or complicating it with advanced language to increase difficulty)—but must ensure correctness.
4. Format of variants as, include <image> in the variants:  
<variant1>[First variant's full text]</variant1>  
<variant2>[Second variant's full text]</variant2>  
...  
<variant5>[Fifth variant's full text]</variant5>  
**Original Problem:**
'''

prompt_format_5_variant_prompt_v3 = '''Generate 5 distinct variants of the following problem that:  
1. Preserve the **exact same correct answer** as the original.  
2. Use **significantly different wording, sentence structure**
3. You can vary the sentence length or use language to explain the content of the question, but ensure correctness.
4. Format of variants as, include <image> in the variants:  
<variant1>[First variant's full text]</variant1>  
<variant2>[Second variant's full text]</variant2>  
...  
<variant5>[Fifth variant's full text]</variant5>  
**Original Problem:**
'''

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

def add_image_tag_if_missing(text: str) -> str:
    """
    检查字符串中是否有<image>标签，如果没有则在字符串前拼接:
    "As shown in the figure <image> , "
    
    参数:
        text (str): 输入字符串
        
    返回:
        str: 处理后的字符串
    """
    if "<image>" not in text:
        return f"As shown in the figure <image>. {text}"
    return text

def parse_variant(response: str) -> List[str]:
    logger.debug("Parsing steps from response")
    steps = []
    number = 1
    while True:
        patterns = [
            rf'<variant{number}>(.*?)</variant{number}>',

            rf'<variant{number}>(.*?)<variant{number}>',

            rf'</variant{number}>(.*?)</variant{number}>',

            rf'<variant{number}>(.*?)<variant{number+1}>',

            rf'<variant{number}>(.*?)</variant{number+1}>',

            rf'</variant{number}>(.*?)<variant{number+1}>',

            rf'</variant{number}>(.*?)</variant{number+1}>',
        ]
        found = False
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                steps.append(add_image_tag_if_missing(match.group(1).strip()))
                found = True
                break
        if not found:
            break
        number += 1
    if not steps:
        logger.warning("No steps found in response")
    else:
        logger.info(f"Found {len(steps)} steps in response")
    return steps

def request_to_qwen3_text(prompt_text, input_text):
    logger.debug("Preparing image for QWEN3 API request")
    client = OpenAI(
        api_key="sk-25678a0b18d24afa86d3185f736fd886",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    messages = [
        {
            "role": "system", 
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "text", "text": input_text}
            ]
        }
    ]
    logger.debug("Sending request to QWEN3 API")
    try:
        completion = client.chat.completions.create(
            model="qwen-vl-max-latest",
            messages=messages
        )
        response = completion.choices[0].message.content
        logger.debug("Successfully received response from QWEN3 API")
        return response
    except Exception as e:
        logger.error("Error in QWEN3 API request", exc_info=True)
        raise

class GeminiClient:
    def __init__(self):
        # 使用基于密钥的身份验证初始化 Azure OpenAI 客户端
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=subscription_key,
            api_version="2025-01-01-preview",
        )
        return
    
    def request_to_azure_text_only(self, prompt_text, input_text):
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
                {"type": "text", "text": prompt_text},
                {"type": "text", "text": input_text}
            ]
        }
    ]
        logger.debug("Sending text-only request to Azure API")
        try:
            response = self.client.chat.completions.create(
                model=deployment,
                messages=messages,
                max_completion_tokens=4000,
                stop=None,
                stream=False
            )
            result = response.choices[0].message.content
            logger.info("Successfully received response from Azure OpenAI API")
            return result
        except Exception as e:
            logger.error(f"Error in Azure OpenAI API request: {str(e)}")
            raise

def process_row(row: Dict[str, Any], gemini_client: GeminiClient, max_retries: int = 5) -> Dict[str, Any]:
    global prompt_format_5_variant_prompt_v1
    global prompt_format_5_variant_prompt_v2
    global prompt_format_5_variant_prompt_v3
    # prompt_format_list = [prompt_format_5_variant_prompt_v1,prompt_format_5_variant_prompt_v2,prompt_format_5_variant_prompt_v3]
    prompt_format_list = [prompt_format_5_variant_prompt_v2]
    row_id = row.get('id', 'unknown')
    logger.info(f"Starting processing for row ID: {row_id}")
    
    result = {
        'id': row_id,
        'problem': row['problem'],
        'original_answer': row['answer'],
        'variant': [],
        'answer': "",
        'status': 'failed',
        'retries': 0
    }
    
    retries = 0
    while retries < max_retries:
        try:
            logger.debug(f"Attempt {retries + 1}/{max_retries} for row {row_id}")
                        
            # Make first API request
            logger.info(f"Sending first request to Azure API for row {row_id}")
            start_time = time.time()
            first_response = gemini_client.request_to_azure_text_only(
                random.choice(prompt_format_list),
                row['problem']
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Received first response for row {row_id} in {elapsed_time:.2f} seconds")
            # Parse steps from first response
            first_variants = parse_variant(first_response)
            # Validate we got at least 3 steps in first response
            if len(first_variants) >= 4:
                result['variant'] = first_variants
                result['status'] = 'success'
                logger.info(f"First request successful with {len(first_variants)} variant for row {row_id}")
                break
            else:
                logger.warning(f"Insufficient steps in first response ({len(first_variants)}) for row {row_id}, retrying...")
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
    """Get set of already processed IDs from output file"""
    processed_ids = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            processed_ids = {r['id'] for r in results if 'id' in r and r['status']=='success'}
            logger.info(f"Loaded {len(processed_ids)} already processed IDs from {output_file}")
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
    output_file = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation/mmk12_train_variants.json"
    
    logger.info("Starting main processing")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    
    # Initialize GeminiClient
    gemini_client = GeminiClient()
    
    # Get already processed IDs
    processed_ids = get_processed_ids(output_file)
    
    # Load input data and filter out processed rows
    df = pd.read_parquet(input_file)
    logger.info(f"Loaded parquet file with {len(df)} rows")
    
    # Filter out already processed rows
    df = df[~df['id'].isin(processed_ids)]
    logger.info(f"After filtering, {len(df)} rows remain to be processed")
    
    # Load existing results from output file if it exists
    existing_results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            logger.info(f"Loaded {len(existing_results)} existing results from {output_file}")
        except Exception as e:
            logger.error(f"Error loading existing results from {output_file}: {str(e)}", exc_info=True)
    
    if len(df) == 0:
        logger.info("No new rows to process")
        return
    
    # Determine number of workers (processes)
    num_workers = 2
    output_dir = Path(output_file).parent
    output_prefix = Path(output_file).stem
    
    # Split data into chunks
    chunks = split_dataframe(df, num_workers)
    
    # Create output paths for each worker
    output_paths = [str(output_dir / f"{output_prefix}_worker_{i+1}.json") for i in range(len(chunks))]
    
    # Create and start processes
    processes = []
    for i, (chunk, path) in enumerate(zip(chunks, output_paths)):
        p = multiprocessing.Process(target=process_chunk, args=(chunk, path, gemini_client))
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
                # 先加载所有数据
                all_results = json.load(f)
                # 过滤出包含'correct_think'的项
                results = [r for r in all_results if r['status']=='success']
            combined_results.extend(results)
            logger.info(f"Loaded {len(results)} results from {path}")
        except Exception as e:
            logger.error(f"Error loading results from {path}: {str(e)}", exc_info=True)
    
    # Combine with existing results
    final_results = existing_results + combined_results
    
    # Save combined results to main output file
    if final_results:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved combined results ({len(final_results)} rows, including {len(existing_results)} existing and {len(combined_results)} new) to {output_file}")
    
    logger.info("Processing completed successfully")

if __name__ == "__main__":
    main()