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
false_list = []
false_list_len = 0
output_false_file = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation/qwen_max_false.json"
def parse_steps(response: str) -> List[str]:
    """Parse the steps from the API response"""
    logger.debug("Parsing steps from response")
    step_pattern = r'<step\d+>(.*?)</step\d+>'
    steps = re.findall(step_pattern, response, re.DOTALL)
    logger.info(f"Found {len(steps)} steps in response")
    return [step.strip() for step in steps]

def parse_answer(response: str) -> List[str]:
    """Parse the steps from the API response"""
    logger.debug("Parsing steps from response")
    answer_pattern = r'<answer>(.*?)</answer>'
    answer = re.findall(answer_pattern, response, re.DOTALL)
    if len(answer)==1:
        answer = answer[0].strip()
    logger.info(f"Found answer ({answer}) in response")
    return answer

def process_row(row: Dict[str, Any], max_retries: int = 5) -> Dict[str, Any]:
    global prompt_format_think_step
    global prompt_format_5_diff_text_image
    global prompt_format_think_step_2
    global false_list
    global output_false_file
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
            logger.info(f"Sending first request to QWEN3 for row {row_id}")
            start_time = time.time()
            first_response = request_to_qwen3_text_image(
                prompt_format_think_step, 
                row['answer'], 
                row['problem'], 
                image_bytes,
                prompt_format_think_step_2,
                "",
                first_request=True,
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Received first response for row {row_id} in {elapsed_time:.2f} seconds")
            # Parse steps from first response
            first_steps = parse_steps(first_response)
            first_answer = parse_answer(first_response)
            # Validate we got at least 3 steps in first response
            if len(first_steps) >= 3:
                result['step1'] = first_steps
                result['answer1'] = first_answer
                logger.info(f"First request successful with {len(first_steps)} steps for row {row_id}")
                # Make second API request
                logger.info(f"Sending second request to QWEN3 for row {row_id}")
                start_time = time.time()
                second_response = request_to_qwen3_text_image(
                    prompt_format_think_step, 
                    row['answer'], 
                    row['problem'], 
                    image_bytes, 
                    prompt_format_think_step_2,
                    first_response,
                    first_request=False
                )
                elapsed_time = time.time() - start_time
                logger.info(f"Received second response for row {row_id} in {elapsed_time:.2f} seconds")
                
                # Parse steps from second response
                second_steps = parse_steps(second_response)
                answer2 = parse_answer(second_response)
                # Validate we got at least 3 steps in second response
                if len(second_steps) >= 3:
                    result['step2'] = second_steps
                    result['answer2'] = answer2
                    result['status'] = 'success'
                    result['retries'] = retries
                    logger.info(f"Successfully processed row {row_id} with both requests")
                    result['1=2'] = True if result['answer1'] == result['answer2'] else False
                    if result['answer1'] != result['answer2']:
                        false_result = copy.deepcopy(result)
                        image_format = imghdr.what(None, h=image_bytes)
                        # Map common format names to standard MIME type suffixes
                        format_mapping = {
                            'jpeg': 'jpeg',
                            'jpg': 'jpeg',
                            'png': 'png',
                            'gif': 'gif',
                            'bmp': 'bmp',
                        }
                        normalized_format = format_mapping.get(image_format.lower(), 'png')
                        # Encode binary image data as Base64
                        base64_image = base64.b64encode(image_bytes).decode('utf-8')
                        false_result['image'] = base64_image
                        false_result['prompt1'] = prompt_format_think_step
                        false_result['prompt2'] = prompt_format_think_step_2
                        with open(output_false_file, 'a', encoding='utf-8') as f:
                            json.dump(false_result, f, ensure_ascii=False, indent=2)
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
    
    if result['status'] == 'failed':
        false_result = copy.deepcopy(result)
        image_format = imghdr.what(None, h=input_image)
        # Map common format names to standard MIME type suffixes
        format_mapping = {
            'jpeg': 'jpeg',
            'jpg': 'jpeg',
            'png': 'png',
            'gif': 'gif',
            'bmp': 'bmp',
        }
        normalized_format = format_mapping.get(image_format.lower(), 'png')
        # Encode binary image data as Base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        false_result['image'] = base64_image
        false_result['prompt1'] = prompt_format_think_step
        false_result['prompt2'] = prompt_format_think_step_2
        with open(output_false_file, 'a', encoding='utf-8') as f:
            json.dump(false_result, f, ensure_ascii=False, indent=2)
    else:
        logger.info(f"Successfully processed row {row_id}")

    return result

def request_to_qwen3_text_image(prompt_text, ground_truth, input_text, input_image, prompt_text2, answer_step_1 = None,first_request=True):
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
    
    client = OpenAI(
        api_key="sk-25678a0b18d24afa86d3185f736fd886",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    if first_request:
        # First request message
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
        # Second request message (includes first response)
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
                    {"type": "text", "text": prompt_text2 + "The correct answer to this question is:" + ground_truth}
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

def process_parquet_file(file_path: str) -> List[Dict[str, Any]]:
    """Process the entire Parquet file"""
    logger.info(f"Starting to process parquet file: {file_path}")
    
    # Read Parquet file
    df = pd.read_parquet(file_path)
    logger.info(f"Loaded parquet file with {len(df)} rows")
    
    results = []
    for _, row in df.iterrows():
        try:
            # Process each row
            result = process_row(row.to_dict())
            results.append(result)
        except Exception as e:
            row_id = row.get('id', 'unknown')
            logger.error(f"Error processing row {row_id}: {str(e)}", exc_info=True)
            continue
    
    success_count = len([r for r in results if r.get('status') == 'success'])
    logger.info(f"Processing complete. Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    
    return results

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

prompt_format_think_step="""You are a mathematician, statistician, and geometer. Below, I will present you with a math problem along with its accompanying diagram. Please carefully observe the details in the image.
Given the text, images, generate a step-by-step reasoning process that logically leads to the correct result in the <answer> step. Requirements:
Flexible step count (3-5 steps): Use only as many steps as needed—no forced extension. Label them clearly with <step1></step1>, <step2></step2>, etc.
Strict dependency on input: Base reasoning only on the provided text and images—do not reverse-engineer from the answer.
Image reference: Must incorporate details from the image, not just text.
Logical rigor: Ensure each step coherently supports the next, with no gaps or contradictions.
Describe in detail the information present in the diagram, including but not limited to: the positional relationships of points, the meaning of numerical values in the diagram, parallel lines...
Do not include the prompt information in the response. Consider what mathematical theorems are available.
Format (example with 4 steps):
<step1> [Focus on the observation of the image, describe in detail the information present in the diagram] </step1>
<step2> [Consider what mathematical theorems are available + Logical inference from step1 + text information and image details] </step2>
<step3> [Consider what mathematical theorems are available + Logical inference from step2 + text information and image details] </step3>
<step4> [Key conclusion leading to answer] </step4>
<answer> [Final answer] </answer>
Input:
"""
prompt_format_think_step_2=f"""You are a mathematician, statistician, and geometer. Please carefully observe the details in the image.
You have already provided the reasoning steps above. Do you think your answer is correct?
Please revise or improve your reasoning steps based on the correct answer. 
Emphasize! DO NOT include the correct answer in <step></step>. DO NOT any text like "because the answer is ." or "as per the correct answer".
After refining, regenerate a step-by-step reasoning process that logically leads to the correct result in the <answer> step. Format (example with 4 steps):
<step1> [Focus on the Observation of the image, describe in detail the information present in the diagram] </step1>
<step2> [Consider what mathematical theorems are available + Logical inference from step1 + text information and image details] </step2>
<step3> [Consider what mathematical theorems are available + Logical inference from step2 + text information and image details] </step3>
<step4> [Key conclusion leading to answer] </step4>
<answer> [Final answer] </answer>
"""

def process_parquet_file_with_autosave(file_path: str, output_file: str = "think_steps.json") -> List[Dict[str, Any]]:
    """Process the entire Parquet file and auto-save results after each row"""
    logger.info(f"Starting processing with autosave. Input: {file_path}, Output: {output_file}")
    
    # Load existing results if output file exists
    results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info(f"Loaded existing results from {output_file} with {len(results)} entries")
        except Exception as e:
            logger.error(f"Error reading existing output file: {str(e)}", exc_info=True)
    
    # Get set of already processed IDs
    processed_ids = {r['id'] for r in results if 'id' in r}
    logger.info(f"Found {len(processed_ids)} already processed rows")
    
    # Load and process parquet file
    df = pd.read_parquet(file_path)
    logger.info(f"Loaded parquet file with {len(df)} rows")
    
    total_rows = len(df)
    processed_count = len(processed_ids)
    remaining_rows = total_rows - processed_count
    logger.info(f"Processing {remaining_rows} new rows out of {total_rows} total")
    
    for _, row in df.iterrows():
        row_id = row.get('id', 'unknown')
        
        # Skip already processed rows
        if row_id in processed_ids:
            logger.debug(f"Skipping already processed row {row_id}")
            continue
            
        try:
            logger.info(f"Processing new row {row_id} ({processed_count + 1}/{total_rows})")
            
            # Process each row
            result = process_row(row.to_dict())
            results.append(result)
            processed_count += 1
            
            # Save after each successful processing
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.debug(f"Autosaved results after processing row {row_id}")
                
        except Exception as e:
            logger.error(f"Error processing row {row_id}: {str(e)}", exc_info=True)
            continue
    
    success_count = len([r for r in results if r.get('status') == 'success'])
    failure_count = len(results) - success_count
    logger.info(f"Processing complete. Results saved to {output_file}")
    logger.info(f"Successfully processed {success_count} rows")
    logger.info(f"Failed to process {failure_count} rows")
    if results:
        logger.info(f"Success rate: {success_count/len(results)*100:.1f}%")
    else:
        logger.info("No results were processed")
    
    return results

if __name__ == "__main__":
    input_file = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/data/MMK12_train.parquet"
    output_file = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation/mmk12_train_think_steps.json"
    
    logger.info("Starting main processing")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    
    # Process the file with auto-saving
    start_time = time.time()
    results = process_parquet_file_with_autosave(input_file, output_file)
    elapsed_time = time.time() - start_time
    
    logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
    logger.info("Processing completed successfully")