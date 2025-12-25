import os
import json
import argparse
import pandas as pd
import multiprocessing
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, Any

# 导入上面定义的模块
from llm_client import AzureQwenClient, BaseLLMClient

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] # 可以添加 FileHandler
)
logger = logging.getLogger(__name__)

def process_chunk(chunk_df: pd.DataFrame, output_json_path: str, api_config: Dict[str, str]):
    """处理单个数据块并保存为 JSON (作为中间检查点)"""
    
    # 在子进程中初始化客户端
    client = AzureQwenClient(
        api_key=api_config['key'],
        endpoint=api_config['endpoint'],
        deployment=api_config.get('deployment', 'qwen-vl-max-latest')
    )
    
    results = []
    
    for idx, row in tqdm(chunk_df.iterrows(), total=len(chunk_df), desc=f"Processing {os.path.basename(output_json_path)}"):
        try:
            row_id = row.get('id', idx)
            problem = row['problem']
            answer = row['answer']
            
            # 获取图片数据 (假设 parquet 中有一列 'images'，其内部结构为 {'bytes': b'...'} 或直接是 bytes)
            image_data = None
            if 'images' in row and row['images']:
                img_entry = row['images']
                if isinstance(img_entry, dict) and 'bytes' in img_entry:
                    image_data = img_entry['bytes']
                elif isinstance(img_entry, bytes):
                    image_data = img_entry
            
            # ============ 1. 生成 Variants ============
            variants = client.generate_variants(problem)
            
            # ============ 2. 生成 Think Steps ============
            think_result = client.generate_think_steps(problem, answer, image_data)
            
            if think_result['status'] == 'success':
                # 构建结果字典
                processed_item = row.to_dict()
                # 如果 images 列包含 bytes，为了 JSON 序列化可能需要处理，或者在最终合并时再从原始 df 读取
                # 这里为了中间存储简单，我们先不存 images 的 bytes 到 json，只存生成的文本
                if 'images' in processed_item:
                    del processed_item['images'] 
                
                processed_item['variants'] = variants
                processed_item['think_steps'] = think_result['refined_steps'] # 使用 refine 后的步骤
                processed_item['think_answer'] = think_result['refined_answer']
                
                results.append(processed_item)
            else:
                logger.warning(f"Row {row_id} failed logic check: {think_result.get('reason')}")
        
        except Exception as e:
            logger.error(f"Error processing row {row.get('id', 'unknown')}: {e}")
            continue

        # 每处理 10 条保存一次，防止丢失
        if len(results) % 10 == 0:
             with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    # 最终保存
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Augment dataset with Variants and Think Steps using LLM.")
    parser.add_argument("--input", type=str, required=True, help="Path to input Parquet file")
    parser.add_argument("--output", type=str, required=True, help="Path to output Parquet file")
    parser.add_argument("--workers", type=int, default=4, help="Number of multiprocessing workers")
    parser.add_argument("--api_key", type=str, default=os.getenv("AZURE_OPENAI_KEY"), help="Azure API Key")
    parser.add_argument("--endpoint", type=str, default=os.getenv("AZURE_OPENAI_ENDPOINT"), help="Azure Endpoint")
    
    args = parser.parse_args()
    
    if not args.api_key or not args.endpoint:
        raise ValueError("API Key and Endpoint must be provided via arguments or environment variables.")

    api_config = {
        "key": args.api_key,
        "endpoint": args.endpoint,
        "deployment": "qwen-vl-max-latest" # 可以参数化
    }

    # 读取 Parquet
    logger.info(f"Loading dataset from {args.input}...")
    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df)} rows.")

    # 创建临时输出目录
    temp_dir = Path(args.output).parent / "temp_processing"
    temp_dir.mkdir(exist_ok=True)

    # 切分数据块
    chunk_size = (len(df) + args.workers - 1) // args.workers
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    processes = []
    temp_files = []

    logger.info(f"Starting {len(chunks)} workers...")
    
    for i, chunk in enumerate(chunks):
        temp_file = temp_dir / f"worker_{i}.json"
        temp_files.append(str(temp_file))
        
        p = multiprocessing.Process(
            target=process_chunk,
            args=(chunk, str(temp_file), api_config)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    logger.info("All workers finished. Merging results...")

    # 合并结果
    all_data = []
    original_df_map = {row['id']: row for _, row in df.iterrows()} # 用于找回原始图片数据

    for tf in temp_files:
        if os.path.exists(tf):
            with open(tf, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    all_data.extend(data)
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode JSON from {tf}")
    
    if not all_data:
        logger.error("No data generated!")
        return

    # 转换为 DataFrame
    result_df = pd.DataFrame(all_data)
    
    # 这一步非常重要：如果我们在 JSON 中丢弃了 'images' 列（因为 bytes 无法序列化），
    # 我们需要从原始 DataFrame 中把 'images' 列根据 ID 合并回来。
    # 假设 'id' 是唯一键
    if 'images' not in result_df.columns and 'id' in result_df.columns:
        logger.info("Restoring image data from original dataframe...")
        # 简单的方法：Merge
        # 注意：这里假设处理后的数据是原始数据的子集（如果有些失败了）
        original_images = df[['id', 'images']]
        result_df = pd.merge(result_df, original_images, on='id', how='left')

    # 保存为 Parquet
    logger.info(f"Saving {len(result_df)} rows to {args.output}...")
    result_df.to_parquet(args.output)
    logger.info("Done.")

if __name__ == "__main__":
    main()