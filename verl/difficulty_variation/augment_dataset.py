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
    handlers=[logging.StreamHandler()] 
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
    
    # 尝试加载已有的进度（如果中断过）
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info(f"Resuming {os.path.basename(output_json_path)} with {len(results)} items processed.")
        except:
            results = []

    # 找出已经处理过的 ID
    processed_ids = set(item.get('id') for item in results)

    for idx, row in tqdm(chunk_df.iterrows(), total=len(chunk_df), desc=f"Processing {os.path.basename(output_json_path)}"):
        row_id = row.get('id', idx)
        
        if row_id in processed_ids:
            continue

        try:
            problem = row['problem']
            answer = row['answer']
            
            # 获取图片数据
            image_data = None
            if 'images' in row and row['images']:
                img_entry = row['images']
                if isinstance(img_entry, dict) and 'bytes' in img_entry:
                    image_data = img_entry['bytes']
                elif isinstance(img_entry, bytes):
                    image_data = img_entry
            
            # ============ 1. 生成 Variants ============
            variants = client.generate_variants(problem)
            
            # ============ 2. 生成 Think Steps (包含对比检查) ============
            # 注意：这里直接传入 answer 用于内部校验
            think_result = client.generate_think_steps(problem, answer, image_data)
            
            if think_result['status'] == 'success':
                # 构建结果字典
                processed_item = row.to_dict()
                
                # 不存 bytes 到 json
                if 'images' in processed_item:
                    del processed_item['images'] 
                
                processed_item['variants'] = variants
                processed_item['think_steps'] = think_result['think_steps'] # 使用初次生成的 steps
                processed_item['think_answer'] = think_result['think_answer']
                
                results.append(processed_item)
            else:
                # 答案不一致或生成失败，丢弃该数据
                logger.warning(f"Row {row_id} skipped: {think_result.get('reason')}")
        
        except Exception as e:
            logger.error(f"Error processing row {row.get('id', 'unknown')}: {e}")
            continue

        # 每处理 5 条保存一次，防止丢失
        if len(results) % 5 == 0:
             with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    # 最终保存
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Augment dataset with Variants and Think Steps using LLM.")
    parser.add_argument("--input", type=str, required=True, help="Path to input Parquet file")
    parser.add_argument("--output", type=str, required=True, help="Path to output Parquet file (will be used as prefix for chunks)")
    parser.add_argument("--workers", type=int, default=4, help="Number of multiprocessing workers")
    parser.add_argument("--api_key", type=str, default=os.getenv("AZURE_OPENAI_KEY"), help="Azure API Key")
    parser.add_argument("--endpoint", type=str, default=os.getenv("AZURE_OPENAI_ENDPOINT"), help="Azure Endpoint")
    
    args = parser.parse_args()
    
    if not args.api_key or not args.endpoint:
        raise ValueError("API Key and Endpoint must be provided via arguments or environment variables.")

    api_config = {
        "key": args.api_key,
        "endpoint": args.endpoint,
        "deployment": "qwen-vl-max-latest" 
    }

    # 读取 Parquet
    logger.info(f"Loading dataset from {args.input}...")
    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df)} rows.")

    # 创建临时输出目录
    temp_dir = Path(args.output).parent / "temp_processing"
    temp_dir.mkdir(exist_ok=True, parents=True)

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

    logger.info("All workers finished. Processing chunks to Parquet...")

    # ============ 分块保存逻辑 ============
    # 不再将所有数据 load 进内存合并，而是逐个处理 chunk 并保存为独立的 parquet 文件
    
    # 获取原始图片的映射 (如果内存够放索引的话；如果不够，这里也需要优化，但通常 id->image 映射还好)
    # 为了避免 OOM，我们假设这里不一次性 copy 整个 df，而是按需提取
    # 如果数据集极大，建议直接在 process_chunk 里处理完图片（不转json），但多进程写 parquet 比较麻烦。
    # 这里我们采用：读取 json -> 匹配 df 中的图片 -> 保存为 output_part_xx.parquet
    
    base_output_name = str(Path(args.output).stem)
    output_dir = Path(args.output).parent
    
    original_images = df[['id', 'images']] if 'images' in df.columns else pd.DataFrame()

    total_saved = 0
    for i, tf in enumerate(temp_files):
        if not os.path.exists(tf):
            continue
            
        try:
            with open(tf, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not data:
                continue
                
            chunk_result_df = pd.DataFrame(data)
            
            # 恢复图片
            if 'images' not in chunk_result_df.columns and not original_images.empty and 'id' in chunk_result_df.columns:
                # 仅在当前 chunk 的数据中做 merge
                chunk_result_df = pd.merge(chunk_result_df, original_images, on='id', how='left')
            
            # 构造分块文件名
            chunk_output_path = output_dir / f"{base_output_name}_part_{i}.parquet"
            chunk_result_df.to_parquet(chunk_output_path)
            
            logger.info(f"Saved chunk {i} with {len(chunk_result_df)} rows to {chunk_output_path}")
            total_saved += len(chunk_result_df)
            
        except Exception as e:
            logger.error(f"Failed to process temp file {tf}: {e}")

    logger.info(f"Done. Total rows saved: {total_saved}")

if __name__ == "__main__":
    main()