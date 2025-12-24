# import os
# import pyarrow as pa
# import pyarrow.parquet as pq
# import random

# def filter_parquet_by_ids(input_file, output_file, keep_ids):
#     """按 id 筛选 Parquet 文件，保留所有列"""
#     pq_writer = None
#     pf = pq.ParquetFile(input_file)
    
#     for batch in pf.iter_batches(batch_size=1024):
#         batch_table = pa.Table.from_batches([batch])
#         mask = batch_table.column("id").to_pandas().isin(keep_ids)
#         filtered_table = batch_table.filter(pa.array(mask))
#         if filtered_table.num_rows == 0:
#             continue
#         if pq_writer is None:
#             pq_writer = pq.ParquetWriter(output_file, filtered_table.schema)
#         pq_writer.write_table(filtered_table)
    
#     if pq_writer is not None:
#         pq_writer.close()

# def process_parquet(file_a, file_b, output_dir):
#     # 1️⃣ 读取简单列
#     simple_cols = ["id", "subject", "correct_think"]
#     table_a_simple = pq.read_table(file_a, columns=simple_cols)
#     table_b_simple = pq.read_table(file_b, columns=simple_cols)

#     # 2️⃣ 文件1: correct_think = True
#     correct_mask = table_a_simple.column("correct_think").to_pandas() == True
#     correct_ids = set(table_a_simple.column("id").to_pandas()[correct_mask])
#     file1_path = os.path.join(output_dir, os.path.basename(file_a).replace(".parquet", "_correct.parquet"))
#     filter_parquet_by_ids(file_a, file1_path, correct_ids)
#     print(f"文件1输出完成: {file1_path}, 条数={len(correct_ids)}")

#     # 3️⃣ 文件2: mid_grpo = correct_think=True 且 B文件抽5000 + others抽5000
#     random.seed(42)

#     # B 文件中 correct_think=True 的 id
#     b_mask = table_b_simple.column("correct_think").to_pandas() == True
#     b_ids_correct = set(table_b_simple.column("id").to_pandas()[b_mask]).intersection(correct_ids)
#     if len(b_ids_correct) < 5000:
#         raise ValueError(f"B 文件可用 correct_id 不足 5000 条, 只有 {len(b_ids_correct)} 条")
#     b_sample = set(random.sample(list(b_ids_correct), 5000))

#     # subject='others' 且 correct_think=True 的 id
#     others_mask = (table_a_simple.column("subject").to_pandas() == "others") & \
#                   (table_a_simple.column("correct_think").to_pandas() == True)
#     others_ids_all = set(table_a_simple.column("id").to_pandas()[others_mask])
#     others_available = others_ids_all - b_sample
#     if len(others_available) < 5000:
#         raise ValueError(f"others 可用 correct_id 不足 5000 条, 只有 {len(others_available)} 条")
#     others_sample = set(random.sample(list(others_available), 5000))

#     # 合并成 mid_grpo
#     mid_ids = b_sample | others_sample
#     file2_path = os.path.join(output_dir, os.path.basename(file_a).replace(".parquet", "_mid_grpo.parquet"))
#     filter_parquet_by_ids(file_a, file2_path, mid_ids)
#     print(f"文件2输出完成: {file2_path}, 条数={len(mid_ids)}")

#     # 4️⃣ 文件3: sft = correct_think=True 且不在 mid_grpo 中
#     sft_ids = correct_ids - mid_ids
#     file3_path = os.path.join(output_dir, os.path.basename(file_a).replace(".parquet", "_sft.parquet"))
#     filter_parquet_by_ids(file_a, file3_path, sft_ids)
#     print(f"文件3输出完成: {file3_path}, 条数={len(sft_ids)}")

# # 使用示例
# process_parquet(
#     "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/R1-Share-VL/R1-ShareVL-52k_merge_fixed.parquet",
#     "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/ablation_dataset/MMK12_train_gpto3_filter_5000.parquet",
#     "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/R1-Share-VL"
# )


import os
import pyarrow as pa
import pyarrow.parquet as pq
import random

def filter_parquet_by_ids(input_file, output_file, keep_ids):
    """按 id 筛选 Parquet 文件，保留所有列"""
    pq_writer = None
    pf = pq.ParquetFile(input_file)
    
    for batch in pf.iter_batches(batch_size=1024):
        batch_table = pa.Table.from_batches([batch])
        mask = batch_table.column("id").to_pandas().isin(keep_ids)
        filtered_table = batch_table.filter(pa.array(mask))
        if filtered_table.num_rows == 0:
            continue
        if pq_writer is None:
            pq_writer = pq.ParquetWriter(output_file, filtered_table.schema)
        pq_writer.write_table(filtered_table)
    
    if pq_writer is not None:
        pq_writer.close()

def process_parquet(file_a, file_b, output_dir):
    # 1️⃣ 读取简单列
    simple_cols = ["id", "subject", "correct_think"]
    table_a_simple = pq.read_table(file_a, columns=simple_cols)

    # 2️⃣ 获取 correct_think=True 的所有 id
    correct_mask = table_a_simple.column("correct_think").to_pandas() == True
    correct_ids = list(table_a_simple.column("id").to_pandas()[correct_mask])

    if len(correct_ids) < 5000:
        raise ValueError(f"correct_think=True 的记录不足 5000 条, 只有 {len(correct_ids)} 条")

    # 随机拆分 5000 条和剩余条
    random.seed(42)
    sample_5000 = set(random.sample(correct_ids, 5000))
    remaining_ids = set(correct_ids) - sample_5000

    # 输出 5000 条文件
    file1_5000_path = os.path.join(output_dir, os.path.basename(file_a).replace(".parquet", "_correct_5000.parquet"))
    filter_parquet_by_ids(file_a, file1_5000_path, sample_5000)
    print(f"文件1-5000输出完成: {file1_5000_path}, 条数={len(sample_5000)}")

    # 输出剩余文件
    file1_rest_path = os.path.join(output_dir, os.path.basename(file_a).replace(".parquet", "_correct_rest.parquet"))
    filter_parquet_by_ids(file_a, file1_rest_path, remaining_ids)
    print(f"文件1-剩余输出完成: {file1_rest_path}, 条数={len(remaining_ids)}")

# 使用示例
process_parquet(
    "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/R1-Share-VL/R1-ShareVL-52k_merge_fixed.parquet",
    None,  # 这里不需要 file_b 了
    "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/R1-Share-VL"
)