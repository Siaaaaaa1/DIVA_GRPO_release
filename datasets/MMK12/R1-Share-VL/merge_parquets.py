import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np

def merge_parquet(files, output_file, difficulty_value=5, chunksize=10000):
    writer = None
    total_rows = 0

    columns_order = [
        "id", "problem", "answer", "subject", "images", "difficulty",
        "category", "step1", "step2", "answer1", "answer2", "correct_think", "variant"
    ]

    for f in files:
        table = pq.read_table(f)

        # 删除 pandas 索引列
        if "__index_level_0__" in table.column_names:
            table = table.drop(["__index_level_0__"])

        # 如果存在 difficulty 列，替换为 int64 常数列
        if "difficulty" in table.column_names:
            num_rows = table.num_rows
            difficulty_array = pa.array(
                np.full(num_rows, difficulty_value, dtype=np.int64)
            )
            table = table.set_column(
                table.schema.get_field_index("difficulty"),
                "difficulty",
                difficulty_array
            )

        # 只保留 correct_think == True 的行
        if "correct_think" in table.column_names:
            mask = table.column("correct_think")
            table = table.filter(mask)

        # 重排列顺序
        table = table.select([c for c in columns_order if c in table.column_names])

        # 分批写入
        batches = table.to_batches(max_chunksize=chunksize)
        for batch in batches:
            if writer is None:
                writer = pq.ParquetWriter(output_file, batch.schema)
            writer.write_batch(batch)
            total_rows += batch.num_rows

    if writer is not None:
        writer.close()

    print(f"✅ Merged {len(files)} parquet files, total {total_rows} rows, saved to {output_file}")


files = [
    "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/R1-Share-VL/R1-ShareVL-52k_merge_conflict_2.parquet",
    "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/R1-Share-VL/R1-ShareVL-52k_merge_fixed_correct_5000.parquet",
    "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/R1-Share-VL/R1-ShareVL-52k_merge_fixed_correct_rest.parquet",
    "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/data/MMK12_train_gpto3.parquet"
]

output_file = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/R1-Share-VL/GRPO_45784.parquet"

# 执行
merge_parquet(files, output_file, chunksize=5000)