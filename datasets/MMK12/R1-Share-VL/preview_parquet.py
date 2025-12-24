import pyarrow.parquet as pq
import random

def preview_parquet(file_path, n=10):
    # 读取 Parquet 文件为 pyarrow.Table
    table = pq.read_table(file_path)
    columns = table.column_names
    num_rows = table.num_rows

    print(f"File: {file_path}")
    print(f"Total rows: {num_rows}, Total columns: {len(columns)}")

    # 检查是否所有行的 correct_think 都为 True
    if "correct_think" in columns:
        correct_think_col = table.column("correct_think").to_pylist()
        all_true = all(correct_think_col)
        print(f"\nAll rows correct_think == True? {all_true}")
    else:
        print("\nColumn 'correct_think' not found in the parquet file.")

    # 随机选择 n 行索引
    sample_indices = random.sample(range(num_rows), min(n, num_rows))

    # 遍历每一列，展示采样内容
    for col in columns:
        print(f"\nPreviewing column: {col}")
        col_values = table.column(col).to_pylist()
        for idx in sample_indices:
            value = col_values[idx]
            if isinstance(value, (dict, list)):
                print(f"Row {idx}: {str(value)[:20]}...")  # 截断前20个字符
            else:
                print(f"Row {idx}: {value}")

# 示例文件路径
file_path = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/R1-Share-VL/GRPO_45784.parquet"

# 执行函数，预览随机的 10 行
preview_parquet(file_path, n=10)
