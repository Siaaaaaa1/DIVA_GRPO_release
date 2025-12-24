import pandas as pd

# 输入文件路径
file1_path = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/ablation_dataset/MMK12_train_gpto3_filter.parquet"
file2_path = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/ablation_dataset/MMK12_train_gpto3_filter_5000.parquet"
output_path = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/ablation_dataset/MMK12_train_gpto3_filter_10000.parquet"

# 读取 parquet 文件
df1 = pd.read_parquet(file1_path)
df2 = pd.read_parquet(file2_path)

# 确保都有'id'列
if "id" not in df1.columns or "id" not in df2.columns:
    raise ValueError("两个文件都必须包含 'id' 列")

# 用isin去除 file2 中的 id
filtered_df = df1[~df1["id"].isin(df2["id"])]

# 保存结果
filtered_df.to_parquet(output_path, index=False)

print(f"已保存结果到 {output_path}，剩余行数: {len(filtered_df)}")
