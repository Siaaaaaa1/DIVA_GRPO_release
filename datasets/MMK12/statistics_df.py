import pandas as pd
import numpy as np

# 写死的 Parquet 文件路径
# file_path = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/ablation_dataset/MMK12_train_gpto3_filter_5000.parquet"
file_path = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/R1-Share-VL/R1-ShareVL-52k_merge.parquet"
def preview_parquet(file_path):
    # 读取 parquet 文件
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 显示前5行
    print("=== 前5行数据预览 ===")
    print(df.head(5))

    # 统计各列信息
    info = {}
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].notnull().sum()
        null_count = df[col].isnull().sum()
        # 处理不可哈希类型
        try:
            unique_count = df[col].nunique()
        except TypeError:
            unique_count = "unhashable"
        info[col] = {
            "dtype": dtype,
            "non_null_count": non_null,
            "null_count": null_count,
            "unique_count": unique_count
        }
    info_df = pd.DataFrame(info).T
    print("\n=== 各列统计信息 ===")
    print(info_df)

    # 随机抽取5行 variant 列（如果存在）
    if "correct_think" in df.columns:
        print("\n=== 随机抽取5行 images 列值 ===")
        print(df["correct_think"].dropna().sample(n=min(20, df["correct_think"].dropna().shape[0])).values)
    else:
        print("\ncorrect_think 列不存在。")
    # if "variant" in df.columns:
    #     print("\n=== 随机抽取5行 variant 列值 ===")
    #     print(df["variant"].dropna().sample(n=min(5, df["variant"].dropna().shape[0])).values)
    # else:
    #     print("\nvariant 列不存在。")

if __name__ == "__main__":
    preview_parquet(file_path)
