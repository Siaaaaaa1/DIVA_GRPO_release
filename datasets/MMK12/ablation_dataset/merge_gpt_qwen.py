import pandas as pd

def merge_parquet_files(file1: str, file2: str, output_file: str):
    """
    按照 id 合并两个 Parquet 文件，补充缺失的 id 到主文件，并保存到新的文件中。
    
    :param file1: 主文件路径 (含有的 id 将作为参考)
    :param file2: 辅助文件路径 (补充缺失的 id)
    :param output_file: 输出文件路径
    """
    # 读取两个 Parquet 文件
    df1 = pd.read_parquet(file1)
    df2 = pd.read_parquet(file2)
    
    # 使用 'id' 列作为合并键，合并数据
    merged_df = pd.merge(df1, df2, on='id', how='right')
    
    # 将结果保存到新的 Parquet 文件
    merged_df.to_parquet(output_file, index=False)
    
    print(f"Merged data saved to {output_file}. Total rows: {len(merged_df)}")

def main():
    # 文件路径
    file1 = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/ablation_dataset/MMK12_train_gpto3_filter.parquet"
    file2 = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/ablation_dataset/MMK12_train_thinkq_variantq.parquet"
    output_file = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/ablation_dataset/MMK12_train_qwen_gpt.parquet"
    
    # 调用合并函数
    merge_parquet_files(file1, file2, output_file)

if __name__ == "__main__":
    main()
