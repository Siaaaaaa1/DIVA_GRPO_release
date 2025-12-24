import pandas as pd
import os

def filter_correct_think(input_file: str):
    """
    读取 Parquet 文件，保留 correct_think 为 True 的行，
    并保存到一个新的文件，文件名为原文件名后加 '_filter'。
    
    :param input_file: 输入 Parquet 文件路径
    """
    # 读取 parquet 文件
    df = pd.read_parquet(input_file)
    
    # 保留 correct_think == True 的行
    df_filtered = df[df['correct_think'] == True]
    
    # 构造输出文件名
    dir_name = os.path.dirname(input_file)
    base_name = os.path.basename(input_file)
    name, ext = os.path.splitext(base_name)
    output_file = os.path.join(dir_name, f"{name}_filter{ext}")
    
    # 保存到新的 parquet 文件
    df_filtered.to_parquet(output_file, index=False)
    
    print(f"Filtered data saved to {output_file}. Total rows kept: {len(df_filtered)}")

filter_correct_think("/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/ablation_dataset/MMK12_train_gpto3.parquet")
