import pandas as pd
import numpy as np
import pyarrow.parquet as pq

def process_parquet_samples(input_file, output_filtered, random_state=None):
    """
    从Parquet文件中创建两个独立的5000行样本
    
    参数:
    input_file: 输入Parquet文件路径
    output_random: 随机抽样数据保存路径
    output_filtered: 筛选后抽样数据保存路径
    sample_size: 抽样大小，默认为5000
    random_state: 随机种子，用于可重复性
    """
    
    # 读取整个Parquet文件
    # df = pd.read_parquet(input_file)
    df2 = pd.read_parquet(input_file)
    # 第一个样本：随机抽取5000行
    # random_sample = df.sample(n=sample_size, random_state=random_state)
    
    # 保存随机抽样数据
    # random_sample.to_parquet(output_random, index=False)
    # print(f"随机抽样数据已保存至: {output_random}")
    # print(f"随机抽样数据行数: {len(random_sample)}")
    # print(f"随机抽样中difficulty分布:\n{random_sample['difficulty'].value_counts().sort_index()}")
    
    # 第二个样本：筛选difficulty为3、4、5、6、7的行
    filtered_df = df2[df2['difficulty'].isin([3, 4, 5, 6, 7])]
    filtered_df['difficulty'] = 5
    # if len(filtered_df) < sample_size:
    #     raise ValueError(f"筛选后的行数不足：需要 {sample_size} 行，但只有 {len(filtered_df)} 行可用")
    
    # 从筛选后的数据中随机抽取5000行
    # filtered_sample = filtered_df.sample(n=sample_size, random_state=random_state)
    
    # 保存筛选后抽样数据
    filtered_df.to_parquet(output_filtered, index=False)
    
    # 返回处理信息
    print(f"\n筛选后抽样数据已保存至: {output_filtered}")
    print(f"筛选后抽样数据行数: {len(filtered_df)}")
    print(f"筛选后抽样中difficulty分布:\n{filtered_df['difficulty'].value_counts().sort_index()}")
    print(f"原始数据中difficulty为2-8的行数: {len(filtered_df)}")
    
    return filtered_df

# 使用示例
if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    
    process_parquet_samples(
        input_file="/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/checkpoints/easy_r1/Varient_Adapter_Weight_After_Newthink_L&Glow_reward_Zscorenorm_qq_noKL_morethink_15000-0828-17/MMK12_Adapter.parquet",
        # output_random="/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/ablation_dataset/original_sample.parquet",
        output_filtered="/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/ablation_dataset/MMK12_2-8.parquet",
        # sample_size=5000,
        random_state=42
    )