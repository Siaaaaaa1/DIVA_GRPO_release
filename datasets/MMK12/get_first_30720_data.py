import pandas as pd
import pyarrow.parquet as pq

def filter_and_save_parquet(input_path, output_path=None):
    """
    读取Parquet文件，筛选step2不为None的数据，限制5000行，保存为新文件
    
    参数:
        input_path: 输入Parquet文件路径
        output_path: 输出文件路径(可选)，如不指定则自动添加"_5000"
    
    返回:
        新文件的保存路径
    """
    # 读取Parquet文件
    df = pd.read_parquet(input_path)
    
    # 筛选step2不为None的行
    filtered_df = df[df['step2'].notna()]
    # 限制行数为5000
    limited_df = filtered_df.head(30720)
    # print(limited_df['step2'])
    # 处理输出文件名
    if output_path is None:
        if input_path.endswith('.parquet'):
            output_path = input_path.replace('.parquet', '_5000.parquet')
        else:
            output_path = input_path + '_'
    print("保存的列：", limited_df.columns.tolist())

    # 保存新文件
    limited_df.to_parquet(output_path, index=False)
    
    print(f"文件已保存至: {output_path}")
    return output_path

# 使用示例
filter_and_save_parquet('/mmu_cd_ssd/zhangzhenyu06/workspace/Rebuttal/EasyR1/datasets/MMK12/R1-Share-VL/R1-ShareVL-52k_merge_fixed_correct.parquet','/mmu_cd_ssd/zhangzhenyu06/workspace/Rebuttal/EasyR1/datasets/RE/RE_30720.parquet')