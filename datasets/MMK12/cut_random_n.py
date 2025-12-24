import pandas as pd
import os

def sample_and_save_parquet(input_path, n, output_path=None, random_state=42):
    """
    读取Parquet文件，随机取样n条数据，保存为新文件
    
    参数:
        input_path: 输入Parquet文件路径
        n: 随机取样的行数（正整数）
        output_path: 输出文件路径(可选)，如不指定则自动添加"_random{n}"
        random_state: 随机种子，保证结果可重现
    
    返回:
        新文件的保存路径
    
    异常:
        ValueError: 当n不是正整数或文件为空时
        FileNotFoundError: 当输入文件不存在时
    """
    # 参数验证
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n必须是正整数")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    # 读取Parquet文件
    df = pd.read_parquet(input_path)
    total_rows = len(df)
    
    if total_rows == 0:
        raise ValueError("输入文件为空")
    
    # 随机取样n条数据
    if n > total_rows:
        print(f"警告：请求的行数({n})大于数据总行数({total_rows})，将返回所有行")
        sampled_df = df
    else:
        sampled_df = df.sample(n=n, random_state=random_state)
    
    # 处理输出文件名
    if output_path is None:
        if input_path.endswith('.parquet'):
            output_path = input_path.replace('.parquet', f'_random{n}.parquet')
        else:
            output_path = f"{input_path}_random{n}.parquet"
    
    print(f"原始数据行数: {total_rows}")
    print(f"取样后行数: {len(sampled_df)}")
    print("保存的列：", sampled_df.columns.tolist())
    
    # 保存新文件
    sampled_df.to_parquet(output_path, index=False)
    
    print(f"文件已保存至: {output_path}")
    return output_path

# 使用示例
# 随机取样1000条，自动命名为: R1-ShareVL-52k_merge_fixed_correct_random1000.parquet
sample_and_save_parquet(
    '/mmu_cd_ssd/zhangzhenyu06/workspace/Rebuttal/EasyR1/datasets/MMK12/R1-Share-VL/GRPO_45117.parquet',
    n=5000
)

# # 或者指定输出路径
# sample_and_save_parquet(
#     '/mmu_cd_ssd/zhangzhenyu06/workspace/Rebuttal/EasyR1/datasets/MMK12/R1-Share-VL/R1-ShareVL-52k_merge_fixed_correct.parquet',
#     n=1000,
#     output_path='/mmu_cd_ssd/zhangzhenyu06/workspace/Rebuttal/EasyR1/datasets/RE/RE_random1000.parquet'
# )