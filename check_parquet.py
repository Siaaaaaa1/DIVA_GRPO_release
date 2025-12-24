import pandas as pd

# 读取 parquet 文件
def preview_parquet(file_path):
    # 使用 pandas 读取 Parquet 文件
    df = pd.read_parquet(file_path, engine='pyarrow')
    
    # 展示 DataFrame 的前几行（默认是前5行）
    print("Preview of the first few rows:")
    print(df.head())

    # 获取并展示第一行的完整内容，确保字符串完整展示
    first_row = df.iloc[0]
    
    print("\nFirst row content with full string length:")
    for column, value in first_row.items():
        if isinstance(value, str):
            print(f"{column}: {value}")
        else:
            print(f"{column}: {value}")

# 示例路径替换成实际文件路径
file_path = '/mmu_cd_ssd/zhangzhenyu06/workspace/Rebuttal/EasyR1/datasets/OCR/ocr_GRPO_18000.parquet'
preview_parquet(file_path)
