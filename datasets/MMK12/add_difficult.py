import pandas as pd
import numpy as np
import json


def read_parquet(file_path: str) -> pd.DataFrame:
    """读取Parquet文件并返回DataFrame"""
    return pd.read_parquet(file_path)


def save_parquet(df: pd.DataFrame,
                 file_path: str,
                 compression: str = 'snappy') -> None:
    """将DataFrame保存为Parquet文件"""
    df.to_parquet(file_path, compression=compression)


def add_constant_column(df: pd.DataFrame,
                        column_name: str,
                        value: any,
                        inplace: bool = False):
    """向DataFrame中添加一个新列"""
    if inplace:
        df[column_name] = value
        return None
    else:
        df_copy = df.copy()
        df_copy[column_name] = value
        return df_copy


def load_json_as_dict(file_path: str) -> dict:
    """读取JSON文件并返回字典"""
    with open(file_path, "r") as f:
        return json.load(f)


def merge_data_with_think_steps(df: pd.DataFrame,
                                think_steps_json: str,
                                difficulty: float = 5.0,
                                category: str = 'origin_problem') -> pd.DataFrame:
    """将思考步骤合并到主DataFrame中，如果原有值为空则填充新值"""
    js_think_step = load_json_as_dict(think_steps_json)

    # 添加常数列
    df = add_constant_column(df, 'difficulty', difficulty)
    df = add_constant_column(df, 'category', category)

    # 合并思考步骤数据
    for k, v in js_think_step.items():
        target_row = df.loc[df['id'] == k]
        if target_row.empty:
            print(f"[警告] 未找到 id: {k}")
            continue
        row_index = target_row.index[0]
        for col in ['step1', 'step2', 'answer1', 'answer2', 'correct_think']:
            if col in v:
                df.at[row_index, col] = v[col]
    return df


if __name__ == "__main__":
    parquet_path = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/R1-Share-VL/R1-ShareVL-52k_merge_conflict.parquet"
    think_steps_json_path = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation/sharevl_train_gpto3/mmk12_train_think_steps_merge_conflict_id_key.json"
    output_path = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/R1-Share-VL/R1-ShareVL-52k_merge_conflict_2.parquet"

    # 读取原始数据
    df = read_parquet(parquet_path)

    # 合并数据
    df = merge_data_with_think_steps(df, think_steps_json_path)

    # 保留 correct_think == True 的行
    df = df[df['correct_think'] == True]

    # 检查含有空值的行
    rows_with_nulls = df[df.isna().any(axis=1)]
    if not rows_with_nulls.empty:
        print("⚠️ 以下 id 含有 None/NaN，将被删除:")
        print(rows_with_nulls['id'].tolist())

    # 删除包含 None/NaN 的行
    df = df.dropna(how="any")

    # 检查并输出结果
    print("更新后的DataFrame:")
    print(df.head(5))

    print("剩余行数:", len(df))

    # 保存结果
    save_parquet(df, output_path)
    print(f"结果已保存到 {output_path}")
