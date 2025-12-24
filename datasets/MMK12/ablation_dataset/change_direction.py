import pandas as pd

def reorder_think_file(input_file, think_file, output_file, random_state=None):
    """
    根据 input_file 中每个 id 的 difficulty，调整 think_file 的行顺序：
    - difficulty 为 3-7 的行打乱放前面
    - 其他行打乱放后面
    - 最后重置索引
    注意：不在 think_file 中添加 difficulty 列

    参数:
    input_file: Parquet 文件路径，包含 'id' 和 'difficulty'
    think_file: 需要重新排序的 Parquet 文件路径，包含 'id'
    output_file: 输出 Parquet 文件路径
    random_state: 随机种子
    """

    # 读取文件
    df_input = pd.read_parquet(input_file)
    df_think = pd.read_parquet(think_file)

    # 创建 id -> difficulty 映射
    id_to_diff = df_input.set_index('id')['difficulty'].to_dict()

    # 根据 id 获取 difficulty
    difficulties = df_think['id'].map(id_to_diff)

    if difficulties.isnull().any():
        print("警告: think_file 中有些 id 在 input_file 中找不到对应的 difficulty。")

    # 分组：3-7 和其他
    mask = difficulties.between(3, 7)
    df_high = df_think[mask].sample(frac=1, random_state=random_state)  # 打乱顺序
    df_low = df_think[~mask].sample(frac=1, random_state=random_state)  # 打乱顺序

    # 合并，high在前，low在后
    df_reordered = pd.concat([df_high, df_low], ignore_index=True)

    # 保存输出
    df_reordered.to_parquet(output_file, index=False)
    print(f"重新排序后的 think_file 已保存至: {output_file}")
    print(f"总行数: {len(df_reordered)}")
    print(f"前面difficulty 3-7行数: {len(df_high)}, 后面其他行数: {len(df_low)}")

    return df_reordered

def main():
    # input_file = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/checkpoints/easy_r1/Varient_Adapter_Weight_After_Newthink_L&Glow_reward_Zscorenorm_noKL_gpto3_15000-0827-00/MMK12_Adapter.parquet"      # 包含 id 和 difficulty
    # think_file = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/data/MMK12_train_gpto3.parquet"      # 需要重新排序的文件
    # output_file = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/ablation_dataset/MMK12_train_gpto3.parquet"  # 输出文件路径

    # input_file = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/checkpoints/easy_r1/Varient_Adapter_Weight_After_Newthink_L&Glow_reward_Zscorenorm_noKL_gpto3_15000-0827-00/MMK12_Adapter.parquet"      # 包含 id 和 difficulty
    # think_file = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/data/MMK12_train_thinkq_variantq.parquet"      # 需要重新排序的文件
    # output_file = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/ablation_dataset/MMK12_train_thinkq_variantq.parquet"  # 输出文件路径

    input_file = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/checkpoints/easy_r1/Varient_Adapter_Weight_After_Newthink_L&Glow_reward_Zscorenorm_noKL_gpto3_15000-0827-00/MMK12_Adapter.parquet"      # 包含 id 和 difficulty
    think_file = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/data/MMK12_train_thinkq_variantg.parquet"      # 需要重新排序的文件
    output_file = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/ablation_dataset/MMK12_train_thinkq_variantg.parquet"  # 输出文件路径
    random_state = 42                       # 随机种子，保证可复现

    # 调用重新排序函数
    df_reordered = reorder_think_file(
        input_file=input_file,
        think_file=think_file,
        output_file=output_file,
        random_state=random_state
    )

if __name__ == "__main__":
    main()
