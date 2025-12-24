import pandas as pd
import json

# 文件路径
json_file = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation/sharevl_train_gpto3/sharevl_think_steps_id_key.json"
parquet_file = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/R1-Share-VL/R1-ShareVL-52k_merge.parquet"
output_parquet = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/R1-Share-VL/R1-ShareVL-52k_merge_fixed.parquet"
conflict_parquet = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/R1-Share-VL/R1-ShareVL-52k_merge_conflict.parquet"
output_csv = output_parquet.replace(".parquet", ".csv")
conflict_csv = conflict_parquet.replace(".parquet", ".csv")

# 读取 JSON
with open(json_file, "r", encoding="utf-8") as f:
    json_data = json.load(f)

# 构建 (problem, answer) -> list of ids 映射
problem_answer_to_ids = {}
for id_key, content in json_data.items():
    problem = content["problem"].strip()
    answer = content.get("original_answer", "").strip()
    key = (problem, answer)
    if key not in problem_answer_to_ids:
        problem_answer_to_ids[key] = []
    problem_answer_to_ids[key].append(id_key)

# 读取 Parquet
df = pd.read_parquet(parquet_file)

# 去掉 problem 和 answer 字符串两端空格
df["problem"] = df["problem"].str.strip()
df["answer"] = df["answer"].str.strip()

# 准备要新增的列
for col in ["step1", "step2", "answer1", "answer2", "correct_think"]:
    df[col] = None

conflict_count = 0
unmatched_count = 0

rows_to_keep = []
conflict_rows = []

for idx, row in df.iterrows():
    if row.get("subject", "") != "others":
        # 非 others，保留原行
        rows_to_keep.append(idx)
        continue

    key = (row["problem"], row["answer"])
    if key not in problem_answer_to_ids:
        # 未匹配，删除该行
        unmatched_count += 1
        continue
    elif len(problem_answer_to_ids[key]) > 1:
        # 冲突，保存到 conflict_rows
        conflict_count += 1
        conflict_rows.append(idx)
        continue
    else:
        # 唯一匹配
        matched_id = problem_answer_to_ids[key][0]
        row["id"] = matched_id

        # 从 JSON 更新额外列
        json_entry = json_data[matched_id]
        row["step1"] = json_entry.get("step1")
        row["step2"] = json_entry.get("step2")
        row["answer1"] = json_entry.get("answer1")
        row["answer2"] = json_entry.get("answer2")
        row["correct_think"] = json_entry.get("correct_think")

        df.loc[idx] = row
        rows_to_keep.append(idx)

# 只保留匹配到或非 others 的行
df_fixed = df.loc[rows_to_keep].reset_index(drop=True)

# 冲突行单独保存
df_conflict = df.loc[conflict_rows].reset_index(drop=True)

# 保存修改后的 Parquet 文件
df_fixed.to_parquet(output_parquet, index=False)
df_conflict.to_parquet(conflict_parquet, index=False)

# 同时保存为 CSV 文件
df_fixed.to_csv(output_csv, index=False)
df_conflict.to_csv(conflict_csv, index=False)
non_others_count = df[df["subject"] != "others"].shape[0]

print(f"Number of non-others rows: {non_others_count}")
print(f"Updated Parquet saved to {output_parquet}")
print(f"Updated CSV saved to {output_csv}")
print(f"Conflict Parquet saved to {conflict_parquet}")
print(f"Conflict CSV saved to {conflict_csv}")
print(f"Conflict rows (subject='others'): {conflict_count}")
print(f"Unmatched rows removed (subject='others'): {unmatched_count}")
