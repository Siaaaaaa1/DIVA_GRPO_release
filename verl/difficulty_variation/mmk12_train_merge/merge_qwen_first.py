import pandas as pd
import json

def is_valid_answer(value):
    """检查 answer 是否有效：必须是非空字符串"""
    return isinstance(value, str) and value.strip() != ""

def merge_json(file_a: str, file_b: str, output_file: str):
    # 读取 JSON 文件（假设是 list[dict] 格式）
    with open(file_a, "r", encoding="utf-8") as f:
        data_a = json.load(f)
    with open(file_b, "r", encoding="utf-8") as f:
        data_b = json.load(f)

    # 提前排除掉 answer1 或 answer2 不符合条件的行
    data_a = [d for d in data_a if is_valid_answer(d.get("answer1")) and is_valid_answer(d.get("answer2"))]
    data_b = [d for d in data_b if is_valid_answer(d.get("answer1")) and is_valid_answer(d.get("answer2"))]

    # 筛选
    part1 = [d for d in data_a if d.get("correct_think") is True]
    part2 = [d for d in data_a if d.get("1=2") is True]
    part3 = [d for d in data_b if d.get("correct_think") is True]

    # 合并，保持顺序
    merged = part1 + part2 + part3

    # 去重（按 id）
    seen = set()
    unique = []
    for item in merged:
        if item.get("id") not in seen:
            unique.append(item)
            seen.add(item.get("id"))

    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)

    print(f"合并完成，保存到 {output_file}")

# 示例调用
merge_json("/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation/mmk12_train_qwen/mmk12_think_steps_qwen.json", 
              "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation/mmk12_train_gpto3/mmk12_train_think_steps_gpto3.json", 
              "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation/mmk12_train_merge/mmk12_train_think_steps_first_qwen_merge.json")
