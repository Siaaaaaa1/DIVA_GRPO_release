import json

def transform_json(input_file):
    # 构造输出文件名
    if input_file.endswith('.json'):
        output_file = input_file.replace('.json', '_id_key.json')
    else:
        output_file = input_file + '_id_key.json'
    
    # 读取原始JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # 转换数据结构
    transformed_data = {}
    for item in original_data:
        item_id = item['id']
        # 创建新条目，包含除id外的所有字段
        new_item = {k: v for k, v in item.items() if k != 'id'}
        transformed_data[item_id] = new_item
    
    # 写入新JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, indent=2, ensure_ascii=False)
    
    print(f"转换完成，结果已保存到 {output_file}")


import json

def transform_json_by_problem(input_file):
    # 构造输出文件名
    if input_file.endswith('.json'):
        output_file = input_file.replace('.json', '_problem.json')
    else:
        output_file = input_file + '_problem.json'
    
    # 读取原始JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # 转换数据结构
    transformed_data = {}
    for item in original_data:
        problem_key = item['problem']
        # 创建新条目，包含除problem外的所有字段
        new_item = {k: v for k, v in item.items() if k != 'problem'}
        transformed_data[problem_key] = new_item
    
    # 写入新JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, indent=2, ensure_ascii=False)
    
    print(f"转换完成，结果已保存到 {output_file}")

# 使用示例
# transform_json_by_problem('/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1/verl/difficulty_variation/mmk12_train_text_variant.json')


# 使用示例
transform_json('/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation/mmk12_train_think_steps_merge.json')