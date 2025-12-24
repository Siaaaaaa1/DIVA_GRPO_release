import json
import sys

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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python script.py <input_json_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    transform_json(input_file)


# # 使用示例
# transform_json('/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation/mmk12_train_think_steps_gpto3.json')
# transform_json('/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation/mmk12_train_variants_gpto3.json')