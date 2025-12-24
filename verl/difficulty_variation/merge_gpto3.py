import json
from pathlib import Path
from collections import defaultdict
import os
import json
from pathlib import Path

def merge_json_lists_with_priority(file_paths, output_file=None, key_field='id', priority_field='success'):
    """
    合并多个JSON文件（每个文件是字典列表），优先保留 priority_field=True 的字典
    
    参数:
        file_paths: list[str] - JSON文件路径列表
        output_file: str - 合并结果输出文件路径（可选）
        key_field: str - 用于检查重复的键字段（默认 'id'）
        priority_field: str - 用于决定优先级的字段（默认 'success'）
        
    返回:
        list[dict] - 合并后的字典列表（重复项已按优先级处理）
        
    异常:
        ValueError - 如果数据格式不符合要求
        JSONDecodeError - 如果JSON文件格式错误
        FileNotFoundError - 如果文件不存在
    """
    merged_data = {}  # 用字典暂存，按 key_field 去重
    
    for file_path in file_paths:
        # 检查文件是否存在
        if not Path(file_path).exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
            
        # 读取并解析JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(f"文件 {file_path} 不是有效的JSON格式", e.doc, e.pos)
            
        # 验证数据是列表且每个元素是字典
        if not isinstance(data, list):
            raise ValueError(f"文件 {file_path} 不包含列表数据")
            
        for item in data:
            if not isinstance(item, dict):
                raise ValueError(f"文件 {file_path} 包含非字典元素")

            if key_field not in item:
                raise ValueError(f"文件 {file_path} 中的字典缺少关键字段 '{key_field}'")
            if 'step1' in item and item['step1']==[]:
                continue
            if 'step2' in item and item['step2']==[]:
                continue
            if 'variant' in item and item['variant']==[]:
                continue
            key_value = item[key_field]
            
            # 如果 key 不存在，直接存入
            if key_value not in merged_data:
                merged_data[key_value] = item
            else:
                # 如果已存在，检查 priority_field 是否为 True，是则替换
                existing_item = merged_data[key_value]
                if priority_field in item and item[priority_field] == 'success':
                    if priority_field not in existing_item or existing_item[priority_field] != 'success':
                        merged_data[key_value] = item
    
    # 转换为列表
    result = list(merged_data.values())
    
    # 如果需要，写入输出文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
    
    return result

def find_files_with_string(directory, search_string):
    """
    查找目录下文件名包含指定字符串的所有文件（不包括子目录）
    
    参数:
        directory (str): 要搜索的目录路径
        search_string (str): 要在文件名中查找的字符串
        
    返回:
        list: 包含匹配文件路径的列表
    """
    matching_files = []
    
    # 获取目录下所有文件和文件夹
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path) and search_string in file:
            matching_files.append(file_path)
    
    return matching_files


# 示例用法
try:
    result = merge_json_lists_with_priority(
        find_files_with_string('/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation','mmk12_train_think_steps'),
        output_file='/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation/think_steps_merge.json',
        key_field='id',       # 按 'id' 去重
        priority_field='status'  # 优先保留 success=True 的记录
    )
    # result2 = merge_json_lists_with_priority(
    #     find_files_with_string('/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation','mmk12_train_variants'),
    #     output_file='/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation/variants_merge.json',
    #     key_field='id',       # 按 'id' 去重
    #     priority_field='status'  # 优先保留 success=True 的记录
    # )
    print(f"合并成功，共 {len(result)} 条think_step记录")
    # print(f"合并成功，共 {len(result2)} 条variant记录")
except Exception as e:
    print(f"合并失败: {str(e)}")