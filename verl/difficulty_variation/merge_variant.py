import json
from pathlib import Path
from collections import defaultdict

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
            
        # # 验证数据是列表且每个元素是字典
        # if not isinstance(data, list):
        #     raise ValueError(f"文件 {file_path} 不包含列表数据")
            
        for item in data:
            if not isinstance(item, dict):
                raise ValueError(f"文件 {file_path} 包含非字典元素")
                
            if key_field not in item:
                raise ValueError(f"文件 {file_path} 中的字典缺少关键字段 '{key_field}'")
            if item['variant'] == []:
                continue
            key_value = item[key_field]
            
            # 如果 key 不存在，直接存入
            if key_value not in merged_data:
                merged_data[key_value] = item
    
    # 转换为列表
    result = list(merged_data.values())
    
    # 如果需要，写入输出文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
    
    return result


# 示例用法
try:
    result = merge_json_lists_with_priority(
        ['/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation/mmk12_train_variants.json', 
         '/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation/mmk12_train_variants_worker_1.json',
         '/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation/mmk12_train_variants_worker_2.json',
         '/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation/mmk12_train_variants_worker_3.json',
         '/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation/mmk12_train_variants_worker_4.json',
         '/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation/mmk12_train_variants_worker_5.json',
         '/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation/mmk12_train_variants_worker_6.json',
         '/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation/mmk12_train_variants_worker_7.json',
         '/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation/mmk12_train_variants_worker_8.json',
         '/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation/mmk12_train_variants_worker_9.json',
         '/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation/mmk12_variant_backup.json'],
        output_file='/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/verl/difficulty_variation/mmk12_train_variants_merge_output.json',
        key_field='id',       # 按 'id' 去重
    )
    print(f"合并成功，共 {len(result)} 条记录")
except Exception as e:
    print(f"合并失败: {str(e)}")