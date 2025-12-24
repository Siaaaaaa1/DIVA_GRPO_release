# import pandas as pd
# import pyarrow as pa
# import pyarrow.parquet as pq
# import uuid

# # 文件路径
# file1_path = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/ablation_dataset/MMK12_train_gpto3.parquet"
# file2_paths = [
#     "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1/datasets/R1-ShareVL-52K/train-00000-of-00003.parquet",
#     "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1/datasets/R1-ShareVL-52K/train-00001-of-00003.parquet",
#     "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1/datasets/R1-ShareVL-52K/train-00002-of-00003.parquet"
# ]
# output_path = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/R1-Share-VL/R1-ShareVL-52k_merge.parquet"

# # file1 列顺序
# file1_columns = [
#     'id', 'problem', 'answer', 'subject', 'images',
#     'difficulty', 'category', 'step1', 'step2',
#     'answer1', 'answer2', 'correct_think', 'variant'
# ]

# # ---------- 处理 file1 行 ----------
# def process_file1_row(row):
#     """保持 file1 原有数据结构"""
#     new_row = {}
#     for col in file1_columns:
#         val = row.get(col, None)
#         new_row[col] = val
#     return new_row

# # ---------- 处理 file2 行 ----------
# def process_file2_row(row):
#     """转换成 file1 格式，保持嵌套结构"""
#     if row.get("problem", "").startswith("<image>As shown in the figure"):
#         return None

#     new_row = {}
#     new_row["id"] = str(uuid.uuid4())
#     new_row["problem"] = row.get("problem", "")
#     new_row["answer"] = row.get("answer", "")

#     # variant 是 list<string>
#     new_row["variant"] = row.get("new_questions", [""])

#     # images 是 struct<bytes, string>
#     images_list = row.get("images", [])
#     image_dict = images_list[0] if images_list else {}
#     new_row["images"] = {
#         "bytes": image_dict.get("bytes", b""),
#         "path": ""
#     }

#     # 其它字段
#     new_row["subject"] = "others"
#     new_row["category"] = "origin_problem"
#     new_row["step1"] = [""]
#     new_row["step2"] = [""]
#     new_row["answer1"] = ""
#     new_row["answer2"] = ""
#     new_row["difficulty"] = 5
#     new_row["correct_think"] = False
#     return new_row

# # ---------- 遍历所有文件 ----------
# all_rows = []

# # file1
# df1 = pd.read_parquet(file1_path)
# for _, row in df1.iterrows():
#     processed = process_file1_row(row.to_dict())
#     all_rows.append(processed)

# # file2
# for file2_path in file2_paths:
#     df2 = pd.read_parquet(file2_path)
#     for col in ["idx", "new_questions"]:
#         if col in df2.columns:
#             df2 = df2.drop(columns=[col])
#     for _, row in df2.iterrows():
#         processed = process_file2_row(row.to_dict())
#         if processed is not None:
#             all_rows.append(processed)

# # ---------- 构造 Arrow 表 ----------
# def build_arrow_column(name, values):
#     """根据列名构造嵌套列类型"""
#     if name == "images":
#         # struct<bytes: binary, path: string>
#         bytes_array = pa.concat_arrays([pa.array([v.get("bytes", b"") for v in values])])
#         path_array = pa.concat_arrays([pa.array([v.get("path", "") for v in values])])
#         return pa.StructArray.from_arrays([bytes_array, path_array], ["bytes", "path"])
#     elif name in ["step1", "step2", "variant"]:
#         # list<string>
#         return pa.array(values, type=pa.list_(pa.string()))
#     elif name == "difficulty":
#         return pa.array(values, type=pa.int64())
#     elif name == "correct_think":
#         return pa.array(values, type=pa.bool_())
#     else:
#         return pa.array(values)



# arrow_columns = {}
# for col in file1_columns:
#     col_values = [row[col] for row in all_rows]
#     arrow_columns[col] = build_arrow_column(col, col_values)

# table = pa.table(arrow_columns)
# pq.write_table(table, output_path)
# print(f"文件已生成：{output_path}")



import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import uuid

# 文件路径
file1_path = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/ablation_dataset/MMK12_train_gpto3.parquet"
file2_paths = [
    "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1/datasets/R1-ShareVL-52K/train-00000-of-00003.parquet",
    "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1/datasets/R1-ShareVL-52K/train-00001-of-00003.parquet",
    "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1/datasets/R1-ShareVL-52K/train-00002-of-00003.parquet"
]
output_path = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/MMK12/R1-Share-VL/R1-ShareVL-52k_merge.parquet"

# file1 列顺序
file1_columns = [
    'id', 'problem', 'answer', 'subject', 'images',
    'difficulty', 'category', 'step1', 'step2',
    'answer1', 'answer2', 'correct_think', 'variant'
]

# --------- 小工具：稳妥转 bytes ----------
# def to_bytes(x):
#     if x is None:
#         return None
#     if isinstance(x, (bytes, bytearray)):
#         return bytes(x)
#     try:
#         import pyarrow as pa
#         if isinstance(x, pa.Buffer):
#             return x.to_pybytes()
#     except Exception:
#         pass
#     if isinstance(x, memoryview):
#         return x.tobytes()
#     # 不强制把 str 当图片字节，避免脏数据
#     return None

def to_bytes(x):
    if x is None:
        return None
    if isinstance(x, bytes):
        return x
    if isinstance(x, bytearray):
        return bytes(x)
    if isinstance(x, str):
        x = x.strip()
        if x == "":
            return None
        try:
            return base64.b64decode(x)
        except Exception:
            # 解码失败当 null
            return None
    # memoryview
    if isinstance(x, memoryview):
        return x.tobytes()
    # pyarrow Buffer
    try:
        import pyarrow as pa
        if isinstance(x, pa.Buffer):
            return x.to_pybytes()
    except Exception:
        pass
    return None

# ---------- 处理 file1 行 ----------
def process_file1_row(row):
    """保持 file1 原有数据结构"""
    new_row = {}
    for col in file1_columns:
        val = row.get(col, None)
        new_row[col] = val
    return new_row

# ---------- 处理 file2 行 ----------
def process_file2_row(row):
    """转换成 file1 格式，保持嵌套结构"""
    if row.get("problem", "").startswith("<image>As shown in the figure"):
        return None

    new_row = {}
    new_row["id"] = str(uuid.uuid4())
    new_row["problem"] = row.get("problem", "")
    new_row["answer"] = row.get("answer", "")

    # variant <- new_questions (list[str])
    v = list(row.get("new_questions"))
    if isinstance(v, list):
        new_row["variant"] = [str(x) if x is not None else "" for x in v]
    else:
        new_row["variant"] = [""]

    # images 是 struct<bytes, string>
    images_list = row.get("images", [])
    image_dict = images_list[0]
    new_row["images"] = {
        "bytes": to_bytes(image_dict.get("bytes")),
        "path": ""
    }

    # 其它字段
    new_row["subject"] = "others"
    new_row["category"] = "origin_problem"
    new_row["step1"] = [""]
    new_row["step2"] = [""]
    new_row["answer1"] = ""
    new_row["answer2"] = ""
    new_row["difficulty"] = 5
    new_row["correct_think"] = False
    return new_row

# ---------- 遍历所有文件 ----------
all_rows = []

# file1
df1 = pd.read_parquet(file1_path)
for _, row in df1.iterrows():
    processed = process_file1_row(row.to_dict())
    all_rows.append(processed)

# file2
for file2_path in file2_paths:
    df2 = pd.read_parquet(file2_path)
    # 只删 idx，保留 new_questions 以填充 variant
    if "idx" in df2.columns:
        df2 = df2.drop(columns=["idx"])
    for _, row in df2.iterrows():
        processed = process_file2_row(row.to_dict())
        if processed is not None:
            all_rows.append(processed)

# ---------- 构造 Arrow 表 ----------
def build_arrow_column(name, values):
    """根据列名构造嵌套列类型"""
    if name == "images":
        # struct<bytes: binary, path: string>
        bytes_list = []
        path_list = []
        for v in values:
            if v is None:
                bytes_list.append(None)
                path_list.append(None)
                continue
            # v 期望是 dict
            if isinstance(v, dict):
                b = to_bytes(v.get("bytes"))
                p = v.get("path", "")
            else:
                # 兜底：未知类型
                b, p = None, ""
            bytes_list.append(b)
            path_list.append("" if p is None else str(p))
        bytes_array = pa.array(bytes_list, type=pa.binary())
        path_array = pa.array(path_list, type=pa.string())

        if isinstance(bytes_array, pa.ChunkedArray):
            bytes_array = pa.concat_arrays(list(bytes_array.chunks))
        if isinstance(path_array, pa.ChunkedArray):
            path_array = pa.concat_arrays(list(path_array.chunks))
        struct_array = pa.StructArray.from_arrays([bytes_array, path_array], ["bytes", "path"])

        return struct_array
        #return pa.StructArray.from_arrays([bytes_array, path_array], ["bytes", "path"])

    elif name in ["step1", "step2", "variant"]:
        norm = []
        for lst in values:
            lst = list(lst)
            if lst is None:
                norm.append([])
            elif isinstance(lst, list):
                norm.append([("" if x is None else str(x)) for x in lst])
            else:
                norm.append([str(lst)])
        return pa.array(norm, type=pa.list_(pa.string()))

    elif name == "difficulty":
        return pa.array([None if v is None else int(v) for v in values], type=pa.int64())

    elif name == "correct_think":
        return pa.array([bool(v) if v is not None else False for v in values], type=pa.bool_())

    else:
        # 其他列让 Arrow 自推断；如果包含混合类型会自动升级到 large_string/object
        return pa.array(values)


writer = None
chunk_size = 5000  # 每 5k 行写一次

for chunk_start in range(0, len(all_rows), chunk_size):
    chunk = all_rows[chunk_start:chunk_start+chunk_size]
    
    arrow_columns = {}
    for col in file1_columns:
        col_values = [row.get(col, None) for row in chunk]
        arrow_columns[col] = build_arrow_column(col, col_values)
    
    table = pa.table(arrow_columns)
    
    if writer is None:
        writer = pq.ParquetWriter(output_path, table.schema)
    
    writer.write_table(table)

if writer:
    writer.close()


# arrow_columns = {}
# for col in file1_columns:
#     col_values = [row.get(col, None) for row in all_rows]
#     arrow_columns[col] = build_arrow_column(col, col_values)

# table = pa.table(arrow_columns)
# pq.write_table(table, output_path)
print(f"文件已生成：{output_path}")
