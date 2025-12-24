import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import pandas as pd
import io
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
import logging
import pyarrow as pa
import pyarrow.parquet as pq

# 关闭 PaddleOCR 的 DEBUG 日志
logging.getLogger("paddleocr").setLevel(logging.ERROR)

ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)


def ocr_detect_text(img_bytes, conf_threshold=0.85):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = np.array(img)
    except Exception as e:
        print("Image decode error:", e)
        return False

    try:
        results = ocr.ocr(img, cls=True)
    except Exception as e:
        print("OCR failed:", e)
        return False

    if not results or not results[0]:
        return False

    output_texts = []
    confidences = []

    for line in results[0]:
        text, conf = line[1][0], line[1][1]
        output_texts.append(text)
        confidences.append(conf)

    if not confidences:
        return False

    avg_conf = sum(confidences) / len(confidences)
    if avg_conf < conf_threshold:
        return False

    full_text = " ".join(output_texts)
    return len(full_text) > 10


def process_parquet_chunked(input_path, output_path, chunksize=1000):
    """
    使用 pyarrow 按 chunksize 分批读取 Parquet，执行 OCR 并写入输出文件
    """
    table = pq.read_table(input_path)
    num_rows = table.num_rows
    writer = None
    total_rows = 0

    for start in range(0, num_rows, chunksize):
        end = min(start + chunksize, num_rows)
        batch_table = table.slice(start, end - start)
        df = batch_table.to_pandas()

        # 处理 OCR
        flags = []
        for i, row in enumerate(df.itertuples()):
            img_bytes = row.images["bytes"]
            flag = ocr_detect_text(img_bytes)
            flags.append(flag)
            if i % 100 == 0:
                print(f"Processed {i} rows in this chunk...")

        df["text_in_image"] = flags

        # 转为 PyArrow Table 写入
        batch_table = pa.Table.from_pandas(df)
        if writer is None:
            writer = pq.ParquetWriter(output_path, batch_table.schema)
        writer.write_table(batch_table)
        total_rows += len(df)
        print(f"Written {total_rows} rows so far...")

    if writer:
        writer.close()
    print(f"Finished processing {input_path}, total rows: {total_rows}")

# 文件列表
files_to_process = {
    # "MMK12_train_gpto3": "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/DIVA_GRPO/MMK12_train_gpto3.parquet",
    # "GRPO_10000": "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/DIVA_GRPO/GRPO_10000.parquet",
    # "R1-ShareVL-52k-5000": "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/DIVA_GRPO/R1-ShareVL-52k_merge_fixed_correct_5000.parquet",
    # "R1-ShareVL-52k-rest": "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/DIVA_GRPO/R1-ShareVL-52k_merge_fixed_correct_rest.parquet",
    # "MMK12_train_thinkq_variantq": "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/DIVA_GRPO/MMK12_train_thinkq_variantq.parquet",
    # "MMK12_train_thinkq_variantg": "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/DIVA_GRPO/MMK12_train_thinkq_variantg.parquet",
    "GRPO_18000": "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/DIVA_GRPO/GRPO_18000.parquet"
}

output_dir = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1_Share_VL_Weighting/datasets/OCR/"

for name, input_path in files_to_process.items():
    output_path = f"{output_dir}ocr_{name}.parquet"
    print(f"Processing {input_path} -> {output_path}")
    process_parquet_chunked(input_path, output_path)
