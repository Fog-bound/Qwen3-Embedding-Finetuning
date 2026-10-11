"""
AFQMC 数据集格式转换
原始格式 (text1, text2, label) → ms-swift Embedding 格式

使用：将原始数据保存为 data/train_raw.jsonl，运行 python scripts/data_converter.py
"""
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_file = os.path.join(BASE_DIR, "data", "train_raw.jsonl")
output_file = os.path.join(BASE_DIR, "data", "train_similarity_data.jsonl")

# 自动创建输出目录
os.makedirs(os.path.dirname(output_file), exist_ok=True)

if not os.path.exists(input_file):
    print(f"错误：找不到 {input_file}")
    print("请将原始 AFQMC 数据保存为 data/train_raw.jsonl")
    print("格式：{\"text1\": \"句子A\", \"text2\": \"句子B\", \"label\": 1}")
    exit(1)

with open(input_file, 'r', encoding='utf-8') as f_in, \
     open(output_file, 'w', encoding='utf-8') as f_out:
    count = 0
    skipped = 0
    for line_no, line in enumerate(f_in, 1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            # 检查必需字段
            if "text1" not in data or "text2" not in data or "label" not in data:
                print(f"警告：第 {line_no} 行缺少必需字段，跳过")
                skipped += 1
                continue
            new_data = {
                "messages": [{"role": "user", "content": str(data["text1"])}],
                "positive_messages": [[{"role": "user", "content": str(data["text2"])}]],
                "label": float(data["label"])
            }
            f_out.write(json.dumps(new_data, ensure_ascii=False) + "\n")
            count += 1
        except json.JSONDecodeError as e:
            print(f"错误：第 {line_no} 行 JSON 解析失败 - {e}")
            skipped += 1
        except (ValueError, KeyError) as e:
            print(f"错误：第 {line_no} 行数据转换失败 - {e}")
            skipped += 1

print(f"转换完成！成功 {count} 条，跳过 {skipped} 条 → {output_file}")
