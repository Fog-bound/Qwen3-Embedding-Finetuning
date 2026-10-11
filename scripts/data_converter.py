"""
AFQMC 数据集格式转换脚本
将原始格式 (text1, text2, label) 转换为 ms-swift Embedding 训练格式

原始格式每行: {"text1": "句子A", "text2": "句子B", "label": 1}
转换后每行: {"messages": [{"role": "user", "content": "句子A"}],
             "positive_messages": [[{"role": "user", "content": "句子B"}]],
             "label": 1.0}

使用方法:
    1. 将原始 AFQMC 数据保存为 data/train_raw.jsonl
    2. 运行: python scripts/data_converter.py
    3. 输出: data/train_similarity_data.jsonl (可直接用于训练)
"""
import json
import os

# 相对路径，基于项目根目录运行
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_file = os.path.join(BASE_DIR, "data", "train_raw.jsonl")
output_file = os.path.join(BASE_DIR, "data", "train_similarity_data.jsonl")

if not os.path.exists(input_file):
    print(f"错误：找不到输入文件 {input_file}")
    print("请将原始 AFQMC 数据保存为 data/train_raw.jsonl")
    print("原始格式: {\"text1\": \"句子A\", \"text2\": \"句子B\", \"label\": 1}")
    exit(1)

with open(input_file, 'r', encoding='utf-8') as f_in, \
     open(output_file, 'w', encoding='utf-8') as f_out:
    count = 0
    for line in f_in:
        data = json.loads(line)
        new_data = {
            "messages": [{"role": "user", "content": data["text1"]}],
            "positive_messages": [[{"role": "user", "content": data["text2"]}]],
            "label": float(data["label"])
        }
        f_out.write(json.dumps(new_data, ensure_ascii=False) + "\n")
        count += 1

print(f"转换完成！共 {count} 条数据")
print(f"输出文件: {output_file}")
