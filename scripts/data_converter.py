import json

input_file = r"C:\Users\huyu\Desktop\大模型应用开发\qwen-embeding(微调)\train_similarity_train_data.jsonl"
output_file = r"C:\Users\huyu\Desktop\大模型应用开发\qwen-embeding(微调)\train_similarity_data_swift_v2_fixed_oldformat.jsonl"

with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        data = json.loads(line)
        new_data = {
            "messages": [{"role": "user", "content": data["text1"]}],
            "positive_messages": [[{"role": "user", "content": data["text2"]}]],
            "label": float(data["label"])
        }
        f_out.write(json.dumps(new_data, ensure_ascii=False) + "\n")
print("转换完成！")
