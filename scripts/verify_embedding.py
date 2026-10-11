# -*- coding: utf-8 -*-
"""
Qwen3-Embedding-0.6B 微调后验证脚本
加载基础模型 + LoRA 权重，计算句子相似度

使用方法：python verify_embedding.py
依赖：torch, transformers, peft, safetensors
"""
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

# 基础模型：填模型名自动下载，也可填本地路径
BASE_MODEL_PATH = "Qwen/Qwen3-Embedding-0.6B"
# LoRA 权重：训练后将 checkpoint-6438 放在项目根目录
LORA_PATH = "./checkpoint-6438"

print("正在加载基础模型...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
base_model = AutoModel.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="cpu"
)

print("正在加载 LoRA 权重...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()
print("模型加载完成！\n")


def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.squeeze()


def cosine_similarity(text1, text2):
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    return torch.dot(emb1, emb2).item()


test_cases = [
    ("支付宝扣款与花呗不一致", "花呗账单与付款金额不一致", "高（相似）"),
    ("今天天气真好", "今天天气不错", "高（相似）"),
    ("我喜欢吃苹果", "我喜欢打篮球", "低（不相似）"),
    ("北京是中国的首都", "北京是中国的首都", "极高（完全相同）"),
    ("如何学习编程", "编程学习方法", "高（相似）"),
    ("猫咪很可爱", "狗狗很忠诚", "低（不相似）"),
]

print("=" * 70)
print(f"{'句子1':<25} {'句子2':<25} {'相似度':<10} {'预期'}")
print("=" * 70)

for text1, text2, expected in test_cases:
    sim = cosine_similarity(text1, text2)
    print(f"{text1:<25} {text2:<25} {sim:<10.4f} {expected}")

print("=" * 70)
