# -*- coding: utf-8 -*-
"""
【最终版】Qwen3-Embedding-0.6B 微调后验证脚本
加载基础模型 + LoRA 权重，计算句子相似度

使用方法：python verify_embedding.py
依赖库：torch, transformers, peft, safetensors

说明：本版本使用 peft 自动加载 LoRA，效果稳定，区分度好。
      手动全量加载版本（v2-v4）存在过度拟合问题，已归档至 _过程与中间文件/。
"""
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

# ============ 路径配置 ============
# 基础模型：填模型名会自动从 ModelScope/HuggingFace 下载
# 也可以填本地路径，如 "./Qwen3-Embedding-0.6B"
BASE_MODEL_PATH = "Qwen/Qwen3-Embedding-0.6B"

# LoRA 权重路径：训练完成后将 checkpoint-6438 文件夹放在项目根目录
# 或修改为你的本地路径，如 "./models/checkpoint-6438"
LORA_PATH = "./checkpoint-6438"

# ============ 加载模型 ============
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
    """获取句子的 embedding 向量"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        # Qwen3-Embedding 取最后一层的均值池化
        embedding = outputs.last_hidden_state.mean(dim=1)
        # 归一化
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.squeeze()


def cosine_similarity(text1, text2):
    """计算两个句子的余弦相似度"""
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    sim = torch.dot(emb1, emb2).item()
    return sim


# ============ 测试用例 ============
test_cases = [
    # (句子1, 句子2, 预期相似度)
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
print("\n说明：")
print("- 相似度范围 [-1, 1]，越接近 1 表示越相似")
print("- 微调后，相似句子的相似度应该更高，不相似句子的相似度应该更低")
print("- 可以对比微调前的基础模型效果来判断微调是否有效")
