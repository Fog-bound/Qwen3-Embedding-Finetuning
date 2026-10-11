# Qwen3-Embedding 句子相似度模型微调

使用 ms-swift 框架对 Qwen3-Embedding-0.6B 进行 LoRA 微调，针对 AFQMC 蚂蚁金融问句匹配数据集训练句子相似度模型。

## 项目简介

本项目实现了对 Qwen3-Embedding-0.6B 模型的轻量微调，通过 LoRA 技术仅训练 0.84% 的参数，即可定制化模型在金融领域的语义向量检索能力。

## 技术架构

```
AFQMC 数据集 → 格式转换（messages+positive_messages）→ ms-swift LoRA 微调 → 验证推理
```

## 训练结果

| 指标 | 值 |
|------|-----|
| 基础模型 | Qwen3-Embedding-0.6B |
| 训练数据 | AFQMC 34,335 条 |
| 微调方法 | LoRA (rank=8, alpha=32) |
| 可训练参数 | 5.04M (0.84%) |
| 训练轮数 | 3 epochs / 6438 steps |
| 最终 Loss | 0.4178 |
| 训练时长 | 32 分钟（阿里云 A10 24GB） |
| 显存占用 | 1.71 GB |

## 效果验证

| 句子对 | 相似度 | 预期 |
|--------|--------|------|
| 支付宝扣款与花呗不一致 / 花呗账单与付款金额不一致 | 0.8433 | 高（相似） |
| 今天天气真好 / 今天天气不错 | 0.9468 | 高（相似） |
| 我喜欢吃苹果 / 我喜欢打篮球 | 0.7061 | 低（不相似） |
| 北京是中国的首都 / 北京是中国的首都 | 1.0000 | 极高（相同） |
| 如何学习编程 / 编程学习方法 | 0.8623 | 高（相似） |
| 猫咪很可爱 / 狗狗很忠诚 | 0.7969 | 低（不相似） |

## 快速开始

### 1. 环境准备

```bash
# 安装 ms-swift（需 torch 2.4+）
pip install ms-swift torch torchvision

# 安装推理依赖
pip install transformers peft safetensors
```

### 2. 数据准备

```bash
# 转换 AFQMC 数据集为 swift 格式
python scripts/data_converter.py
# 输出：data/train_similarity_data.jsonl
```

数据格式（关键：positive_messages 需两层嵌套）：
```json
{"messages": [{"role": "user", "content": "句子A"}],
 "positive_messages": [[{"role": "user", "content": "句子B"}]],
 "label": 1.0}
```

### 3. 开始训练

```bash
# 云端训练（阿里云 PAI DSW / 任意 GPU 环境）
bash train.sh
```

### 4. 验证效果

```bash
# 本地加载 LoRA 权重，计算句子相似度
python scripts/verify_embedding.py
```

## 项目结构

```
Qwen3-Embedding-Finetuning/
├── scripts/
│   ├── data_converter.py      # 数据格式转换脚本
│   └── verify_embedding.py    # 推理验证脚本
├── data/
│   └── train_similarity_data.jsonl  # 训练数据（34335条）
├── train.sh                   # 训练启动脚本
├── requirements.txt           # 依赖清单
└── README.md
```

## 关键技术点

1. **数据格式**：swift embedding 任务要求 `positive_messages` 为两层列表 `[[{...}]]`
2. **LoRA 微调**：仅训练 q/k/v/o/gate/up/down 投影层，参数量小、训练快
3. **云端训练**：阿里云 PAI DSW 平台，A10 显卡，32 分钟完成训练

## 环境依赖

- Python 3.10+
- PyTorch 2.4+（CUDA 12.1）
- ms-swift 最新版
- transformers, peft, safetensors

## 作者

胡蔚 - AI 应用开发工程师
- GitHub: [github.com/Fog-bound](https://github.com/Fog-bound)
