#!/bin/bash
# Qwen3-Embedding-0.6B 句子相似度模型 LoRA 微调训练脚本
# 最终跑通版本（ms-swift 新版参数）
#
# 使用前请确认：
# 1. 已安装 ms-swift（需 torch 2.4+）
# 2. 训练数据已放在 data/train_similarity_data.jsonl
# 3. 如有本地模型，修改 MODEL_PATH 为本地路径
#
# 训练时长参考：A10 24GB 显卡，34335 条数据，3 epoch，约 32 分钟，显存约 1.7GB

set -e  # 任何命令失败就停止

CUDA_VISIBLE_DEVICES=0

# 模型路径：可填模型名（自动下载）或本地路径
MODEL_PATH="Qwen/Qwen3-Embedding-0.6B"
# 训练数据路径（相对路径）
DATASET_PATH="./data/train_similarity_data.jsonl"
# 输出目录
OUTPUT_DIR="./output_model/qwen_embedding"

# 检查数据集是否存在
if [ ! -f "$DATASET_PATH" ]; then
    echo "错误：找不到数据集 $DATASET_PATH"
    echo "请先运行 python scripts/data_converter.py 生成训练数据"
    exit 1
fi

echo "开始训练..."
echo "模型: $MODEL_PATH"
echo "数据: $DATASET_PATH"
echo "输出: $OUTPUT_DIR"
echo ""

python -m swift.cli.sft \
  --model $MODEL_PATH \
  --task_type embedding \
  --model_type qwen3_emb \
  --dataset $DATASET_PATH \
  --loss_type cosine_similarity \
  --label_names label \
  --tuner_type lora \
  --output_dir $OUTPUT_DIR \
  --torch_dtype float16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-4 \
  --lora_rank 8

echo ""
echo "训练完成！"
echo "LoRA 权重位置: $OUTPUT_DIR/checkpoint-*/"
echo "验证效果: python scripts/verify_embedding.py（需将 checkpoint 复制到项目根目录）"
