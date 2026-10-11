CUDA_VISIBLE_DEVICES=0
python -m swift.cli.sft \
  --model /mnt/workspace/data/ft/Qwen/Qwen3-Embedding-0.6B \
  --task_type embedding \
  --model_type qwen3_emb \
  --dataset ./train_similarity_data_swift_v2_fixed_oldformat.jsonl \
  --loss_type cosine_similarity \
  --label_names label \
  --train_type lora \
  --output_dir output_model/qwen_embedding_20260806 \
  --torch_dtype fp16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-4 \
  --lora_rank 8










CUDA_VISIBLE_DEVICES=0
python -m swift.cli.sft \
  --model /mnt/workspace/data/ft/Qwen/Qwen3-Embedding-0.6B \
  --task_type embedding \
  --model_type qwen3_emb \
  --dataset /mnt/workspace/data/ft/train_similarity_data_swift_v2_fixed_oldformat.jsonl \
  --columns '{"messages": "messages", "positive_messages": "positive_messages", "label": "label"}' \
  --loss_type cosine_similarity \
  --label_names label \
  --tuner_type lora \
  --output_dir output_model/qwen_embedding_20260806 \
  --torch_dtype float16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-4 \
  --lora_rank 8





CUDA_VISIBLE_DEVICES=0
python -m swift.cli.sft \
  --model /mnt/workspace/data/ft/Qwen/Qwen3-Embedding-0.6B \
  --task_type embedding \
  --model_type qwen3_emb \
  --dataset train_sim \
  --custom_dataset_info '{"train_sim": {"train": "/mnt/workspace/data/ft/train_similarity_data_swift_v2_fixed_oldformat.jsonl"}}' \
  --loss_type cosine_similarity \
  --label_names label \
  --tuner_type lora \
  --output_dir output_model/qwen_embedding_20260806 \
  --torch_dtype float16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-4 \
  --lora_rank 8
