#!/bin/bash

# You can choose Qwen2 or Qwen2.5 here
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

cd CoMEM/CoMEM-train
export PYTHONPATH=CoMEM/CoMEM-train:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0,1
export LOCAL_WORLD_SIZE=2

GLOBAL_BATCH_SIZE=16  # Adjusted for 2 GPUs
BATCH_PER_DEVICE=1    # Keep same per-device batch size
NUM_DEVICES=2     # Using only 2 GPUs (0,1)
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))


# If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together
# You should freeze the the merger also, becuase the merger is included in the vision_tower.

deepspeed --master_port 2058 src_vlm/training/train.py \
    --use_liger True \
    --lora_enable True \
    --vision_lora True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path path of training data \
    --image_folder '' \
    --knowledge_image_folder '' \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --tune_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output path of model checkpoint \
    --num_train_epochs 2 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 800 \
    --save_total_limit 10 \
    --dataloader_num_workers 2
     \
    --max_grad_norm 3.0\
    --report_to  wandb