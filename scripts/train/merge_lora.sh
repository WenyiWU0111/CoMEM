#!/bin/bash

# Please set the model name according to your needs
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"

cd CoMEM/CoMEM-train
export PYTHONPATH=CoMEM/CoMEM-train:$PYTHONPATH

python src_vlm/merge_lora_weights.py \
    --model-path path of your lora checkpoint before merge \
    --model-base $MODEL_NAME  \
    --save-model-path path to save your merged lora checkpoint \
    --safe-serialization