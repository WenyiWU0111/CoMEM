#!/bin/bash

# Set the model name and script path according to your needs
# You can choose the following models:
# - qwen2.5llm

# You can choose the following benchmarks:
# - Infoseek: CoMEM-inference/infoseek/run_infoseek_finetunekv_clip_llm.py
# - OVEN: CoMEM-inference/OVEN/run_oven_finetunekv_clip.py

MODEL_NAME='qwen2.5llm'
CHECKPOINT_PATH="" # Change this to your checkpoint path

SCRIPT_PATH="CoMEM-inference/infoseek/run_infoseek_finetunekv_clip_llm.py"
OUTPUT_DIR="CoMEM-inference/infoseek/output" # Change this to your desired output directory
SIMILAR_NUM=10 # Number of relevant image-text pairs to retrieve


chmod +x $SCRIPT_PATH
echo "Running finetune CoMEM inference..."
CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH --model_name MODEL_NAME --output_dir $OUTPUT_DIR --similar_num $SIMILAR_NUM --checkpoint_path $CHECKPOINT_PATH

echo "All runs completed!"

# # Note: For OVEN, you need to set the split to 'val_entity' and 'val_query' for the entity and query splits respectively.
# SPLIT='val_entity' 
# SCRIPT_PATH="CoMEM-inference/OVEN/run_oven_finetune_clip.py"
# OUTPUT_DIR="CoMEM-inference/OVEN/output" 
# chmod +x $SCRIPT_PATH
# echo "Running baseline inference..."
# CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH --model_name MODEL_NAME --output_dir $OUTPUT_DIR --split $SPLIT --similar_num $SIMILAR_NUM --checkpoint_path $CHECKPOINT_PATH