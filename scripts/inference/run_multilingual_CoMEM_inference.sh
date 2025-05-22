#!/bin/bash

# Set the model name and script path according to your needs
# You can choose the following models:
# - qwen2
# - qwen2.5

# You can choose the following benchmarks:
# - CVQA: CoMEM-inference/CVQA/run_cvqa_rag_clip.py
# - Infoseek: CoMEM-inference/infoseek/run_infoseek_rag_clip.py

# You can choose the following languages for multilingual Infoseek:
# - spanish
# - portuguese
# - chinese
# - russian
# - bulgarian

MODEL_NAME='qwen2.5'
CHECKPOINT_PATH="" # Change this to your checkpoint path

SCRIPT_PATH="CoMEM-inference/infoseek/run_infoseek_rag_clip.py"
OUTPUT_DIR="CoMEM-inference/infoseek/output" # Change this to your desired output directory
SIMILAR_NUM=10 # Number of relevant image-text pairs to retrieve
LANGUAGE="spanish"

chmod +x $SCRIPT_PATH
echo "Running Multilingual Infoseek vanilla RAG inference..."
CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH --model_name MODEL_NAME --output_dir $OUTPUT_DIR --similar_num $SIMILAR_NUM --split $LANGUAGE --checkpoint_path $CHECKPOINT_PATH

SCRIPT_PATH="CoMEM-inference/CVQA/run_cvqa_rag_clip.py"
OUTPUT_DIR="CoMEM-inference/CVQA/output" # Change this to your desired output directory

chmod +x $SCRIPT_PATH
echo "Running CVQA vanilla RAG inference..."
CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH --model_name MODEL_NAME --output_dir $OUTPUT_DIR --similar_num $SIMILAR_NUM --split $LANGUAGE --checkpoint_path $CHECKPOINT_PATH

echo "All runs completed!"
