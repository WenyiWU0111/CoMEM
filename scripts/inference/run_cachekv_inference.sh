#!/bin/bash

# Set the model name and script path according to your needs
# You can choose the following models:
# - qwen2
# - qwen2.5

# You can choose the following benchmarks:
# - OK-VQA: CoMEM-inference/OK-VQA/run_okvqa_prefixkv_clip.py
# - A-OKVQA: CoMEM-inference/AOK-VQA/run_aokvqa_prefixkv_clip.py
# - Infoseek: CoMEM-inference/infoseek/run_infoseek_prefixkv_clip.py
# - ViQUAE: CoMEM-inference/Viquae/run_viquae_prefixkv_clip.py

MODEL_NAME='qwen2.5'
CHECKPOINT_PATH="" # Change this to your checkpoint path
SCRIPT_PATH="CoMEM-inference/infoseek/run_infoseek_prefixkv_clip.py"
OUTPUT_DIR="CoMEM-inference/infoseek/output" # Change this to your desired output directory
SIMILAR_NUM=10 # Number of relevant image-text pairs to retrieve
TOP_TOKENS=100 # Percentage of top tokens to keep


chmod +x $SCRIPT_PATH
echo "Running prefix CoMEM inference..."
CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH --model_name MODEL_NAME --output_dir $OUTPUT_DIR --similar_num $SIMILAR_NUM --top-tokens $TOP_TOKENS

echo "All runs completed!"