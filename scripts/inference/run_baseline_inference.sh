#!/bin/bash

# Set the model name and script path according to your needs
# You can choose the following models:
# - llava1.5
# - llava1.6
# - qwen2
# - qwen2.5
# - llama3
# - mplug
# - internlm2.5

# You can choose the following benchmarks:
# - OK-VQA: CoMEM-inference/OK-VQA/run_okvqa_baseline.py
# - A-OKVQA: CoMEM-inference/AOK-VQA/run_aokvqa_baseline.py
# - Infoseek: CoMEM-inference/infoseek/run_infoseek_baseline.py
# - ViQUAE: CoMEM-inference/Viquae/run_viquae_baseline.py
# - MRAG-Bench: CoMEM-inference/MRAG_Bench/run_mrag_baseline.py
# - OVEN: CoMEM-inference/OVEN/run_oven_baseline.py

MODEL_NAME='qwen2.5'
SCRIPT_PATH="CoMEM-inference/infoseek/run_infoseek_baseline.py"
OUTPUT_DIR="CoMEM-inference/infoseek/output" # Change this to your desired output directory

chmod +x $SCRIPT_PATH
echo "Running baseline inference..."
CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH --model_name MODEL_NAME --output_dir $OUTPUT_DIR

echo "All runs completed!"

# # Note: For OVEN, you need to set the split to 'val_entity' and 'val_query' for the entity and query splits respectively.
# SPLIT='val_entity' 
# SCRIPT_PATH="CoMEM-inference/OVEN/run_oven_baseline.py"
# OUTPUT_DIR="CoMEM-inference/OVEN/output" 
# chmod +x $SCRIPT_PATH
# echo "Running baseline inference..."
# CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH --model_name MODEL_NAME --output_dir $OUTPUT_DIR --split $SPLIT