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
# - OK-VQA: CoMEM-inference/OK-VQA/run_okvqa_rag_clip.py
# - A-OKVQA: CoMEM-inference/AOK-VQA/run_aokvqa_rag_clip.py
# - Infoseek: CoMEM-inference/infoseek/run_infoseek_rag_clip.py
# - ViQUAE: CoMEM-inference/Viquae/run_viquae_rag_clip.py
# - MRAG-Bench: CoMEM-inference/MRAG_Bench/run_mrag_rag_clip.py
# - OVEN: CoMEM-inference/OVEN/run_oven_rag_clip.py

MODEL_NAME='qwen2.5'
SCRIPT_PATH="CoMEM-inference/infoseek/run_infoseek_rag_clip.py"
OUTPUT_DIR="CoMEM-inference/infoseek/output" # Change this to your desired output directory
SIMILAR_NUM=10 # Number of relevant image-text pairs to retrieve

chmod +x $SCRIPT_PATH
echo "Running vanilla RAG inference..."
CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH --model_name MODEL_NAME --output_dir $OUTPUT_DIR --similar_num $SIMILAR_NUM

echo "All runs completed!"

# # Note: For OVEN, you need to set the split to 'val_entity' and 'val_query' for the entity and query splits respectively.
# SPLIT='val_entity' 
# SCRIPT_PATH="CoMEM-inference/OVEN/run_oven_rag_clip.py"
# OUTPUT_DIR="CoMEM-inference/OVEN/output" 
# chmod +x $SCRIPT_PATH
# echo "Running baseline inference..."
# CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH --model_name MODEL_NAME --output_dir $OUTPUT_DIR --split $SPLIT --similar_num $SIMILAR_NUM