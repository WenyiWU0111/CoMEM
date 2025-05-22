#!/bin/bash

# Set the model name and script path according to your needs
# You can choose the following models:
# - qwen2
# - qwen2.5

# You can choose the following benchmarks:
# - CVQA: CoMEM-inference/CVQA/run_cvqa_baseline.py
# - Infoseek: CoMEM-inference/infoseek/run_infoseek_baseline.py

# You can choose the following languages for multilingual Infoseek:
# - spanish
# - portuguese
# - chinese
# - russian
# - bulgarian

MODEL_NAME='qwen2.5'

SCRIPT_PATH="CoMEM-inference/infoseek/run_infoseek_baseline.py"
OUTPUT_DIR="CoMEM-inference/infoseek/output" # Change this to your desired output directory
LANGUAGE="spanish"

chmod +x $SCRIPT_PATH
echo "Running Multilingual Infoseek baseline inference..."
CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH --model_name MODEL_NAME --output_dir $OUTPUT_DIR --split $LANGUAGE

SCRIPT_PATH="CoMEM-inference/CVQA/run_cvqa_baseline.py"
OUTPUT_DIR="CoMEM-inference/CVQA/output" # Change this to your desired output directory

chmod +x $SCRIPT_PATH
echo "Running CVQA baseline inference..."
CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH --model_name MODEL_NAME --output_dir $OUTPUT_DIR

echo "All runs completed!"
