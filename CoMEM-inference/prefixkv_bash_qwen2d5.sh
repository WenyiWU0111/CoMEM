#!/bin/bash



SCRIPT_PATH="/home/wenyi/CoMEM-inference/OK-VQA/run_okvqa_prefixkv_clip.py"

# Make sure the script is executable
chmod +x $SCRIPT_PATH

# Run the script with different top-tokens values
echo "Running with top-tokens=25..."
CUDA_VISIBLE_DEVICES=7 python $SCRIPT_PATH --top-tokens 5 --model_name 'qwen2.5'

SCRIPT_PATH="/home/wenyi/CoMEM-inference/OK-VQA/run_okvqa_prefixkv_clip.py"

# Make sure the script is executable
chmod +x $SCRIPT_PATH

# Run the script with different top-tokens values
echo "Running with top-tokens=25..."
CUDA_VISIBLE_DEVICES=7 python $SCRIPT_PATH --top-tokens 10 --model_name 'qwen2.5'

echo "All runs completed!"