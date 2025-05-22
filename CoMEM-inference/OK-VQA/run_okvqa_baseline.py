"""Zeroshot Inference - OKVQA"""
import os
import json
import torch
from PIL import Image
import argparse
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.load_model_test import *

from io import BytesIO
import dask.dataframe as dd

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen2.5", help="model name")
    parser.add_argument("--model_type", type=str, default="baseline", help="baseline or steered")
    parser.add_argument("--output_dir", type=str, default="CoMEM-inference/OK-VQA/result", help="output directory")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    args = parser.parse_args()

    df = dd.read_parquet("hf://datasets/lmms-lab/OK-VQA/data/val2014-*.parquet").compute()
    # setup device to use
    device = 'cuda'
    PROMPT = """Question: {} 
    For this question, please perform step-by-step reasoning, to obtain the final answer. Your final answer should be short and concise.
    Note that the final answer should be formatted as:
    Reasoning Process: all thinking steps
    Final answer: \\boxed{{your short answer here}}"""
    print("Load pretrained model...")
    model_path_map = {
        'llava1.5': 'llava-hf/llava-1.5-7b-hf',
        'llava1.6': 'llava-hf/llava-v1.6-mistral-7b-hf',
        'qwen2': "Qwen/Qwen2-VL-7B-Instruct",
        'qwen2.5': "Qwen/Qwen2.5-VL-7B-Instruct",
        'llama3': "lmms-lab/llama3-llava-next-8b",
        'mplug': "mPLUG/mPLUG-Owl3-7B-240728",
        'internlm2.5': "internlm/internlm-xcomposer2d5-7b"
    }
    model_path = model_path_map.get(args.model_name, None)
    processor, tokenizer, model = load_model(args.model_name, model_path)
    
    output = []
    # Load existing progress if available
    output_file_path = os.path.join(args.output_dir, "okvqa_{}_{}.jsonl".format(
                args.model_name, args.model_type))
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r') as f:
            for line in f:
                output.append(json.loads(line))
    else:
        output = []
    
    selected_batch_data = df.iloc[len(output):]
    print('start from ', len(output))
    for i in range(selected_batch_data.shape[0]):
        item = selected_batch_data.iloc[i]
        image = Image.open(BytesIO(item['image']['bytes'])).convert("RGB")
        if image.size[0] > 512 or image.size[1] > 512:
            image = image.resize((512, 512), Image.LANCZOS)    
        question = item['question']
        prompt = PROMPT.format(question)
        ans, past_kv = generate_response(args.model_name, processor, model, image, prompt)
        delimiters = ["[/INST]", "ASSISTANT:"]
        for delimiter in delimiters:
            if delimiter in ans:
                ans = ans.split(delimiter)[-1].strip()
        torch.cuda.empty_cache()
        print(ans)
        output.append({"data_id": item['question_id'],
                       "question": question,
                       "prediction": ans,
                       "answer": str(item['answers']),
                       "question_type": item['question_type']
                       })
        # Save output every 10 iteratioclens
        if (i + 1) % 10 == 0:
            with open(output_file_path, 'w') as f:
                print('save output every 10 iterations to', output_file_path)
                for item in output:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Save any remaining output
    if output:
        with open(output_file_path, 'w') as f:
            for item in output:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")