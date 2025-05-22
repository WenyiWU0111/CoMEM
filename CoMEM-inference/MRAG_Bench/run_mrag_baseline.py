import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from dataloader import bench_data_loader 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.load_model_test import *
import torch


def eval_model(args):
    answers_file = f"CoMEM-inference/MRAG_Bench/result/original_{args.model_name}_{args.model_type}.jsonl"
    ans_file = open(answers_file, "w")
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

    for idx, item in enumerate(bench_data_loader(args)):
        qs = item['question']
        ans, past_kv = generate_response(args.model_name, processor, model, item["image_files"], qs)
        delimiters = ["[/INST]", "ASSISTANT:"]
        for delimiter in delimiters:
            if delimiter in ans:
                ans = ans.split(delimiter)[-1].strip()
        print('gt_answer', item['gt_choice'])
        print('answer', ans)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
                                   "qs_id": item['id'],
                                   "prompt": item['prompt'],
                                   "output": ans,
                                   "gt_answer": item['answer'],
                                   "shortuuid": ans_id,
                                   "model_id": 'args.model_name',
                                   "gt_choice": item['gt_choice'],
                                   "scenario": item['scenario'],
                                   "aspect": item['aspect'],
                                   }) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="qwen2")
    parser.add_argument("--model-type", type=str, default="original")
    parser.add_argument("--extra-prompt", type=str, default="Only answer with A/B/C/D, don't response other things")
    ############# added for mrag benchmark ####################
    args = parser.parse_args()

    eval_model(args)
