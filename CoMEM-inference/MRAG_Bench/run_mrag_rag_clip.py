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
from streaming import StreamingDataset
import base64 
from tqdm import tqdm
from io import BytesIO

def load_mds(mds_dir):
    dataset = StreamingDataset(local=mds_dir,
                           remote=None,
                           shuffle=False,
                           batch_size=1)
    records = []
    for sample in tqdm(dataset, desc="Loading MDS files"):
        records.append(sample)
    return records

def eval_model(args):
    answers_file = f"CoMEM-inference/MRAG_Bench/result/clip/{args.model_name}_{args.model_type}_{args.similar_num}.jsonl"
    if os.path.exists(answers_file):
        exists_file = open(answers_file, "r")
        exists_results = [json.loads(line) for line in exists_file.readlines()]
        exists_file.close()
    else:
        exists_results = []
    ans_file = open(answers_file, "w")
    for item in exists_results:
        ans_file.write(json.dumps(item) + "\n")

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
    image_processor, tokenizer, model = load_model(args.model_name, model_path)

    retrival_kb = load_mds("path of retirval database for MRAG-Bench here")
    retrival_kb = {item['id']: item for item in retrival_kb}
    
    for idx, item in enumerate(bench_data_loader(args)):
        if idx < len(exists_results):
            continue
        def process_similar_infos(item, similar_num):
            similar_infos = item["retrieval_info"][:similar_num]
            similar_infos_dict = {}
            for idx, info in enumerate(similar_infos):
                key = idx
                image_data = base64.b64decode(info['image'])
                fact_img = Image.open(BytesIO(image_data)).convert("RGB")
                if fact_img.size[0] > 512 or fact_img.size[1] > 512:
                    fact_img = fact_img.resize((512, 512), Image.LANCZOS) 
                fact_text = info["passage_content"]
                similar_infos_dict[key] = {"image": fact_img, "desc": fact_text}
            return similar_infos_dict
        k = retrival_kb[str(item['id'])]
        similar_infos = process_similar_infos(k, args.similar_num)
        image = item["image_files"]
        qs = item['question']
        ans = generate_response_rag(args.model_name, image_processor, model, image, qs, similar_infos, tokenizer)
        delimiter = "[/INST]"
        if delimiter in ans:
            ans = ans.split(delimiter)[-1].strip()
        torch.cuda.empty_cache()
        print('gt_answer', item['gt_choice'])
        print('answer', ans)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
                                   "qs_id": item['id'],
                                   "prompt": item['prompt'],
                                   "output": ans,
                                   "gt_answer": item['answer'],
                                   "shortuuid": ans_id,
                                   "model_id": args.model_name,
                                   "gt_choice": item['gt_choice'],
                                   "scenario": item['scenario'],
                                   "aspect": item['aspect'],
                                   }) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="mplug")
    parser.add_argument("--model-type", type=str, default="rag_clip")
    parser.add_argument("--similar-num", type=int, default=10)
    parser.add_argument("--extra-prompt", type=str, default="Only answer with A/B/C/D, don't response other things.")
    parser.add_argument("--use_rag", type=bool, default=True)
    ############# added for mrag benchmark ####################
    args = parser.parse_args()

    eval_model(args)
