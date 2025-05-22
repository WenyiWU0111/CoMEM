import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
module_path = "CoMEM-train"
sys.path.append(module_path)
from src_vlm.training.qwenVL_inference import Qwen2_5_VLForConditionalGeneration_new
from src_vlm.training.qwenVL_inference2 import Qwen2VLForConditionalGeneration_new
from dataloader import bench_data_loader 
sys.path.insert(0, "CoMEM-inference")
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

    max_memory = { 
        0: "23GiB",
        1: "23GiB",
    }
    print("Load pretrained model...")
    if 'qwen2.5' in args.model_name:
        checkpoint_path = args.checkpoint_path
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)
        tokenizer = processor.tokenizer
        model = Qwen2_5_VLForConditionalGeneration_new.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                    max_memory=max_memory,
                    low_cpu_mem_usage=True)
    elif 'qwen2' in args.model_name:
        print('load qwen2 model')
        checkpoint_path = args.checkpoint_path
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", use_fast=True)
        tokenizer = processor.tokenizer
        model = Qwen2VLForConditionalGeneration_new.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                    max_memory=max_memory,
                    low_cpu_mem_usage=True)
    
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
        texts = [item['desc'] for item in similar_infos.values()]
        images = [item['image'] for item in similar_infos.values()]
        ans = generate_response_knowledge(args.model_name, processor, model, image, qs, texts, images)
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
    parser.add_argument("--model_name", type=str, default="qwen2.5")
    parser.add_argument("--model-type", type=str, default="CoMEM")
    parser.add_argument("--similar-num", type=int, default=10)
    parser.add_argument("--extra-prompt", type=str, default="Only answer with A/B/C/D, don't response other things.")
    parser.add_argument("--use-rag", type=bool, default=True)
    parser.add_argument("--checkpoint_path", type=str, default="", help="checkpoint path")
    ############# added for mrag benchmark ####################
    args = parser.parse_args()

    eval_model(args)
