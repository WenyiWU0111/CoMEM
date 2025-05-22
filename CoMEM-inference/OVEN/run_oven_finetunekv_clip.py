"""LLAVA zeroshot OVEN inference script."""
import os
import sys
module_path = "CoMEM-train"
sys.path.append(module_path)
from src_vlm.training.qwenVL_inference import Qwen2_5_VLForConditionalGeneration_new
from src_vlm.training.qwenVL_inference2 import Qwen2VLForConditionalGeneration_new
from src_llm.training.qwenVL_inference import Qwen2_5_VLForConditionalGeneration_new as Qwen2_5_VLForConditionalGeneration_new_llm
import json
import torch
from PIL import Image
import argparse
from tqdm import tqdm
import time
sys.path.insert(0, "CoMEM-inference")
from src.load_model_test import *
from help_functions import *
from io import BytesIO
import base64
from streaming import StreamingDataset

def load_mds_test(mds_dir):
    dataset = StreamingDataset(local=mds_dir,
                           remote=None,
                           shuffle=False,
                           batch_size=100)
    records = []
    for sample in tqdm(dataset, desc="Loading MDS files"):
        # decode base64‐encoded images inside retrieval_info
        decoded_retrieval = []
        for entry in sample["retrieval_info"]:
            decoded_retrieval.append({
                "passage_content": entry["passage_content"],
                "image": base64.b64decode(entry["image"]),
            })
        sample["retrieval_info"] = decoded_retrieval
        records.append(sample)
        
    return records


def process_images_in_batches(processor, model, retrival_kb, batch_size, prompt, args):
    ########## Get output saving path ###########
    file_path = os.path.join(args.output_dir, "{}_{}_{}_{}.jsonl".format(
                    args.model_name, args.model_type, args.split, args.similar_num
                    ))
    # Check if the file exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            output = [json.loads(line) for line in f]
    else:
        output = []
    batch_data = retrival_kb[len(output):]

    print("Generate predictions...")
    # Process images in batches
    for idx, i in enumerate(range(0, len(batch_data), batch_size)):
        if (idx + 1) % 100 == 0:
            print(f"Processing batch {idx}/{len(batch_data)/batch_size}")
        # Subset results for the current batch
        batch_subset = batch_data[i:i+batch_size]

        # Separate the images, questions, and ids
        batch_ids, answers = [], []

        # Load and preprocess the images
        # Generate predictions for the batch
        start_time = time.time()
        for item in batch_subset:
            tmp_img = Image.open(BytesIO(item["image"])).convert("RGB")
            tmp_q = item["question"]
            tmp_q = prompt.format(tmp_q)
            batch_ids.append(item["data_id"])
            ####### Find Similar Images #######
            def process_similar_infos(item, similar_num):
                similar_infos = item["retrieval_info"][:similar_num]
                similar_infos_dict = {}
                for idx, info in enumerate(similar_infos):
                    key = idx
                    fact_img = Image.open(BytesIO(info['image'])).convert("RGB")   
                    if fact_img.size[0] > 512 or fact_img.size[1] > 512:
                        fact_img = fact_img.resize((512, 512), Image.LANCZOS)
                    fact_text = info["passage_content"] or ""
                    similar_infos_dict[key] = {"image": fact_img, "desc": fact_text}
                return similar_infos_dict
            similar_infos = process_similar_infos(item, args.similar_num)
            texts = [item['desc'] for item in similar_infos.values()]
            images = [item['image'] for item in similar_infos.values()]
            ans = generate_response_knowledge(args.model_name, processor, model, tmp_img, tmp_q, texts, images)
            delimiters = ["[/INST]", "ASSISTANT:"]
            for delimiter in delimiters:
                if delimiter in ans:
                    ans = ans.split(delimiter)[-1].strip()
            print(ans)
            answers.append(ans)
            torch.cuda.empty_cache()
        print(f"Time for batch {idx}: {time.time() - start_time}")
        for idx, ans in zip(batch_ids, answers):
            output.append({"data_id": idx, "prediction": ans})
        # save output into jsonl
        with open(file_path, 'w') as f:
            for item in output:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    return output

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val_entity", help="val_entity, val_query")
    parser.add_argument("--model_name", type=str, default="qwen2.5", help="blip2_t5 | blip2_vicuna_instruct | blip2_t5_instruct")
    parser.add_argument("--model_type", type=str, default="CoMEM", help="pretrain_flant5xxl | vicuna13b | flant5xxl")
    parser.add_argument("--output_dir", type=str, default="CoMEM-inference/OVEN/result", help="output directory")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--similar_num", type=int, default=10, help="number of similar samples")
    parser.add_argument("--checkpoint_path", type=str, default="", help="checkpoint path")
    args = parser.parse_args()

    retrival_kb = load_mds_test("path of retirval database for OVEN here")
    if args.split == "val_entity":
        retrival_kb = [item for item in retrival_kb if "entity_val" in item["data_split"]]
    elif args.split == "val_query":
        retrival_kb = [item for item in retrival_kb if "query_val" in item["data_split"]]
    
    # setup device to use
    device = "cuda"
    max_memory = { 
        0: "23GiB",
        1: "23GiB",
    }
    print("Load pretrained model...")
    if 'qwen2.5llm' in args.model_name:
        checkpoint_path = args.checkpoint_path
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)
        model = Qwen2_5_VLForConditionalGeneration_new_llm.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                    max_memory=max_memory,
                    low_cpu_mem_usage=True)
    elif 'qwen2.5' in args.model_name:
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
    
    # Desired batch size
    batch_size = args.batch_size

    PROMPT = """Question: {} 
    For this question, please refer to the given information and then perform step-by-step reasoning, to obtain the final answer. Note that the final answer should be formatted as:
    Reasoning Process: your thinking steps here
    Final answer: \\boxed{{your short answer here}}
    """
    # Run the batch processing function
    output = process_images_in_batches(processor, model, retrival_kb, batch_size, prompt=PROMPT, args=args)

    # save output into jsonl
    file_path = os.path.join(args.output_dir, "{}_{}_{}_{}.jsonl".format(
                    args.model_name, args.model_type, args.split, args.similar_num
                    ))
    with open(file_path, 'w') as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
