"""zeroshot OVEN inference script."""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
from PIL import Image
from src.load_model_test import *
import argparse
import time

def load_and_process_image(item):
    # Load and preprocess the image
    path = f"your image folder path/{item['image_id']}.jpg"
    raw_image = Image.open(path).convert("RGB")        
    if raw_image.size[0] > 512 or raw_image.size[1] > 512:
        raw_image = raw_image.resize((512, 512), Image.LANCZOS)  
    return raw_image, item["question"], item["data_id"]

def process_images_in_batches(processor, model, batch_data, batch_size, prompt, args):
    ########## Get output saving path ###########
    file_path = os.path.join(args.output_dir, "{}_{}_{}.jsonl".format(
                    args.model_name, args.model_type, args.split
                    ))
    # Check if the file exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            output = [json.loads(line) for line in f]
    else:
        output = []
    batch_data = batch_data[len(output):]

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
            tmp_img, tmp_q, tmp_id = load_and_process_image(item)
            batch_ids.append(tmp_id)
            tmp_q = prompt.format(tmp_q)
            ans, past_kv = generate_response(args.model_name, processor, model, image, prompt)
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
    parser.add_argument("--split", type=str, default="val_entity", help="val_entity, val_query, test_entity, test_query, or human")
    parser.add_argument("--model_name", type=str, default="qwen2", help="blip2_t5 | blip2_vicuna_instruct | blip2_t5_instruct")
    parser.add_argument("--model_type", type=str, default="original", help="pretrain_flant5xxl | vicuna13b | flant5xxl")
    parser.add_argument("--output_dir", type=str, default="CoMEM-inference/OVEN/result", help="output directory")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")

    args = parser.parse_args()

    split2data = {
        "val_entity": "CoMEM-inference/OVEN/oven_entity_val_3k.jsonl",
        "val_query": "CoMEM-inference/OVEN/oven_query_val.jsonl",
    }

    # Read the input JSONL file
    print('Read the input JSONL file')
    with open(split2data[args.split], 'r') as f:
        batch_data = [json.loads(line) for line in f]

    # double check data exists:
    print('double check data exists')
    not_exist = []
    clean_batch_data = []
    for idx, item in enumerate(batch_data):
        if idx % 1000 == 0:
            print(f"Processing {idx}/{len(batch_data)}")
        path = f"your image folder path/{item['image_id']}.jpg"
        # check path exists
        if not os.path.exists(path):
            print(f"Image not found: {path}")
            not_exist.append(item["image_id"])
        else:
            clean_batch_data.append(item)
    print(len(not_exist))
    
    # setup device to use
    device = "cuda"
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
            
    # Desired batch size
    batch_size = args.batch_size

    PROMPT = """Question: {} 
    For this question, please refer to the given information and then perform step-by-step reasoning, to obtain the final answer. Note that the final answer should be formatted as:
    Reasoning Process: all thinking steps
    Final answer: \\boxed{{your short answer here}}
    """
    # Run the batch processing function
    output = process_images_in_batches(processor, model, clean_batch_data, batch_size, prompt=PROMPT, args=args)

    # save output into jsonl
    with open(os.path.join(args.output_dir, "{}_{}_{}.jsonl".format(
                args.model_name, args.model_type, args.split
                )), 'w') as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
