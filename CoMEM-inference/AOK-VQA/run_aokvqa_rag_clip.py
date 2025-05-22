"""Zeroshot Inference - OKVQA"""
import os
import json
import torch
from PIL import Image
from multiprocessing import Pool
import argparse
from tqdm import tqdm
import time
import transformers
import sys
from datasets import load_dataset
sys.path.insert(0, "CoMEM-inference")
from src.load_model_test import *
import pandas as pd
from io import BytesIO
import dask.dataframe as dd
from io import BytesIO
from streaming import StreamingDataset
import shutil
import base64 
from tqdm import tqdm

def load_mds(mds_dir):
    dataset = StreamingDataset(local=mds_dir,
                           remote=None,
                           shuffle=False,
                           batch_size=1)
    records = []
    for sample in tqdm(dataset, desc="Loading MDS files"):
        records.append(sample)
    return records

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mplug", help="model name")
    parser.add_argument("--model_type", type=str, default="rag", help="rag_direct or rag_direct_mcq")
    parser.add_argument("--output_dir", type=str, default="CoMEM-inference/AOK-VQA/result", help="output directory")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--similar_num", type=int, default=10, help="number of similar samples")
    args = parser.parse_args()

    test_ds = load_dataset("HuggingFaceM4/A-OKVQA")
    use_split = "validation"
    test_ds = test_ds[use_split]
    df = test_ds.to_pandas()
    
    retrive_kb = load_mds("path of retirval database for AOK-VQA here")
    retrive_kb = {item['question_id']: item for item in retrive_kb}
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
    output_file_path = os.path.join(args.output_dir, "okvqa_{}_{}_{}.jsonl".format(
                args.model_name, args.model_type, args.similar_num))
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
        ####### Find Similar Images #######
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
        k = retrive_kb[item['question_id']]
        similar_infos = process_similar_infos(k, args.similar_num)
        ans = generate_response_rag(args.model_name, processor, model, image, prompt, similar_infos, tokenizer)
        delimiters = ["[/INST]", "ASSISTANT:"]
        for delimiter in delimiters:
            if delimiter in ans:
                ans = ans.split(delimiter)[-1].strip()
        torch.cuda.empty_cache()
        print(ans)
        output.append({"data_id": item['question_id'],
                       "question": question,
                       "prediction": ans,
                       "answer": str(item['direct_answers'])
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