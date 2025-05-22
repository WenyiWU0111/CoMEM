"""Zeroshot Inference - OKVQA"""
import os
import json
import torch
from PIL import Image
import argparse
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.load_model_test import *
import pandas as pd
from io import BytesIO
from io import BytesIO
from streaming import StreamingDataset
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
    parser.add_argument("--model_name", type=str, default="qwen2.5", help="model name")
    parser.add_argument("--model_type", type=str, default="prefixkv_clip", help="baseline or steered")
    parser.add_argument("--output_dir", type=str, default="CoMEM-inference/Viquae/result", help="output directory")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--similar_num", type=int, default=10, help="number of similar samples")
    parser.add_argument("--prefix_idx", type=list, default=[17,18,19], help="prefix index of cache KV")
    parser.add_argument("--top-tokens", type=int, default=25, help="whether limit top tokens")
    args = parser.parse_args()

    splits = {'train': 'train.jsonl', 'validation': 'validation.jsonl', 'test': 'test.jsonl'}
    df = pd.read_json("hf://datasets/PaulLerner/viquae_dataset/" + splits["test"], lines=True)
    
    retrive_kb = load_mds("path of retirval database for ViQUAE here")
    retrive_kb = {item['id']: item for item in retrive_kb}
    # setup device to use
    device = 'cuda'
    PROMPT = """Question: {} 
    For this question, please perform step-by-step reasoning, to obtain the final answer. Your final answer should be short and concise.
    Note that the final answer should be formatted as:
    Reasoning Process: all thinking steps
    Final answer: \\boxed{{your short answer here}}"""
    max_memory = { 
        0: "23GiB",
        1: "23GiB",
        # 2: "10GiB",
        # 3: "10GiB",
        # 4: "23GiB",
        # 5: "23GiB",
        }
    print("Load pretrained model...")
    model_path_map = {
        'qwen2': "Qwen/Qwen2-VL-7B-Instruct",
        'qwen2.5': "Qwen/Qwen2.5-VL-7B-Instruct",
    }
    model_path = model_path_map.get(args.model_name, None)
    processor, tokenizer, model = load_model(args.model_name, model_path)
    
    output = []
    # Load existing progress if available
    output_file_path = os.path.join(args.output_dir, "{}_{}_{}_{}_{}.jsonl".format(
                args.model_name, args.model_type,  args.prefix_idx, args.similar_num, args.top_tokens
                ))
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
        img_path = os.path.join("path of image folder", item["image"])
        img = Image.open(img_path).convert("RGB")
        img = img.resize((512, 512))    
        question = item['input']
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
        k = retrive_kb[item['id']]
        similar_infos = process_similar_infos(k, args.similar_num)
        prefix_kvs = [get_past_key_value_text(processor, tokenizer, model, value['image'], value['desc'], args.prefix_idx, args.top_tokens, prompt) for value in similar_infos.values()]
        prefix_kvs = [item for item in prefix_kvs if item is not None]
        if prefix_kvs!=[]:
            prefix_kv = concatenate_past_key_values(prefix_kvs, args.prefix_idx)
            # move prefix_kv to different devices
            prefix_kv = move_prefix_kv_to_model_device(prefix_kv, model, args.model_name)
            ans, past_kv = generate_response_with_kv(
            args.model_name,    
            processor, 
            model, 
            image=img, 
            prompt=prompt,
            prefix_kv=prefix_kv)
        else:
            ans, past_kv = generate_response(args.model_name, processor, model, img, prompt)
        delimiters = ["[/INST]", "ASSISTANT:"]
        for delimiter in delimiters:
            if delimiter in ans:
                ans = ans.split(delimiter)[-1].strip()
        torch.cuda.empty_cache()
        print(ans)
        output.append({"data_id": item['id'],
                       "question": question,
                       "prediction": ans,
                       "answer": item['output']['answer']
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
    