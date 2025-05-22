"""zeroshot Infoseek inference script."""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
from PIL import Image
import argparse
import time
import pickle
from src.load_model_test import *


def load_and_process_image(item):
    # Load and preprocess the image
    path = item["image_path"]
    raw_image = Image.open(path).convert("RGB")     
    if raw_image.size[0] > 512 or raw_image.size[1] > 512:
        raw_image = raw_image.resize((512, 512), Image.LANCZOS)       
    return raw_image, item["question"]

def process_images_in_batches(batch_data, question_ids, batch_size, prompt, args):
    ########## Get output saving path ###########
    file_path = os.path.join(args.output_dir, "{}_{}_{}_{}_{}_{}_{}.jsonl".format(
                    args.model_name, args.model_type, args.split, args.prefix_idx, args.similar_num, args.top_tokens, args.pool
                    ))
    # Check if the file exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            output = [json.loads(line) for line in f]
    else:
        output = []
    batch_data = batch_data[len(output):]
    question_ids = question_ids[len(output):]
    # setup device to use
    model_path_map = {
        'qwen2': "Qwen/Qwen2-VL-7B-Instruct",
        'qwen2.5': "Qwen/Qwen2.5-VL-7B-Instruct",
    }
    model_path = model_path_map.get(args.model_name, None)
    processor, tokenizer, model = load_model(args.model_name, model_path)

    print("Generate predictions...")
    # Process images in batches
    for idx, i in enumerate(range(0, len(batch_data), batch_size)):
        if (idx + 1) % 100 == 0:
            print(f"Processing batch {idx}/{len(batch_data)/batch_size}")
        # Subset results for the current batch
        batch_subset = batch_data[i:i+batch_size]
        question_ids_subset = question_ids[i:i+batch_size]

        # Separate the images, questions, and ids
        batch_ids, answers = [], []

        # Load and preprocess the images
        start_time = time.time()
        for tmp_id, item in zip(question_ids_subset, batch_subset):
            tmp_img, tmp_q = load_and_process_image(item)
            batch_ids.append(tmp_id)
            format_tmp_q = prompt.format(tmp_q)
            ####### Find Similar Images #######
            def process_similar_infos(item, similar_num):
                similar_infos = item["retrieval_info"][:similar_num]
                similar_infos_dict = {}
                for idx, info in enumerate(similar_infos):
                    key = idx
                    fact_img = info['image']
                    if fact_img.size[0] > 512 or fact_img.size[1] > 512:
                        fact_img = fact_img.resize((512, 512), Image.LANCZOS) 
                    fact_text = info["passage_content"]
                    similar_infos_dict[key] = {"image": fact_img, "desc": fact_text}
                return similar_infos_dict
            similar_infos = process_similar_infos(item, args.similar_num)
            prefix_kvs = [get_past_key_value_text(processor, tokenizer, model, value['image'], value['desc'], args.prefix_idx, args.top_tokens, tmp_q, pool=args.pool) for value in similar_infos.values()]
            prefix_kvs = [item for item in prefix_kvs if item is not None]
            if prefix_kvs!=[]:
                prefix_kv = concatenate_past_key_values(prefix_kvs, args.prefix_idx)
                # move prefix_kv to different devices
                prefix_kv = move_prefix_kv_to_model_device(prefix_kv, model, args.model_name)
                ans, past_kv = generate_response_with_kv(
                args.model_name,    
                processor, 
                model, 
                image=tmp_img, 
                prompt=format_tmp_q,
                prefix_kv=prefix_kv)
            else:
                ans, past_kv = generate_response(args.model_name, processor, model, tmp_img, tmp_q)
            delimiters = ["[/INST]", "ASSISTANT:"]
            for delimiter in delimiters:
                if delimiter in ans:
                    ans = ans.split(delimiter)[-1].strip()
            print(ans)
            answers.append(ans)

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
    parser.add_argument("--split", type=str, default="val", help="val, test, or human")
    parser.add_argument("--model_name", type=str, default="qwen2_clip", help="blip2_t5 | blip2_vicuna_instruct | blip2_t5_instruct")
    parser.add_argument("--model_type", type=str, default="prefix_kv", help="pretrain_flant5xxl | vicuna13b | flant5xxl")
    parser.add_argument("--output_dir", type=str, default="CoMEM-inference/infoseek/result", help="output directory")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--similar_num", type=int, default=10, help="number of similar samples")
    parser.add_argument("--prefix_idx", type=list, default=[17,18,19], help="prefix index of cache KV")
    parser.add_argument("--top-tokens", type=int, default=25, help="whether limit top tokens")
    parser.add_argument("--pool", type=bool, default=True, help="whether use pool")
    args = parser.parse_args()

    split2data = {
        "val": "CoMEM-inference/infoseek/Infoseek_test_full.pkl"
    }

    # Read the input JSONL file
    print('Read the input JSONL file')
    with open(split2data[args.split], 'rb') as f:
        batch_data = pickle.load(f)

    # double check data exists:
    not_exist = []
    clean_batch_data = []
    clean_question_ids = []
    for idx, (qid, item)in enumerate(batch_data.items()):
        if idx % 10000 == 0:
            print(f"Processing {idx}/{len(batch_data)}")
        path = item['image_path']
        # check path exists
        if not os.path.exists(path):
            not_exist.append(qid)
        else:
            clean_batch_data.append(item)
            clean_question_ids.append(qid)
    print(len(not_exist))
    # setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # Desired batch size
    batch_size = args.batch_size

    PROMPT = PROMPT = """Question: {} 
    For this question, please refer to the given information and then perform step-by-step reasoning, to obtain the final answer. Note that the final answer should be formatted as:
    Reasoning Process: all thinking steps
    Final answer: \\boxed{{your short answer here}}
    """
    
    # Run the batch processing function
    output = process_images_in_batches(clean_batch_data, clean_question_ids, batch_size, prompt=PROMPT, args=args)

    # save output into jsonl
    with open(os.path.join(args.output_dir, "{}_{}_{}_{}_{}_{}.jsonl".format(
                args.model_name, args.model_type, args.split, args.prefix_idx, args.similar_num, args.top_tokens
                )), 'w') as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")