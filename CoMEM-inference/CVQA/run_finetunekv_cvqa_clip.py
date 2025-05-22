import argparse
from tqdm import tqdm
import time
import transformers
import sys
module_path = "CoMEM-train"
sys.path.append(module_path)
from src_vlm.training.qwenVL_inference import Qwen2_5_VLForConditionalGeneration_new
from src_vlm.training.qwenVL_inference2 import Qwen2VLForConditionalGeneration_new
sys.path.insert(0, "CoMEM-inference")
from src.load_model_test import *

from CVQA.prompt import PROMPT
import pandas as pd
from datasets import load_dataset
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
    parser.add_argument("--model_name", type=str, default="qwen2.5", help="model name")
    parser.add_argument("--model_type", type=str, default="CoMEM", help="baseline or steered")
    parser.add_argument("--output_dir", type=str, default="CoMEM-inference/CVQA/result", help="output directory")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--similar_num", type=int, default=10, help="number of similar samples")
    parser.add_argument("--checkpoint_path", type=str, default="", help="checkpoint path")
    args = parser.parse_args()
    # setup device to use
    device = torch.device(args.device)
    
    knowledge_base = load_mds("path of retirval database for CVQA here")
    knowledge_base = {item['ID']: item for item in knowledge_base}
    ds = load_dataset("afaji/cvqa")
    df = ds['test'].to_pandas()
    lang_set = ["('Chinese', 'China')", "('Spanish', 'Spain')", "('Russian', 'Russia')", "('Bulgarian', 'Bulgaria')", "('Portuguese', 'Brazil')"]
    filtered_df = df[df['Subset'].isin(lang_set)]
    filtered_df = filtered_df[filtered_df['ID'].isin(knowledge_base.keys())]

    print("Load pretrained model...")
    max_memory = { 
    0: "23GiB",
    1: "23GiB",
    }
    if args.model_name == "qwen2.5":
        checkpoint_path = checkpoint_path = args.checkpoint_path
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
    
    output = []
    # Load existing progress if available
    output_file_path = os.path.join(args.output_dir, "cvqa_{}_{}_{}.jsonl".format(
                args.model_name, args.model_type, args.similar_num))
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r') as f:
            for line in f:
                output.append(json.loads(line))
    else:
        output = []
    
    selected_batch_data = filtered_df.iloc[len(output):]
    print('start from ', len(output))
    for i in range(selected_batch_data.shape[0]):
        item = selected_batch_data.iloc[i]
        image = Image.open(BytesIO(item['image']['bytes'])).convert("RGB")
        if image.size[0] > 512 or image.size[1] > 512:
            image = image.resize((512, 512), Image.LANCZOS)    
        question = item['Question']
        for idx, option in enumerate(item['Options']):
            question += f"{idx+1}. {option} "
        prompt = PROMPT[item['Subset']].format(question)
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
        similar_infos = process_similar_infos(knowledge_base[item['ID']], args.similar_num)
        texts = [item['desc'] for item in similar_infos.values()]
        images = [item['image'] for item in similar_infos.values()]
        ans = generate_response_knowledge(args.model_name, processor, model, image, prompt, texts, images)
        torch.cuda.empty_cache()
        print(ans)
        output.append({"data_id": item['ID'],
                       "question": question,
                       "prediction": ans,
                       "answer": str(item['Label']),
                       "question_type": item['Category'],
                       "languaege": item['Subset'],
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