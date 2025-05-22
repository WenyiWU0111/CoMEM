"""LLAVA zeroshot OVEN inference script."""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
from PIL import Image
import argparse
import time
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
    question_ids = question_ids[len(output):]
    print('start from ', len(output))
    # setup device to use
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
            tmp_q = prompt.format(tmp_q)
            ans, past_kv = generate_response(args.model_name, processor, model, tmp_img, tmp_q)
            delimiters = ["[/INST]", "ASSISTANT:"]
            for delimiter in delimiters:
                if delimiter in ans:
                    ans = ans.split(delimiter)[-1].strip()
            torch.cuda.empty_cache()
            print(ans)
            answers.append(ans)

        print(f"Time for batch {idx}: {time.time() - start_time}")
        for idx, ans in zip(batch_ids, answers):
            output.append({"data_id": idx, "prediction": ans})
        # save output into jsonl
        with open(os.path.join(args.output_dir, "{}_{}_{}.jsonl".format(
                    args.model_name, args.model_type, args.split
                    )), 'w') as f:
            for item in output:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
       
    return output

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val", help="val, test, or human")
    parser.add_argument("--model_name", type=str, default="qwen2.5", help="blip2_t5 | blip2_vicuna_instruct | blip2_t5_instruct")
    parser.add_argument("--model_type", type=str, default="orimodel", help="pretrain_flant5xxl | vicuna13b | flant5xxl")
    parser.add_argument("--output_dir", type=str, default="CoMEM-inference/infoseek/result", help="output directory")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")

    args = parser.parse_args()

    split2data = {
        "val": "CoMEM-inference/infoseek/val_dataset/infoseek_val.jsonl",
        "spanish": "CoMEM-inference/infoseek/val_dataset/infoseek_val_spanish.jsonl",
        "portuguese": "CoMEM-inference/infoseek/val_dataset/infoseek_val_portuguese.jsonl",
        "chinese": "CoMEM-inference/infoseek/val_dataset/infoseek_val_chinese.jsonl",
        "russian": "CoMEM-inference/infoseek/val_dataset/infoseek_val_russian.jsonl",
        "bulgarian": "CoMEM-inference/infoseek/val_dataset/infoseek_val_bulgarian.jsonl",
    }

    with open(split2data['val'], 'r') as f:
        batch_data = json.load(f)
    # Read the input JSONL file
    print('Read the input JSONL file')
    if 'jsonl' in split2data[args.split]:
        with open(split2data[args.split], 'r') as f:
            lang_batch_data = [json.loads(line) for line in f]
            lang_batch_data = {item['data_id']: item for item in lang_batch_data}
        for key, item in batch_data.items():
            if key in lang_batch_data:
                item['question'] = lang_batch_data[key]['question']
                batch_data[key] = item

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

    templa_dict = {
        "val": """Question: {} 
    For this question, please perform step-by-step reasoning, to obtain the final answer. Note that the final answer should be formatted as:
    Reasoning Process: all thinking steps
    Final answer: \\boxed{{your short answer here}}""",
    "chinese": """问题: {} 
请对这个问题进行逐步推理，以得出最终答案。
请注意，最终答案应采用以下格式：
推理过程: 所有思考步骤
最终答案: \\boxed{{你的简短答案}}""",
"spanish": """Pregunta: {} 
Para esta pregunta, por favor realiza un razonamiento paso a paso para llegar a la respuesta final.
Ten en cuenta que la respuesta final debe tener el siguiente formato:
Proceso de razonamiento: todos los pasos del pensamiento
Respuesta final: \\boxed{{tu respuesta breve aquí}}""",
"russian": """Вопрос: {} 
Пожалуйста, выполните пошаговое рассуждение, чтобы получить окончательный ответ.
Обратите внимание, что окончательный ответ должен быть в следующем формате:
Ход рассуждений: все шаги размышлений
Окончательный ответ: \\boxed{{ваш краткий ответ здесь}}""",
"portuguese": """Pergunta: {}  
Para esta pergunta, por favor, realize um raciocínio passo a passo para chegar à resposta final.  
Note que a resposta final deve estar formatada da seguinte forma:  
Processo de raciocínio: todos os passos do pensamento  
Resposta final: \\boxed{{sua resposta curta aqui}}""",
"bulgarian": """Въпрос: {}
Моля, извършете поетапно разсъждение, за да достигнете до крайния отговор. Обърнете внимание, че крайният отговор трябва да бъде във формат:
Процес на разсъждение: всички мисловни стъпки  
Краен отговор: \\boxed{{вашият кратък отговор тук}}""",
    }
    
    PROMPT = templa_dict[args.split]
    
    # Run the batch processing function
    output = process_images_in_batches(clean_batch_data, clean_question_ids, batch_size, prompt=PROMPT, args=args)

    # save output into jsonl
    with open(os.path.join(args.output_dir, "{}_{}_{}.jsonl".format(
                args.model_name, args.model_type, args.split
                )), 'w') as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")