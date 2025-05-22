import os
import json
import math
import io
from PIL import Image
import io    
from tqdm.auto import tqdm
import pandas as pd 
import ast
import sys
sys.path.insert(0, "CoMEM-inference")
from src.load_model_test import *

def bench_data_loader(args, image_placeholder="<image>"):
    """ 
    Data loader for benchmarking models
    Args:
        args: arguments
        image_placeholder: placeholder string for image
    Returns:
        generator: a generator that yields data (queries, image paths, ...) for each sample
    """
    # Data
    final_df=pd.read_csv('CoMEM-inference/MRAG_Bench/questions.csv')
        
    for index, item in tqdm(final_df.iterrows(), total=len(final_df)):
        qs_id = item['id'] 
        qs = item['question']
        ans = item['answer']
        gt_choice = item['answer_choice']
        scenario = item['scenario']
        choices_A = item['A']
        choices_B = item['B']
        choices_C = item['C']
        choices_D = item['D']
        image = ast.literal_eval(item['image'])['bytes']
        image = Image.open(io.BytesIO(image)).convert("RGB") 
        height = image.size[0]
        while height > 100:
                image = pool_image(image, pooling_rounds=1)
                height = image.size[0]
        
        ### our evaluation instuction for all the models 
        if not args.use_rag: 
            prompt = f"Answer according to the image with the option's letter from the given choices directly.\n"
        else: 
            prompt = f"For this question, please perform step-by-step reasoning, to obtain the final answer. Note that the final answer should only be the option's letter.\n"
        
        qs += f"\n Choices:\nA: {choices_A}\nB: {choices_B}\nC: {choices_C}\nD: {choices_D}"
        prompt_question_part = qs
        prompt_instruction_part = prompt
        qs = prompt + qs
    
        cur_prompt = args.extra_prompt + qs

        yield {
            "id": qs_id, 
            "question": qs, 
            "image_files": image, 
            "prompt": cur_prompt,
            "answer": ans,
            "gt_choice": gt_choice,
            "scenario": scenario,
            "prompt_question_part": prompt_question_part,
            "prompt_instruction_part": prompt_instruction_part,
            "aspect": item['aspect']
        }