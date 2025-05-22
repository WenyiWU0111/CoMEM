import re
import json
import string
from typing import Any, Dict, Generator, List, Tuple, Union

from openai import OpenAI
from pydantic import BaseModel
import os

def exact_match_score(prediction: str, ground_truth: str, question=None) -> bool:
    """Check if the normalized prediction exactly matches the normalized ground truth."""
    boxed_pattern = r'\\boxed\{(.*?)\}'
    match = re.search(boxed_pattern, prediction)
    if match:
        prediction = match.group(1)
    else:
        prediction = prediction.split(':')[-1]

    return prediction == ground_truth

pred_path = ""
with open(pred_path, 'r') as f:
    pred_data = [json.loads(line) for line in f.readlines()]
    
results = []
for item in pred_data:
    data_id = item['data_id']
    question = item['question']
    prediction = item['prediction']
    ground_truth = str(int(item['answer'])+1)
    
    # Calculate exact match score
    em_score = exact_match_score(prediction, ground_truth, question)
    
    results.append({
        'data_id': data_id,
        'language': item['languaege'],
        'exact_match': em_score
    })
    
output_name = pred_path.split("/")[-1].split(".")[0]    
output_path = f"CoMEM-inference/CVQA/result/eval/{output_name}.jsonl"
with open(output_path, 'w') as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
all_languages = set()
for item in results:
    all_languages.add(item['language'])
for lang in all_languages:
    lang_results = [item for item in results if item['language'] == lang]
    em_scores = [item['exact_match'] for item in lang_results]
    avg_em_score = (sum(em_scores) / len(em_scores))*100
    print(f"Language: {lang}, Average Exact Match Score: {avg_em_score:.2f}")
all_em_scores = [item['exact_match'] for item in results]
avg_all_em_score = (sum(all_em_scores) / len(all_em_scores))*100
print(f"Average Exact Match Score for all languages: {avg_all_em_score:.2f}")
