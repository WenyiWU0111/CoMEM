import re
import json
import string
from typing import Any, Dict, Generator, List, Tuple, Union
import ast
from datasets import load_dataset

def normalize_answer(text: str) -> str:
    """Normalize a given text by removing articles, punctuation, and white spaces, and converting to lowercase."""
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def split_sentence(text: str) -> str:
        text = text.split('is')[-1]
        text = text.split('the')[-1]
        text = text.split('of')[-1]
        text = text.split('by')[-1]
        return text
    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())

    def remove_punctuation(text: str) -> str:
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def lowercase(text: str) -> str:
        return text.lower()
    return white_space_fix(remove_articles(split_sentence(remove_punctuation(lowercase(text)))))
    
def extract_and_normalize_boxed_answer(latex_text: str) -> str:
    """
    Extract content from LaTeX \boxed{} command and normalize it.
    
    Args:
        latex_text: String potentially containing LaTeX \boxed{} command
        
    Returns:
        Normalized content of the boxed expression
    """
    # Extract content from \boxed{...}
    boxed_pattern = r'\\boxed\{(.*?)\}'
    match = re.search(boxed_pattern, latex_text)
    
    if match:
        # Extract the content inside \boxed{}
        boxed_content = match.group(1)
        # Normalize the extracted content
        normalized_answer = normalize_answer(boxed_content)
        
    else:
        content = latex_text.split(':')[-1]
        normalized_answer = normalize_answer(content)
    
    return normalized_answer

def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Check if the normalized prediction exactly matches the normalized ground truth."""
    prediction = extract_and_normalize_boxed_answer(prediction)
    return prediction == ground_truth

def run_evaluation(pred_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Evaluate the predictions against the ground truth.
    
    Args:
        pred_data: List of dictionaries containing prediction data
        
    Returns:
        List of dictionaries with evaluation results
    """    
    results = []
    for item in pred_data:
        try:
            data_id = item['data_id']
        except KeyError:
            data_id = item['question_id']
        prediction = item['prediction']
        if isinstance(prediction, list):
            prediction = prediction[0]
        try:
            ground_truth = item['answer']
        except KeyError:
            ground_truth = item['direct_answers']
        if isinstance(ground_truth, str):
            ground_truth = ast.literal_eval(ground_truth)
        ground_truth = [item.strip() for item in ground_truth if item.strip()]
        # print("ground_truth", ground_truth)
        # print("prediction", prediction)
        em_score = max(
            exact_match_score(prediction, gt) for gt in ground_truth
        )
        results.append({
            'data_id': data_id,
            'exact_match': em_score
        })

    all_em_score = [item['exact_match'] for item in results]
    overall_accuracy = sum(all_em_score) / len(all_em_score) * 100
    print(f"Overall Accuracy: {overall_accuracy:.2f}")

pred_path = ""
with open(pred_path, 'r') as f:
    pred_data = [json.loads(line) for line in f.readlines()]
run_evaluation(pred_data)