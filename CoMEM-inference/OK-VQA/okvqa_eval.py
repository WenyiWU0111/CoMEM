import re
import json
import string

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

pred_path = ""
with open(pred_path, 'r') as f:
    pred_data = [json.loads(line) for line in f.readlines()]
    
results = []
for item in pred_data:
    data_id = item['data_id']
    prediction = item['prediction']
    ground_truth = item['answer']
    ground_truth = ground_truth.replace("[", '').replace("]", "").split("'")
    ground_truth = [item.strip() for item in ground_truth if item.strip()]
    em_score = max(
        exact_match_score(prediction, gt) for gt in ground_truth
    )
    results.append({
        'data_id': data_id,
        'question_type': item['question_type'],
        'exact_match': em_score
    })
    
output_name = pred_path.split("/")[-1].split(".")[0]    
output_path = f"CoMEM-inference/OK-VQA/result/eval/{output_name}.jsonl"
with open(output_path, 'w') as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

all_question_types = set([item['question_type'] for item in results])
question_type_accuracy = {}
for question_type in all_question_types:
    type_em_score = [item['exact_match'] for item in results if item['question_type'] == question_type]
    type_accuracy = sum(type_em_score) / len(type_em_score) * 100
    question_type_accuracy[question_type] = type_accuracy
    print(f"Accuracy for {question_type}: {type_accuracy:.2f}")
all_em_score = [item['exact_match'] for item in results]
overall_accuracy = sum(all_em_score) / len(all_em_score) * 100
print(f"Overall Accuracy: {overall_accuracy:.2f}")