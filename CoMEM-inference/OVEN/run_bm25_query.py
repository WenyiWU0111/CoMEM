import argparse
import json
import time
import pickle
from multiprocessing import Pool
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import re
import string
from typing import Any, Dict, Generator, List, Tuple, Union


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

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

def run_query(query):
    prediction = extract_and_normalize_boxed_answer(query["prediction"])[:50]
    tokenized_query = tokenizer.tokenize(prediction)
    results = bm25.get_top_n(tokenized_query, corpus, n=5)
    return {"data_id": query["data_id"], "prediction": prediction, "bm25": results}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="")
    parser.add_argument("--output_file", type=str, default="")
    args = parser.parse_args()

    corpus = load_json("CoMEM-inference/OVEN/Wiki6M_ver_1_0_title_only.jsonl")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    corpus = [
        [d["wikipedia_title"], d["wikidata_id"]] for d in corpus
    ]

    print("Loading bm25 pickle...")
    with open("CoMEM-inference/OVEN/wikipedia6m_index.pkl", "rb") as f:
        bm25 = pickle.load(f)

    input_query = load_json(args.input_file)

    print("Running bm25 query...")
    with Pool() as p:
        max_ = len(input_query)
        with tqdm(total=max_) as pbar:
            for i, _ in tqdm(enumerate(p.imap_unordered(run_query, input_query))):
                pbar.update()
        output = list(p.imap_unordered(run_query, input_query))

    with open(args.output_file, "w", encoding="utf-8") as f:
        for d in output:
            f.write(json.dumps(d) + "\n")