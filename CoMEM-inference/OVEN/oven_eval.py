"""OVEN Evaluation Script."""
import re
import json
import string
from typing import Any, Dict, Generator, List, Tuple, Union
from collections import defaultdict


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of Dictionary."""
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def prepare_qid2example(
    reference: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
    """Convert reference to qid2example dictionary."""
    qid2example = dict()
    for r in reference:
        qid = r['data_id']
        qid2example[qid] = r
    return qid2example

def evaluate_oven(ref: List[Dict[str, Any]], pred: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate the predicted results against the reference.

    :param ref: a list of dictionaries each representing a reference item.
    :param pred: a list of dictionaries each representing a predicted item.
    :return: a dictionary of accuracy for each split.
    """
    split2res = defaultdict(list)
    split2mismatches = defaultdict(list)
    qid2example = prepare_qid2example(ref)

    for pred_item in pred:
        data_id = pred_item["data_id"]
        ref_item = qid2example[data_id]
        ref_ent_id = ref_item["entity_id"]
        pred_ent_id = pred_item["pred_entity_id"]
        data_split = ref_item["data_split"]
        
        match_score = int(ref_ent_id == pred_ent_id)
        split2res[data_split].append(match_score)

        if ref_ent_id != pred_ent_id:
            mismatch_info = {
                "data_id": data_id,
                "ref_entity_id": ref_ent_id,
                "pred_entity_id": pred_ent_id,
                "prediction": pred_item.get("prediction", ""),  # Optional: save prediction text
                "reference": ref_item.get("text", "")  # Optional: save reference text
            }
            split2mismatches[data_split].append(mismatch_info)


    result = {}
    mismatch_result = {}
    for split, results in split2res.items():
        accuracy = round(sum(results) / len(results) * 100, 2)
        if "_seen" in split:
            result["seen"] = accuracy
            mismatch_result["seen"] = split2mismatches[split]
        elif "_unseen" in split:
            result["unseen"] = accuracy
            mismatch_result["unseen"] = split2mismatches[split]
    
    return result, mismatch_result

def harmonic_mean(*args: float) -> float:
    """Calculate the harmonic mean of the input arguments."""
    args_safe = [a if a != 0 else 1e-12 for a in args]
    hmean = len(args_safe) / sum((1.0 / val) for val in args_safe)
    return hmean

def validate_prediction_inputs(predictions: List[Dict[str, Any]]) -> None:
    """
    Validate that all required keys are present in the prediction inputs.

    :param predictions: a list of dictionaries each representing a predicted item.
    :raises ValueError: if a required key is missing from any prediction input.
    """
    for prediction in predictions:
        if "pred_text" not in prediction:
            raise ValueError(f"pred_text is missing in prediction data_id {prediction['data_id']}")


def evaluate_oven_full(ref_query: List[Dict[str, Any]], ref_entity: List[Dict[str, Any]], 
                  pred_query: List[Dict[str, Any]], pred_entity: List[Dict[str, Any]]) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Calculate the final result based on both query and entity results.

    :param ref_query: a list of dictionaries each representing a reference query.
    :param ref_entity: a list of dictionaries each representing a reference entity.
    :param pred_query: a list of dictionaries each representing a predicted query.
    :param pred_entity: a list of dictionaries each representing a predicted entity.
    :return: a dictionary containing calculated scores and results.
    """
    if pred_query is not None:
        # validate_prediction_inputs(pred_query)
        query_result, query_mismatch_result = evaluate_oven(ref_query, pred_query)
        try:
            query_score = harmonic_mean(query_result["seen"], query_result["unseen"])
        except:
            query_score = harmonic_mean(query_result["seen"], query_result["seen"])
        query_score = round(query_score, 2)
        # Optionally save mismatches to file
        for split, mismatches in query_mismatch_result.items():
            if mismatches:  # Only save if there are mismatches
                filename = f"query_mismatches_{split}.json"
                with open(filename, 'w') as f:
                    json.dump(mismatches, f, indent=2)
                print(f"Saved {len(mismatches)} mismatches for {split} split to {filename}")
    else:
        query_result = {"seen": None, 
                         "unseen": None}
        query_score = None 

    if pred_entity is not None:
        # validate_prediction_inputs(pred_entity)
        entity_result, entity_mismatch_result = evaluate_oven(ref_entity, pred_entity)
        try:
            entity_score = harmonic_mean(entity_result["seen"], entity_result["unseen"])
        except:
            entity_score = harmonic_mean(entity_result["seen"], entity_result["seen"])
        entity_score = round(entity_score, 2)
        # Optionally save mismatches to file
        for split, mismatches in entity_mismatch_result.items():
            if mismatches:  # Only save if there are mismatches
                filename = f"entity_mismatches_{split}.json"
                with open(filename, 'w') as f:
                    json.dump(mismatches, f, indent=2)
                print(f"Saved {len(mismatches)} mismatches for {split} split to {filename}")
    else:
        entity_result = {"seen": None, 
                         "unseen": None}
        entity_score = None
    
    if query_score is not None and entity_score is not None:
        final_score = harmonic_mean(query_score, entity_score)
        final_score = round(final_score, 2)
    else:
        final_score = None 

    final_result = {
        "query_score": query_score,
        "entity_score": entity_score,
        "final_score": final_score,
        "query_result": query_result,
        "entity_result": entity_result
    }
    return final_result