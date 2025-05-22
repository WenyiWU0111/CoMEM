""" Infoseek Validation Set Evaluation script."""
from infoseek_eval import evaluate

split2data = {
        "val": "CoMEM-inference/infoseek/val_dataset/infoseek_val.jsonl",
        "spanish": "CoMEM-inference/infoseek/val_dataset/infoseek_val_spanish.jsonl",
        "portuguese": "CoMEM-inference/infoseek/val_dataset/infoseek_val_portuguese.jsonl",
        "chinese": "CoMEM-inference/infoseek/val_dataset/infoseek_val_chinese.jsonl",
        "russian": "CoMEM-inference/infoseek/val_dataset/infoseek_val_russian.jsonl",
        "bulgarian": "CoMEM-inference/infoseek/val_dataset/infoseek_val_bulgarian.jsonl",
    }
if __name__ == "__main__":
        pred_path = ""
        reference_path = split2data['val']
        reference_qtype_path = f"infoseek_val_qtype.jsonl"

        result, results_by_id = evaluate(pred_path, reference_path, reference_qtype_path)
        final_score = result["final_score"]
        unseen_question_score = result["unseen_question_score"]["score"]
        unseen_entity_score = result["unseen_entity_score"]["score"]
        print(f"Validation final score: {final_score}")
        print(f"Validation unseen question score: {unseen_question_score}")
        print(f"Validation unseen entity score: {unseen_entity_score}")