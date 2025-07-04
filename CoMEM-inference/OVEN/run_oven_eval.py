import os
from oven_eval import evaluate_oven_full, load_jsonl

ref_path = "oven"
pred_path = "predictions"
split2data = {
        "val_entity": "CoMEM-inference/OVEN/oven_entity_val.jsonl",
        "val_query": "CoMEM-inference/OVEN/oven_query_val.jsonl",
    }
ref_query_val = load_jsonl(split2data["val_query"])
ref_entity_val = load_jsonl(split2data["val_entity"])

pred_query_val = load_jsonl("")
pred_entity_val = load_jsonl("")

# Note - prediction file format:
# {"data_id": "", "pred_entity_id": ""}
# pred_entity_id is the entity_id from Wikidata (e.g., Q31 is Belgium)

pred_query_val_formal = [{"data_id": line["data_id"], "pred_entity_id": line["bm25"][0][1]} for line in pred_query_val]
pred_entity_val_formal = [{"data_id": line["data_id"], "pred_entity_id": line["bm25"][0][1]} for line in pred_entity_val]
#pred_entity_val_formal = None
llava_zeroshot_oven_val = evaluate_oven_full(ref_query_val, ref_entity_val, pred_query_val_formal, pred_entity_val_formal)
print("===== Qwen2.5 Zeroshot ====")
print("===== Validation ========")
print("===== Final score {}".format(llava_zeroshot_oven_val["final_score"]))
print("===== Query Split score {}".format(llava_zeroshot_oven_val["query_score"]))
print("===== Entity Split score {}".format(llava_zeroshot_oven_val["entity_score"]))
print("===== Query Seen Accuracy {}".format(llava_zeroshot_oven_val["query_result"]["seen"]))
print("===== Query Unseen Accuracy {}".format(llava_zeroshot_oven_val["query_result"]["unseen"]))
print("===== Entity Seen Accuracy {}".format(llava_zeroshot_oven_val["entity_result"]["seen"]))
print("===== Entity Unseen Accuracy {}".format(llava_zeroshot_oven_val["entity_result"]["unseen"]))
"""
===== BLIP2 Zeroshot ====
===== Validation ========
===== Final score 7.87
===== Query Split score 20.58
===== Entity Split score 4.87
===== Query Seen Accuracy 24.63
===== Query Seen Accuracy 17.68
===== Entity Seen Accuracy 8.55
===== Entity Seen Accuracy 3.4
"""