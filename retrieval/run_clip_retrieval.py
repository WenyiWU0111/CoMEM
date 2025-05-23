import os
import json
import torch
from PIL import Image
from multiprocessing import Pool
import argparse
from tqdm import tqdm
import time
import transformers
import faiss
import numpy as np
from utilities import *
import os
import traceback
from datasets import load_dataset, load_from_disk
from datasets import DatasetDict
from streaming import MDSWriter, StreamingDataset
import shutil
import base64 
from io import BytesIO
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def load_mds(mds_dir):
    dataset = StreamingDataset(local=mds_dir,
                           remote=None,
                           shuffle=False,
                           batch_size=1)
    records = []
    for sample in tqdm(dataset, desc="Loading MDS files"):
        # Decode the base64-encoded knowledge images back to bytes, if needed.
        sample["knowledge_image"] = [
            base64.b64decode(img_str)
            for img_str in sample["knowledge_image"]
        ]
        records.append(sample)
    return records

        
def image_to_bytes(img):
    if isinstance(img, str):  # file path
        img = Image.open(img).convert("RGB")
        img = img.resize((512, 512))
    buf = BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()

def batch_search(index, query_embedding, k=5, batch_size=1024):
    """
    Perform batched search to avoid GPU memory issues
    """
    n = query_embedding.shape[0]
    D, I = [], []
    
    for start_idx in range(0, n, batch_size):
        print(f'{start_idx}/{batch_size}')
        end_idx = min(start_idx + batch_size, n)
        batch = query_embedding[start_idx:end_idx]
        
        # Ensure float32
        if batch.dtype != np.float32:
            batch = batch.astype(np.float32)
        
        batch_D, batch_I = index.search(batch, k)
        D.append(batch_D)
        I.append(batch_I)
    
    return np.vstack(D), np.vstack(I)
    
    
if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_split", type=str, default="test", help="val, test, or human")
    parser.add_argument("--model_type", type=str, default="retreive", help="baseline or steered")

    parser.add_argument("--batch_size", type=int, default=10, help="batch size")

    # parser.add_argument("--max_num", type=int, default=150, help="max num of dataset")
    parser.add_argument("--similar_num", type=int, default=10, help="number of similar images")
    parser.add_argument("--query_dataset", type=str, default="Infoseek", help="query dataset")
    parser.add_argument("--query_dataset_hf_path", type=str, default="", help="query dataset hf path")
    parser.add_argument("--query_img_embedding_path", type=str, default="/home/dataset/infoseek/infoseek_val_img_embeddings.npy", help="query image embedding path")
    parser.add_argument("--query_img_ids_path", type=str, default="/home/dataset/infoseek/infoseek_val_img_names.npy", help="query image ids path")
    parser.add_argument("--retrieval_dir", type=str, default="/home/dataset/infoseek/infoseek_finetuning/clip", help="retrieval path")
    
    
    args = parser.parse_args()

    test_ds = load_dataset(args.query_dataset_hf_path, args.query_dataset + "_data")

    use_split = args.use_split
    test_ds = test_ds[use_split]

    id2path = dict()

    # load image paths: Prepare a jsonl file to map image_id to image_path
    with open("id2image.jsonl", "r") as f:
        for line in f:
            line = json.loads(line)
            image_id = line["image_id"]
            path = line["image_path"]
            id2path[image_id] = path
    
    
    tsv_idxs = [2, 4, 6, 8]
    chunk_idxs = [0, 1, 2, 3, 4]
    inference_embedding, inference_csvs = get_inference_embeddings(tsv_idxs, chunk_idxs)
    inference_embedding = inference_embedding.astype(np.float32)
    query_embedding = np.load(args.query_img_embedding_path, allow_pickle=True)
    query_img_ids = np.load(args.query_img_ids_path, allow_pickle=True)
    d = inference_embedding.shape[1]
    
    # Initialize FAISS index on GPU
    res = faiss.StandardGpuResources()  # use a single GPU
    index_flat = faiss.IndexFlatL2(d)  # build the index
    index = faiss.index_cpu_to_gpu(res, 3, index_flat)  # move index to GPU
   
    # index = faiss.IndexFlatL2(d)  
    index.add(inference_embedding)
    print("Index built")
 
    D, I = batch_search(index, query_embedding, k=20, batch_size=1024)
    # D, I = index.search(query_embedding, args.similar_num)  # perform search on GPU
    
    
    print("Search done")
    columns = {
        "data_id": "str",
        "question": "str",
        "image_id": "str",
        "answer_eval": "json",
        "retrieval_info": "json",
    }
    output_dir = os.path.join(args.retrieval_dir, "Infoseek_eval_full")

    shutil.rmtree(output_dir, ignore_errors=True)
    records = []

    with MDSWriter(out=output_dir, columns=columns, compression=None) as out:
        for data in tqdm(test_ds, desc="Processing batches"):
            qid = str(data["question_id"])
            converted = {
                "data_id": data["question_id"],
                "question": data["question"],
                "image_id": data["image_id"],
                "answer_eval": data["answers"],
                "retrieval_info": [],
            }
            
            img_id = data["image_id"]
            query_idx = query_img_ids.index(img_id)
            similar_idx = I[query_idx]
            similar_infos = get_similar_infos_parallel(similar_idx, inference_csvs, wiki_search=False, summarize=False)
    
            for _, passage_dict in similar_infos.items():
                if len(converted["retrieval_info"]) == args.similar_num:
                    break
                if passage_dict["image"] != None:
                
                    converted["retrieval_info"].append({
                        "passage_content": passage_dict["desc"],
                        "image": base64.b64encode(image_to_bytes(passage_dict["image"])).decode("utf-8"),
                    })
            if len(converted["retrieval_info"]) < args.similar_num:
                # drop qid
                # records.pop(qid)
                print(f"Drop {qid} because of not enough retrieval results")
                continue
            else:
                out.write(converted)
                
        
    print(f"Save retrieval results to {output_dir}")