import numpy as np 
import pandas as pd 
import json 
import os
import io
import sys
import faiss
from collections import defaultdict
import clip
import torch
import requests  
from urllib.parse import quote  
from PIL import Image  
import random
import pickle
import wikipediaapi
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from langdetect import detect

def get_random_headers():
    # List of real browser User-Agents
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    ]
    
    # List of valid languages from Wikimedia Commons
    languages = ['en-US,en;q=0.9', 'en-GB,en;q=0.9', 'de-DE,de;q=0.9', 'fr-FR,fr;q=0.9', 'es-ES,es;q=0.9']
    
    # Different accept values
    accept_values = [
        'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
        'image/webp,image/apng,image/*,*/*;q=0.8',
        'image/avif,image/webp,image/apng,image/*,*/*;q=0.8'
    ]
    
    headers = {
        'User-Agent': random.choice(user_agents),
        'Accept': random.choice(accept_values),
        'Accept-Encoding': 'gzip, deflate, br',
        #'Accept-Language': random.choice(languages),
        'Referer': 'https://commons.wikimedia.org/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
    }
    
    return headers
def get_inference_embeddings(tsv_idxs, chunk_idxs):
    inference_embeddings = []
    inference_csvs = []
    
    # 1. First collect all file paths and indices
    print('1. First collect all file paths and indices')
    file_info = []
    for idx in tsv_idxs:
        for chunk_idx in chunk_idxs:
            # Prepare paths
            npy_path = f'/home/wit/clip_embeddings_wit_v1.train.all-0000{idx}-of-00010/embeddings_chunk_{chunk_idx}.npy'
            json_path = f'/home/wit/clip_embeddings_wit_v1.train.all-0000{idx}-of-00010/indices_chunk_{chunk_idx}.json'
            tsv_path = f'/home/wit/wit_v1.train.all-0000{idx}-of-00010.tsv'
            
            # Load embeddings
            inference_embeddings.append(np.load(npy_path, allow_pickle=True))
            
            # Load indices
            with open(json_path, 'r') as f:
                processed_indices = json.load(f)['processed_indices']
            #print(idx, chunk_idx, processed_indices)
            file_info.append({
                'tsv_path': tsv_path,
                'chunk_idx': chunk_idx,
                'processed_indices': processed_indices
            })
    
    # 2. Group operations by TSV file to reduce file openings
    print('2. Group operations by TSV file to reduce file openings')
    
    tsv_groups = defaultdict(list)
    for info in file_info:
        tsv_groups[info['tsv_path']].append((info['chunk_idx'], info['processed_indices']))
    
    # 3. Process each TSV file once
    print('3. Process each TSV file once')
    for tsv_path, chunk_info in tsv_groups.items():
        # Sort by chunk_idx to process in order
        chunk_info.sort(key=lambda x: x[0])
        
        for i, chunk in enumerate(pd.read_csv(tsv_path, sep='\t', chunksize=100000)):
            # Process all chunks that match current chunk index
            if i>max(chunk_idxs):
                break
            print(chunk.index)
            matching_chunks = [indices for chunk_idx, indices in chunk_info if chunk_idx == i]
            if matching_chunks:
                for indices in matching_chunks:
                    inference_csvs.append(chunk.iloc[indices])
    
    # 4. Combine results
    inference_csvs = pd.concat(inference_csvs, axis=0, ignore_index=True)
    inference_embeddings = np.vstack(inference_embeddings)
    
    print(inference_embeddings.shape)
    print(len(inference_csvs))
    return inference_embeddings, inference_csvs

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        encoded_url = quote(image_file, safe=':/?=&') 
        # More complete headers for Wikimedia
        headers = get_random_headers()
        session = requests.Session()
        response = session.get(encoded_url, headers=headers)
        if response.status_code != 200:
            # If first attempt fails, try again with different headers
            headers = get_random_headers()
            response = requests.get(image_file, headers=headers)
        #print(response)
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


# Function to preprocess and encode images
def get_image_embedding(image_path, preprocess, model, device):
    """Preprocess and get the CLIP embedding for an image."""
    image = preprocess(load_image(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image)
        embedding /= embedding.norm(dim=-1, keepdim=True)  # Normalize the embedding
    return embedding


def get_text_embedding(text, preprocess, model, device):
    """Preprocess and get the CLIP embedding for a text."""
    text = clip.tokenize(text).to(device)  # Tokenize the caption
    with torch.no_grad():
        text_embedding = model.encode_text(text)
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)  # Normalize the embedding
    return text_embedding

def save_embedding_mat(text, references, preprocess, model, save_path=None, save_name_path=None, type='clip'):
    if text:
        print("Computing embeddings for reference querys...")
        reference_embeddings = []
        reference_names = references
        for query in references:
            embedding = get_text_embedding(query, preprocess, model)
            reference_embeddings.append(embedding)
        # Stack reference embeddings into a single tensor (shape: [num_references, embedding_dim])
        reference_embeddings = torch.cat(reference_embeddings, dim=0) 
    else:
        print("Computing embeddings for reference images...")
        reference_embeddings = []
        reference_names = []
        for idx, image_path in enumerate(references):
            image_id = image_path.split('.')[0]
            #print(idx)
            try:
                if type=='clip':
                    embedding = get_image_embedding(image_path, preprocess, model)
                    
                reference_embeddings.append(embedding)
                reference_names.append(image_id)
            except:
                print(idx)
            if idx%100 == 0:
                print(f"Finished {idx+1} image embeddings calculation.")
        # Stack reference embeddings into a single tensor (shape: [num_references, embedding_dim])
        reference_embeddings = torch.cat(reference_embeddings, dim=0)
    if save_path is not None:
        np.save(save_path, reference_embeddings.cpu())
        with open(save_name_path, "wb") as f:  # Open in binary write mode
            pickle.dump(reference_names, f)  
    return reference_embeddings, reference_names

def get_similar_infos_parallel(similar_idx, inference_csvs, wiki_search=True, summarize=False):
    similar_infos = {}
    similar_rows = inference_csvs.iloc[similar_idx]
    
    # Use ThreadPoolExecutor to parallelize downloads
    with ThreadPoolExecutor(max_workers=20) as executor:  # Adjust workers as needed
        future_to_idx = {}
        for idx, row in similar_rows.iterrows():
            future = executor.submit(process_single_row, idx, row, wiki_search)
            future_to_idx[future] = idx
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                if result:
                    similar_infos[idx] = result
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
                
    return similar_infos

def process_single_row(idx, row, wiki_search):
    try:
        url = row['image_url']
        img = load_image(url)
        if img is not None:
            img = img.resize((512, 512), Image.Resampling.LANCZOS)
        else:
            return {
                'image': None,
                'desc': ''
            }
        info = get_entity_info(row, wiki_search=wiki_search)
        return {
            'image': img,
            'desc': f"{info['name']}: {info['desc']}"
        }
    except Exception as e:
        print(f"Error processing entire row {idx}: {e}")
        return None
    
def get_entity_info(row, wiki_search):
    """
    Create entity info from row columns
    """
    # Use page_title as the primary entity name
    name = row['page_title']
    context_description = ''
    # Combine relevant descriptions, filtering out None/NaN values
    if row['context_section_description'] and row['context_section_description'] != '':
        context_description = row['context_section_description']
    elif row['context_page_description'] and row['context_page_description'] != '':
        context_description = row['context_page_description']
    
    if wiki_search:
        language = row['language']
        if language != 'en':
            try:
                language = detect(name)
            except:
                language = 'en'
        
        wiki = wikipediaapi.Wikipedia(user_agent='memory-vector', language=language)
        page = wiki.page(name)
        if page.exists() and page.summary != "":
            description = page.summary
        else:
            description = context_description
    else:
        description = context_description
        
    return {
        'name': name,
        'desc': description
    }
