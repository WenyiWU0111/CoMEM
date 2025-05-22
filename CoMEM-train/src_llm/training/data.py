import copy
import os
from dataclasses import dataclass, field
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info
from PIL import Image
import re
import pickle
from PIL import Image

from .params import DataArguments
from .constants import *

from io import BytesIO
from PIL import Image
from tqdm import tqdm
from streaming import MDSWriter, StreamingDataset
import shutil
import base64 
import uuid
import subprocess

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

def truncate_sequence(input_ids, labels, max_length, eos_token_id):
    if input_ids.size(0) > max_length:
        input_ids = input_ids[:max_length-1]
        labels = labels[:max_length-1]

    if eos_token_id is not None:
        input_ids = torch.cat([input_ids, torch.tensor([eos_token_id])])
        labels = torch.cat([labels, torch.tensor([eos_token_id])])

    return input_ids, labels

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

def get_image_info(image_path, min_pixel, max_pixel):
    # Using this because of process_vision_info function
    # Need to fix this in the future    
    
    messages = [
        {"role": "user", 
         "content": [
             {
                "type": "image", 
                "image": image_path,
                "min_pixel": min_pixel,
                "max_pixel": max_pixel

            }
            ]
        }
    ]

    image_input, _ = process_vision_info(messages)

    return image_input[0]

def get_video_info(video_path, min_pixels, max_pixels, fps):
    # Using this because of process_vision_info function
    # Need to fix this in the future

    messages = [
        {"role": "user", 
         "content": [
             {
                "type": "video", 
                "video": video_path,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "fps": fps
            }
            ]
        }
    ]

    _, video_input, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    return video_input[0], video_kwargs

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(SupervisedDataset, self).__init__()
        if isinstance(data_path, str) and 'pkl' in data_path:
            with open(data_path, "rb") as file:
                list_data_dict = pickle.load(file)
        elif isinstance(data_path, str) and 'json' in data_path:
            with open(data_path, "r") as file:
                list_data_dict = json.load(file)
        elif isinstance(data_path, str):
            list_data_dict = load_mds(data_path)
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.fps = data_args.fps

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        is_video = False

        processor = self.processor
        if "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            knowledge_grid_key = "knowledge_image_grid_thw"
            knowledge_pixel_key = "knowledge_pixel_values"
            # Prepare Query Images
            image_files = sources["image"]
            image_folder = self.data_args.image_folder
            if isinstance(image_files, str) or isinstance(image_files, Image.Image) or isinstance(image_files, bytes):
                image_files = [image_files]

            images = []
            for image_file in image_files:
                if isinstance(image_file, Image.Image):
                    image_file = image_file
                elif isinstance(image_file, bytes):
                    image_file = Image.open(BytesIO(image_file))
                elif not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                    else:
                        image_file = image_file
                images.append(get_image_info(image_file, self.image_min_pixel, self.image_max_pixel))

            # EDIT: Prepare Knowledge Images
            knowledge_image_files = sources["knowledge_image"]
            knowledge_image_folder = self.data_args.knowledge_image_folder
            if isinstance(knowledge_image_files, str):
                knowledge_image_files = [knowledge_image_files]
            elif isinstance(knowledge_image_files, list):
                if isinstance(knowledge_image_files[0], bytes):
                    knowledge_image_files = [Image.open(BytesIO(img))for img in knowledge_image_files]
            # Prepare Knowledge Images
            knowledge_images = []
            success_idxs = []
            max_possible_idx = len(sources['knowledge_conversations']) - 1
            for idx, knowledge_image_file in enumerate(knowledge_image_files):
                if 2*idx+1 > max_possible_idx:
                    break
                if isinstance(knowledge_image_file, Image.Image):
                    knowledge_image_file = knowledge_image_file
                elif not os.path.exists(knowledge_image_file):
                    if not knowledge_image_file.startswith("http"):
                        knowledge_image_file = os.path.join(knowledge_image_folder, knowledge_image_file)
                    else:
                        knowledge_image_file = knowledge_image_file
                try:
                    image_info = get_image_info(knowledge_image_file, 16*28*28, 64*28*28)
                    if 2*idx+1 <= max_possible_idx:
                        knowledge_images.append(image_info)
                        success_idxs.append(2*idx)
                        success_idxs.append(2*idx+1)
                except Exception as e:
                    print(f"Error loading image {knowledge_image_file}: {e}")
            # Only extract the first 10 valid images and corresponding knowlegdes
            if knowledge_images:
                knowledge_images = knowledge_images[:10] ##NOTE: Change here if 10
                needed_conversations = len(knowledge_images) * 2
                sources['knowledge_conversations'] = [sources['knowledge_conversations'][i] for i in success_idxs[:needed_conversations]]
                # sources['knowledge_conversations'] = [sources['knowledge_conversations'][i] for i in success_idxs]
                # sources['knowledge_conversations'] = sources['knowledge_conversations'][:6]
            else:
                knowledge_images = []
                sources['knowledge_conversations'] = []
    
        elif "video" in sources:
            is_video = True
            images=None
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"

            video_files = sources["video"]
            video_folder = self.data_args.image_folder

            if isinstance(video_files, str):
                video_files = [video_files]

            videos = []
            for video_file in video_files:
                if not os.path.exists(video_file):
                    if not video_file.startswith("http"):
                        video_file = os.path.join(video_folder, video_file)
                video_input, video_kwargs = get_video_info(video_file, self.video_min_pixel, self.video_max_pixel, self.data_args.fps)
                videos.append(video_input)
        else:
            grid_key = None
            pixel_key = None
            images=None
            videos=None

        ori_sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video))
        # EDIT: Knowlegde texts are also constructed in conversation format, so process in the same way
        knowledge_sources = copy.deepcopy(llava_to_openai(sources['knowledge_conversations'], is_video=is_video))

        # Prepare Query
        all_input_ids = [] 
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []

        # Qwen2-VL uses a default system message so I've added this.
        if len(SYSTEM_MESSAGE) > 0:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX) 
            
            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))
        # Generate Input IDs and Labels
        for _, j in enumerate(range(0, len(ori_sources), 2)):
            user_input = ori_sources[j] 
            gpt_response = ori_sources[j + 1] 

            user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
            gpt_response = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"
            
            if DEFAULT_IMAGE_TOKEN in user_input:
                inputs = processor(text=[user_input], images=images, videos=videos, padding=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
            
            elif DEFAULT_VIDEO_TOKEN in user_input:
                if "Qwen2.5" in self.model_id:
                    inputs = processor(text=[user_input], images=images, videos=videos, padding=False, return_tensors='pt', **video_kwargs)
                    all_second_gird.extend(inputs["second_per_grid_ts"])
                else:
                    inputs = processor(text=[user_input], images=images, videos=videos, padding=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])

            else:
                prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)
        
        # There is no need for eos or bos tokens in the input_ids
        # Qwen2-VL does not use them
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        # EDIT: Prepare Knowledge input_ids and pixel_values
        all_knowledge_input_ids = [] 
        all_knowledge_pixel_values = []
        all_knowledge_image_grid_thw = []
        all_knowledge_second_gird = []

        # Generate Knowledge Input IDs
        for _, j in enumerate(range(0, len(knowledge_sources), 2)):
            k_user_input = knowledge_sources[j] # <image> Similar image and text knowledge for reference
            k_gpt_response = knowledge_sources[j + 1] # text knowledge
            
            knowledge_image = knowledge_images[j // 2]

            k_user_input = f"{DEFAULT_IM_START_TOKEN}{k_user_input['role']}\n{k_user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{k_gpt_response['role']}\n"
            k_gpt_response = f"{k_gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"
            
            if DEFAULT_IMAGE_TOKEN in user_input:
                k_inputs = processor(text=[k_user_input], images=knowledge_image, videos=videos, padding=False, return_tensors='pt')
                k_prompt_input_ids = k_inputs['input_ids']
                all_knowledge_pixel_values.append(k_inputs[pixel_key])
                all_knowledge_image_grid_thw.append(k_inputs[grid_key])

            k_response_input_ids = processor.tokenizer(k_gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            k_input_ids = torch.cat([k_prompt_input_ids, k_response_input_ids], dim=1).squeeze(0)
            all_knowledge_input_ids.append(k_input_ids)

        # attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids, 
            # attention_mask=attention_mask, 
            labels=labels, 
            knowledge_input_ids=all_knowledge_input_ids, # list of length 3
        )

        if pixel_key and grid_key:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw
            data_dict[knowledge_pixel_key] = all_knowledge_pixel_values # list of length 3
            data_dict[knowledge_grid_key] = all_knowledge_image_grid_thw # list of length 3

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird
        
        return data_dict

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        # EDIT: The batch size is 1
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []
        # Added for knowledge
        batch_knowledge_input_ids = []
        batch_knowledge_pixel_values = []
        batch_knowledge_image_thw = []
        
        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])
                batch_knowledge_pixel_values.append(example["knowledge_pixel_values"])
                batch_knowledge_image_thw.append(example["knowledge_image_grid_thw"])
            
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
            batch_knowledge_input_ids.append(example["knowledge_input_ids"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.append(example["second_per_grid_ts"])
        
        input_ids = pad_sequence(
            batch_input_ids, padding_side='left', padding_value=self.pad_token_id
        )
        labels = pad_sequence(batch_label_ids, padding_side='left', padding_value=IGNORE_INDEX)
        attention_mask = input_ids != self.pad_token_id
        # EDIT: no padding for Knowledge input_ids, because each of them are processed seperately
        knowledge_input_ids = [pad_sequence(knowledge_input_ids_per_batch, padding_side='left', padding_value=self.pad_token_id) for knowledge_input_ids_per_batch in batch_knowledge_input_ids if len(knowledge_input_ids_per_batch) > 0]
        knowledge_attention_mask = [k != self.pad_token_id for k in knowledge_input_ids]
        
        # print('data.py')
        # print('knowledge_input_ids', len(knowledge_input_ids))
        # print('knowledge_pixel_values', len(batch_knowledge_pixel_values))
        # print('knowledge_image_thw', len(batch_knowledge_image_thw))
        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'knowledge_input_ids': knowledge_input_ids,
            'knowledge_attention_mask': knowledge_attention_mask,
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw
            
            knowledge_pixel_values = [torch.cat(knowledge_pixel_values_per_batch, dim=0) for knowledge_pixel_values_per_batch in batch_knowledge_pixel_values]
            knowledge_image_thw = [torch.cat(knowledge_image_thw_per_batch, dim=0) for knowledge_image_thw_per_batch in batch_knowledge_image_thw]
            data_dict["knowledge_pixel_values"] = knowledge_pixel_values
            data_dict["knowledge_image_grid_thw"] = knowledge_image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        return data_dict
    

def replace_image_tokens(input_string, is_video=False):
    if is_video:
        pattern = r'\n?' + re.escape(LLAVA_VIDEO_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN
    else:
        pattern = r'\n?' + re.escape(LLAVA_IMAGE_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN

    return re.sub(pattern, replacement, input_string)

def llava_to_openai(conversations, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content[:5000],
        }
        transformed_data.append(transformed_entry)

    return transformed_data

def make_supervised_data_module(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
    )
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)
