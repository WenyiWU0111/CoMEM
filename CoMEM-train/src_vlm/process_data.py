import sys
import copy
import torch
module_path = "CoMEM-train"
sys.path.append(module_path)
from src_vlm.training.data import get_image_info, llava_to_openai, pad_sequence
from src_vlm.training.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

# INFERENCE: Convert knowledge image and text to input_ids
def knowledge_processor(processor, inputs, texts=None, images=None, tokenizer=None, formatted_prompt=None):
    knowledge_images = []
    for idx, knowledge_image_file in enumerate(images):
        knowledge_images.append(get_image_info(knowledge_image_file, 16*28*28, 64*28*28))
    
    k_conversations = []
    for text in texts:
        k_conversations.append({
            "from": "human",
                "value": "<image>\n Similar image and text knowledge for reference"
        })
        k_conversations.append({
            "from": "gpt",
            "value": text
        })
    knowledge_sources = copy.deepcopy(llava_to_openai(k_conversations, is_video=False))
    # Prepare Knowledge input_ids and pixel_values
    all_knowledge_input_ids = [] 
    all_knowledge_pixel_values = []
    all_knowledge_image_grid_thw = []

    # Generate Knowledge Input IDs
    for _, j in enumerate(range(0, len(knowledge_sources), 2)):
        user_input = knowledge_sources[j] # <image> Similar image and text knowledge for reference
        gpt_response = knowledge_sources[j + 1] # text knowledge
        
        knowledge_image = knowledge_images[j // 2]

        k_user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
        k_gpt_response = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"
        
        if DEFAULT_IMAGE_TOKEN in k_user_input:
            k_inputs = processor(text=[k_user_input], images=knowledge_image, videos=None, padding=False, return_tensors='pt')
            k_prompt_input_ids = k_inputs['input_ids']
            all_knowledge_pixel_values.append(k_inputs["pixel_values"])
            all_knowledge_image_grid_thw.append(k_inputs["image_grid_thw"])

        k_response_input_ids = processor.tokenizer(k_gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']
        k_input_ids = torch.cat([k_prompt_input_ids, k_response_input_ids], dim=1).squeeze(0)
        all_knowledge_input_ids.append(k_input_ids)
    
    knowledge_input_ids = [pad_sequence(all_knowledge_input_ids, padding_side='left', padding_value=151643)]
    knowledge_attention_mask = [k != 151643 for k in knowledge_input_ids]
    knowledge_pixel_values = [torch.cat(all_knowledge_pixel_values, dim=0)]
    knowledge_image_grid_thw = [torch.cat(all_knowledge_image_grid_thw, dim=0)]
    
    inputs['knowledge_input_ids'] = knowledge_input_ids
    inputs['knowledge_attention_mask'] = knowledge_attention_mask
    inputs['knowledge_pixel_values'] = knowledge_pixel_values
    inputs['knowledge_image_grid_thw'] = knowledge_image_grid_thw
    
    return inputs