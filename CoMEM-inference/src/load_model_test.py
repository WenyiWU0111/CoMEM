from transformers import AutoTokenizer, LlavaNextProcessor, LlavaNextForConditionalGeneration, LlavaForConditionalGeneration, AutoModelForCausalLM
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
import requests
import numpy as np
import os
import sys
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import auto_configure_device_map
from accelerate import dispatch_model
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
from qwen_vl_utils import process_vision_info
module_path = "CoMEM-train"
sys.path.insert(0, module_path)
from src_vlm.process_data import knowledge_processor as knowledge_processor_vlm
from src_llm.process_data import knowledge_processor as knowledge_processor_llm
from src_distill_llm.process_data import knowledge_processor as knowledge_processor_distill_llm


def load_model(model_name, model_path, max_memory=None):
    print("Loading processor and model...")
    if 'llava1.6' in model_name:
        processor = LlavaNextProcessor.from_pretrained(model_path, use_fast=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if max_memory is not None:
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto", 
                max_memory=max_memory,
                low_cpu_mem_usage=True,
            )
        else:
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        model.to("cuda")
    elif 'llava1.5' in model_name:
        processor = LlavaNextProcessor.from_pretrained(model_path, use_fast=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto", 
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )
        model.to("cuda")
    elif 'qwen2.5llm' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        processor = tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16, 
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True
        )
        model.to("cuda")
    elif 'qwen2.5' in model_name:
        processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        tokenizer = None
        if max_memory is not None:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, 
                torch_dtype=torch.float16, 
                device_map="auto",
                attn_implementation="flash_attention_2",
                max_memory=max_memory,
                low_cpu_mem_usage=True)
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, 
                torch_dtype=torch.float16, 
                attn_implementation="flash_attention_2",
                low_cpu_mem_usage=True)
        model.to("cuda")
    elif 'qwen2' in model_name:
        processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        tokenizer = None
        if max_memory:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, 
                torch_dtype=torch.float16, 
                device_map="auto",
                attn_implementation="flash_attention_2",
                max_memory=max_memory,
                low_cpu_mem_usage=True)
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, 
                torch_dtype=torch.float16, 
                attn_implementation="flash_attention_2",
                low_cpu_mem_usage=True)
        model.to("cuda")
    elif 'mplug' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if max_memory:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16, 
                device_map="auto",        
                attn_implementation="flash_attention_2",
                max_memory=max_memory,
                trust_remote_code=True
            )
        else:   
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16, 
                attn_implementation="flash_attention_2",
                trust_remote_code=True
            )
        processor = model.init_processor(tokenizer)
    elif 'llama3' in model_name:
        tokenizer, model, processor, max_length = load_pretrained_model(model_path, None, "llava_llama3", device_map="auto", max_memory=max_memory ) # Add any other thing you want to pass in llava_model_args        
    elif 'internlm2.5' in model_name:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        if max_memory:
            device_map = auto_configure_device_map(max_memory)
            print("Device map:", device_map)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=device_map,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
                max_memory=max_memory,
                low_cpu_mem_usage=True
            ).eval()
            model = dispatch_model(model, device_map=device_map)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            ).eval()
            model.to("cuda")
        model.tokenizer = tokenizer
        
    return processor, tokenizer, model

def generate_response(model_name, processor, model, image, prompt, conversation=None):
    if not conversation:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image}
                ],
            }
        ]
    if 'llava' in model_name:
        formatted_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=formatted_prompt, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=10**3, use_cache=True)
        past_kv = None
        output_text = processor.decode(output[0], skip_special_tokens=True)
    elif 'qwen2.5llm' in model_name:
        tokenizer = processor
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=10**3
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        past_kv = None
    elif 'qwen' in model_name:
        formatted_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = processor(
            text=[formatted_prompt],
            images=image_inputs,
            return_tensors="pt",
        ).to("cuda")
        outputs = model(**inputs, use_cache=True)
        past_kv = outputs.past_key_values
        past_kv = tuple(
            (tup[0].to('cpu').detach().numpy(), tup[1].to('cpu').detach().numpy()) 
            for tup in past_kv
        )
        generated_ids = model.generate(**inputs, max_new_tokens=10**3, use_cache=True)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output_text = output_text[0]
    elif 'mplug' in model_name:
        messages = [
            {'role': 'user',
             'content': f"""<|image|>{prompt}"""},
            {'role': 'assistant',
             'content': ""}]
        inputs = processor(messages, images=[image], videos=None).to("cuda")
        inputs.update({
            'tokenizer': tokenizer,
            'max_new_tokens':10**3,
            'decode_text':True,
        })
        output_text = model.generate(**inputs)[0]
        past_kv = None
    elif 'llama3' in model_name:
        image_tensor = process_images([image], processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device="cuda") for _image in image_tensor]
        conv_template = "llava_llama_3"
        question = DEFAULT_IMAGE_TOKEN + prompt
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")
        image_sizes = [image.size]
        
        cont = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=256,
        )
        output_text = tokenizer.batch_decode(cont, skip_special_tokens=True)
        output_text = output_text[0]
        past_kv = None
    elif 'internlm2.5' in model_name:
        placeholder  = "<ImageHere>"
        lines = []
        for msg in conversation:
            parts = []
            for part in msg["content"]:
                if part["type"] == "text":
                    parts.append(part["text"])
                else:  # image
                    parts.append(placeholder)
            lines.append(f"[{msg['role'].capitalize()}] " + " ".join(parts))
        eos = getattr(processor, "eos_token", "")
        formatted_prompt = "\n".join(lines) + eos
        
        output_text, _ = model.chat(tokenizer, query=formatted_prompt, image=[image], history=[], do_sample=False)
        past_kv = None
    return output_text, past_kv

def generate_response_rag(model_name, processor, model, image, prompt, similar_infos,tokenizer=None):
    if not 'llm' in model_name:
        conversation = []
        images = []
        for value in similar_infos.values():
            images.append(value['image'])
            conversation.append({
                "role": "user",
            "content": [
                {"type": "text", "text": value['desc']},
                {"type": "image", "image": value['image']}
            ]
            })
        images.append(image)
        conversation.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image}
                ],
            })
    if 'llava1.6' in model_name:
        formatted_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=images, text=formatted_prompt, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, 
                            max_new_tokens=10**3)
        output_text = processor.decode(output[0], skip_special_tokens=True)
    elif 'llava1.5' in model_name:
        context = 'Reference Information: '
        context += 'Reference Information: '.join([value['desc'][:500] for value in similar_infos.values()])
        context += prompt
        conversation = [({
                "role": "user",
                "content": [
                    {"type": "text", "text": context},
                    {"type": "image", "image": image}
                ],
            })]
        formatted_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=formatted_prompt, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, 
                            max_new_tokens=10**3)
        output_text = processor.decode(output[0], skip_special_tokens=True)
    elif 'qwen2.5llm' in model_name:
        texts = "Reference Information: "
        texts += "Reference Information:  ".join([item['desc'] for item in similar_infos.values()])
        texts += "Question: " + prompt
        tokenizer = processor
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": texts}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=10**3
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    elif 'qwen' in model_name:
        formatted_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = processor(
            text=[formatted_prompt],
            images=image_inputs,
            return_tensors="pt",
        ).to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=10**3)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output_text = output_text[0]
    elif 'mplug' in model_name:
        images = []
        messages = []
        for value in similar_infos.values():
            images.append(value['image'])
            messages.append({
                "role": "user",
                'content': f"""<|image|>{value['desc']}"""},
            )
        images.append(image)
        messages.extend(
            [{'role': 'user',
             'content': f"""<|image|>{prompt}"""},
            {'role': 'assistant',
             'content': ""}])
        inputs = processor(messages, images=images, videos=None).to("cuda")
        inputs.update({
            'tokenizer': tokenizer,
            'max_new_tokens':10**3,
            'decode_text':True,
        })
        output_text = model.generate(**inputs)[0]
    elif 'llama3' in model_name:
        
        # image_tensor = process_images(images, processor, model.config)
        image_tensor = process_images([image], processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device="cuda") for _image in image_tensor]
        conv_template = "llava_llama_3"
        conv = copy.deepcopy(conv_templates[conv_template])
        for value in similar_infos.values():
            # description = DEFAULT_IMAGE_TOKEN + value['desc']
            description = value['desc']
            conv.append_message(conv.roles[0], description)
            # conv.append_message(conv.roles[1], None)
            
            
        question = DEFAULT_IMAGE_TOKEN + prompt
        
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")
        # image_sizes = [image.size for image in images]
        image_sizes = [image.size]
        cont = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=256,
        )
        output_text = tokenizer.batch_decode(cont, skip_special_tokens=True)
        output_text = output_text[0]
        past_kv = None
    elif 'internlm2.5' in model_name:
        placeholder  = "<ImageHere>"
        lines = []
        for msg in conversation:
            parts = []
            for part in msg["content"]:
                if part["type"] == "text":
                    parts.append(part["text"])
                else:  # image
                    parts.append(placeholder)
            lines.append(f"[{msg['role'].capitalize()}] " + " ".join(parts))
        eos = getattr(processor, "eos_token", "")
        formatted_prompt = "\n".join(lines) + eos
        
        output_text, _ = model.chat(tokenizer, query=formatted_prompt, image=images, history=[], do_sample=False)
    return output_text

def generate_response_knowledge(model_name, processor, model, image, prompt, knowlegde_texts, knowlegde_images, knowledge_embedding=None, prefix_kv=None, tokenizer=None, conversation=None):
    if not conversation:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image}
                ],
            }
        ]
    formatted_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(conversation)
    inputs = processor(
        text=[formatted_prompt],
        images=image_inputs,
        return_tensors="pt",
    ).to("cuda")
    if 'distill' in model_name:
        inputs_with_knowledge = knowledge_processor_distill_llm(
            processor=processor,
            inputs=inputs,
            texts=knowlegde_texts,
            images=knowlegde_images,
            tokenizer=tokenizer,
            formatted_prompt=formatted_prompt
        ).to("cuda")
    elif 'llm' in model_name:
        inputs_with_knowledge = knowledge_processor_llm(
            processor=processor,
            inputs=inputs,
            texts=knowlegde_texts,
            images=knowlegde_images,
            tokenizer=tokenizer,
            formatted_prompt=formatted_prompt
        ).to("cuda")
    else:
        inputs_with_knowledge = knowledge_processor_vlm(
            processor=processor,
            inputs=inputs,
            texts=knowlegde_texts,
            images=knowlegde_images,
            tokenizer=tokenizer,
            formatted_prompt=formatted_prompt
        ).to("cuda")
    inputs_with_knowledge['kowledge_compress_embedding'] = knowledge_embedding
    
    import time
    time0 = time.time()
    generated_ids = model.generate(**inputs_with_knowledge, max_new_tokens=100*10, 
                                   use_cache=True, past_key_values=prefix_kv,
                                   temperature=0.1,
                                   top_p=0.001,
                                   repetition_penalty=1.05
                                   )
    time1 = time.time()
    print('time for 1 sample generation:', time1-time0)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs_with_knowledge.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    output_text = output_text[0]
    return output_text


def get_past_key_value_text(processor, tokenizer, model, image, info, prefix_idx, top_tokens, question=None, pool=False):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "text", "text": 'Given Information:'+info},
                {"type": "image", "image": image}
            ],
        }
    ]
    formatted_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    try:
        input_ids = processor(images=image, text=formatted_prompt, return_tensors="pt").to("cuda")
    except:
        return None

    # Enable evaluation mode (disable dropout)
    model.eval()
    # Generate next token with attention weights
    if top_tokens != 100:
        with torch.no_grad():
            outputs = model(**input_ids, output_attentions=True)

        past_key_values = outputs.past_key_values
        print('before', past_key_values[0][0].shape)
        ########### Top Attention Weight to compress Vision Tokens ###########
        def select_top_tokens_past_kv(past_key_values, top_token_idxs):
            final_past_key_values = []
            # Iterate over each layer's key-value pair
            for layer_idx, (layer_k, layer_v) in enumerate(past_key_values):
                batch_size, num_heads, _, head_dim = layer_k.shape  
                #Move tensors to CPU first
                layer_k = layer_k.to("cpu")
                layer_v = layer_v.to("cpu")          
                # Initialize new tensors with new sequence length
                new_k = torch.zeros((batch_size, num_heads, len(top_token_idxs), head_dim), 
                                dtype=layer_k.dtype, device="cpu")
                new_v = torch.zeros((batch_size, num_heads, len(top_token_idxs), head_dim), 
                                dtype=layer_v.dtype, device="cpu")            
                new_k[..., :len(top_token_idxs), :] = layer_k[..., top_token_idxs, :]
                new_v[..., :len(top_token_idxs), :] = layer_v[..., top_token_idxs, :]
                final_past_key_values.append((new_k, new_v))
            return final_past_key_values   
        
        layer_attention = outputs.attentions[0][0].cpu().numpy()
        layer_attention = np.mean(layer_attention, axis=0) 
        attention_avg = np.mean(layer_attention, axis=0) 
        top_relative_indices = np.argsort(attention_avg)[-top_tokens:] 
        all_indices = list(range(len(input_ids['input_ids'][0])))
        size = int(len(all_indices) * top_tokens * 0.01)
        top_relative_indices = np.argsort(attention_avg)[-size:] 
        past_key_values = select_top_tokens_past_kv(past_key_values, top_relative_indices)
        print('after', past_key_values[0][0].shape)
    else:
        with torch.no_grad():
            outputs = model(**input_ids, output_attentions=False)
        past_key_values = outputs.past_key_values
        
    final_past_key_values = list(past_key_values)
    ##### Only keep past_key_values at a specific layer #####
    for layer_idx in range(len(final_past_key_values)):
        if layer_idx not in prefix_idx: 
            final_past_key_values[layer_idx] = (None, None)  
        else:
            key, value = final_past_key_values[layer_idx]
            final_past_key_values[layer_idx] = (key*1.5, value*1.5) 
    return tuple(final_past_key_values)

def concatenate_past_key_values(all_past_key_values, prefix_idx, group_indices=None, filter_idx_list=None):
    # Initialize the combined past_key_values
    num_layers = len(all_past_key_values[0])
    combined = []
    
    for layer_idx in range(num_layers):
        if layer_idx not in prefix_idx:
            combined.append((None, None))
            continue
            
        # Get all key tensors and all value tensors for this layer
        layer_keys = []
        layer_values = []
        
        for i, past_kv in enumerate(all_past_key_values):
            if past_kv is None:
                continue
            if past_kv[layer_idx][0] is not None:  # Check if layer was kept
                if filter_idx_list is not None:
                    layer_keys.append(past_kv[layer_idx][0][:, :, filter_idx_list[i], :])
                    layer_values.append(past_kv[layer_idx][1][:, :, filter_idx_list[i], :])
                else:
                    layer_keys.append(past_kv[layer_idx][0])
                    layer_values.append(past_kv[layer_idx][1])
        
        # Concatenate along sequence length dimension (dim=2)
        combined_keys = torch.cat(layer_keys, dim=2)
        combined_values = torch.cat(layer_values, dim=2)
        if group_indices is not None:
            group_keys = []
            group_values = []
            for indices in group_indices.values():
                #print('indices:',list(indices))
                group_keys.append(combined_keys[:, :, list(indices), :].mean(dim=2, keepdim=True))
                group_values.append(combined_values[:, :, list(indices), :].mean(dim=2, keepdim=True))
            group_keys = torch.cat(group_keys, dim=2)
            group_values = torch.cat(group_values, dim=2)
            combined.append((group_keys, group_values))
        else:
            combined.append((combined_keys.to("cuda"), combined_values.to("cuda")))
    
    return tuple(combined)

def move_prefix_kv_to_model_device(prefix_kv, model, model_name):
    """
    Ensure that each layer's key-value tensors in prefix_kv are on the correct device
    based on where the corresponding layer resides in a sharded model.
    """
    new_prefix_kv = []
    for layer_idx, (key, value) in enumerate(prefix_kv):
        if key is not None and value is not None:
            # Get the device of the corresponding layer
            if 'llava' in model_name:
                layer_device = next(model.language_model.model.layers[layer_idx].parameters()).device
            elif 'qwen' in model_name:
                layer_device = next(model.model.layers[layer_idx].parameters()).device
            key = key.to(layer_device)
            value = value.to(layer_device)
        new_prefix_kv.append((key, value))
    return tuple(new_prefix_kv)

def generate_response_with_kv(model_name, processor, model, image, prompt, prefix_kv):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image}
            ],
        }
    ]
    formatted_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    if 'llava' in model_name:
        inputs = processor(images=image, text=formatted_prompt, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, 
                                max_new_tokens=10**3,
                                past_key_values=prefix_kv,
                                use_cache=True)
        output_text = processor.decode(output[0], skip_special_tokens=True)
        past_kv = None
    elif 'qwen' in model_name:
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = processor(
            text=[formatted_prompt],
            images=image_inputs,
            return_tensors="pt",
        ).to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=10**3,
                                       past_key_values=prefix_kv,
                                       temperature=0.1,
                                        top_p=0.001,
                                        repetition_penalty=1.05,
                                       use_cache=True)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output_text = output_text[0]
        past_kv = None
    return output_text, past_kv

