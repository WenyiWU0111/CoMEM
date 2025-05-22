def auto_configure_device_map(max_memory):
    num_gpus = len(max_memory)
    gpu_list = list(max_memory)
    
    num_trans_layers = 32
    per_gpu_layers = 38 / num_gpus

    device_map = {
        'vit': gpu_list[0],
        'vision_proj': gpu_list[0],
        'model.tok_embeddings': gpu_list[0],
        'plora_glb_GN': gpu_list[0],
        'plora_sub_GN': gpu_list[0],
        'model.norm': gpu_list[-1],
        'output': gpu_list[-1],
    }

    used = 3
    
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'model.layers.{i}'] = gpu_list[gpu_target]
        used += 1

    return device_map