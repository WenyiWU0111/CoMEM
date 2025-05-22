import torch

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
            if model_name == 'llava':
                layer_device = next(model.language_model.model.layers[layer_idx].parameters()).device
            elif model_name == 'qwen':
                layer_device = next(model.model.layers[layer_idx].parameters()).device
            key = key.to(layer_device)
            value = value.to(layer_device)
        new_prefix_kv.append((key, value))
    return tuple(new_prefix_kv)
