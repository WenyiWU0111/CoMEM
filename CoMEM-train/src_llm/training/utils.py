# QEDIT: calculate the position id for qformer_embedding+raw_embedding, and concatenate qformer_embedding+raw_embedding
import torch

def get_qformer_position_id(q_former_list, raw_embeddings, raw_position_ids):
    q_former_outputs = q_former_list
    # 1. Get dimensions
    batch_size = q_former_outputs.shape[0]
    q_former_total_len = q_former_outputs.shape[1]
    raw_len = raw_embeddings.shape[1]
    total_len = q_former_total_len + raw_len
    device = q_former_outputs.device
    dtype = raw_embeddings.dtype

    # 2. Create sequential position IDs for Q-former outputs
    q_former_position_ids = torch.zeros(
        3,  # 3 dimensions for Qwen2.5
        batch_size,
        q_former_total_len,
        dtype=raw_embeddings.dtype,
        device=raw_embeddings.device
    )

    # Fill sequentially (all 3 dimensions get same values for Q-former outputs)
    for dim in range(3):
        q_former_position_ids[dim] = torch.arange(
            q_former_total_len, 
            device=device
        ).unsqueeze(0).expand(batch_size, -1)

    # 3. Ensure position_ids_4 starts after Q-former position IDs
    max_q_former_pos = q_former_position_ids.max()
    min_raw_pos = raw_position_ids.min()

    # Apply offset if needed
    raw_position_ids = raw_position_ids.clone()
    if min_raw_pos <= max_q_former_pos:
        offset = max_q_former_pos + 1 - min_raw_pos
        offset = offset.to(dtype=raw_position_ids.dtype)
        raw_position_ids += offset

    # 4. Concatenate position IDs
    final_position_ids = torch.cat([q_former_position_ids, raw_position_ids], dim=2)

    # 5. Concatenate embeddings
    q_former_outputs = q_former_outputs.to(raw_embeddings.device)
    concatenated_embeddings = torch.cat([
        q_former_outputs,
        raw_embeddings
    ], dim=1)

    # 6. Calculate final mrope_position_deltas
    # This is typically the max position ID + 1 - total sequence length
    max_position = final_position_ids.max()
    final_mrope_deltas = (max_position + 1 - total_len).unsqueeze(0).expand(batch_size, 1).to(q_former_outputs.device)
    
    return concatenated_embeddings, final_position_ids