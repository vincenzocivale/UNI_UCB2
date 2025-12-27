import torch
from typing import Tuple, List

def get_input_aware_and_global_token_indices(
    initial_x: torch.Tensor,
    first_block: torch.nn.Module, # The first UCBBlock
    ucb_count_scores: torch.Tensor, # For global pruning
    pos_embed_shape: torch.Size, # For global pruning
    global_keep_ratio: float,
    input_aware_extra_tokens: int,
    get_global_pruning_indices_fn # Pass the function itself
) -> torch.Tensor:
    # This utility combines globally pruned tokens with input-aware selected tokens.
    # It first performs a lightweight pass through the first transformer block to identify
    # input-specific tokens, then merges these with the globally determined important tokens.

    B, N_full, C = initial_x.shape # N_full includes CLS and all potential image tokens

    # Step 1: Perform a single full-token attention pass through ONLY the first transformer block.
    # This is done to obtain attention probabilities without applying any UCB-based pruning
    # or updating UCB counters. This pass is purely for input-aware token identification.
    with torch.no_grad(): # Critical: Ensure no gradients are tracked for this temporary pass.
        # We pass dummy values for `counter` and `ucb_enabled` as UCB is disabled,
        # and `ucb_count_score` will not be updated due to `no_grad()`.
        _, _, attn_probs_first_block = first_block(
            initial_x, 
            counter=0, 
            ucb_enabled=False, 
            ucb_count_score=ucb_count_scores[0], 
            return_attn_probs=True # Request attention probabilities for analysis.
        )
    
    # `attn_probs_first_block` shape: (Batch_Size, Num_Heads, Seq_Length, Seq_Length)
    # Step 2: Use the CLS-to-token attention from this first block to identify M image-specific tokens.
    # The CLS token is typically at index 0; image tokens start from index 1.
    cls_to_image_token_attn = attn_probs_first_block[:, :, 0, 1:] # (B, H, N_full - 1)
    
    # Average attention scores over all heads and across the batch.
    # This provides a single importance score for each image token across the entire batch,
    # enabling batch-wise token selection as requested.
    avg_cls_to_image_token_attn_across_batch = cls_to_image_token_attn.mean(dim=(0, 1)) # (N_full - 1)
    
    # Select the top-M image tokens based on these averaged attention scores.
    # M is capped by the total number of available image tokens.
    M = min(input_aware_extra_tokens, N_full - 1) 
    _, input_aware_image_token_indices_relative = torch.topk(
        avg_cls_to_image_token_attn_across_batch, k=M, dim=-1
    ) # (M,)
    
    # Convert these relative image token indices to global token indices by adding 1
    # (to account for the CLS token's presence at index 0).
    input_aware_token_indices = input_aware_image_token_indices_relative + 1 # (M,)
    
    # Step 3: Merge these M input-aware tokens with the K globally selected UCB tokens.
    # Retrieve the globally important tokens based on accumulated UCB scores.
    globally_important_tokens = get_global_pruning_indices_fn(
        ucb_count_scores=ucb_count_scores,
        pos_embed_shape=pos_embed_shape,
        keep_ratio=global_keep_ratio
    )
    
    # Combine the globally important tokens and the input-aware tokens.
    final_token_indices = torch.cat([
        globally_important_tokens.to(initial_x.device),
        input_aware_token_indices.to(initial_x.device) 
    ])
    
    # Remove any duplicate indices and sort the final set to maintain a deterministic order.
    final_token_indices = torch.unique(final_token_indices).sort().values
    
    return final_token_indices