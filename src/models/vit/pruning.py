import torch

def get_global_pruning_indices(ucb_count_scores: torch.Tensor, pos_embed_shape: torch.Size, keep_ratio: float):
    """
    Analyzes the learned ucb_count_scores to find the most important patch indices
    for global pruning.

    Args:
        ucb_count_scores (torch.Tensor): Tensor of UCB count scores from the model.
        pos_embed_shape (torch.Size): Shape of the positional embeddings (e.g., model.pos_embed.shape).
                                      Used to determine the total number of patches.
        keep_ratio (float): The ratio of patches to keep (e.g., 0.7 for 70%).

    Returns:
        torch.Tensor: A 1D tensor containing the indices of tokens to keep,
                      including the CLS token.
    """
    # Exclude CLS token from importance calculation (index 0)
    # Token importance scores are averaged across all layers and heads from UCB counts.
    # ucb_count_scores shape: (num_layers, num_heads, total_tokens)
    # We slice [:, :, 1:] to exclude CLS token (index 0) from patch importance calculation.
    token_importance_scores = ucb_count_scores[:, :, 1:].mean(dim=(0, 1))
    
    if keep_ratio <= 0:
        # If keep_ratio is 0 or less, only the CLS token is retained.
        return torch.tensor([0], device=ucb_count_scores.device)
    
    # `total_image_patches` refers to image patch tokens, excluding the CLS token.
    total_image_patches = pos_embed_shape[1] - 1 
    num_patches_to_keep = max(1, int(total_image_patches * keep_ratio))
    
    # Get the indices of the patches with the highest importance scores.
    # These indices are relative to the `token_importance_scores` tensor (0 to total_image_patches-1).
    top_patch_indices = torch.topk(token_importance_scores, k=num_patches_to_keep, dim=-1).indices
    
    # Add 1 to offset these patch indices to convert them to global token indices (due to CLS token at index 0).
    token_indices = top_patch_indices + 1
    
    # Always include the CLS token (index 0) as it's critical for classification.
    cls_token_index = torch.tensor([0], device=token_indices.device)
    
    # Concatenate and sort the indices to maintain a predictable order and ensure uniqueness.
    final_indices = torch.cat([cls_token_index, token_indices]).sort().values
    
    return final_indices