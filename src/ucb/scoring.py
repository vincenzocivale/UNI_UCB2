import torch
from typing import Tuple

def calculate_ucb_selection(
    attn_probs: torch.Tensor,
    iteration: int,
    count_score_buffer: torch.Tensor,
    keep_k: int, # The dynamic k (number of patches to keep)
    beta: float,
    exclude_cls: bool,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates UCB scores and determines selected patch indices along with score delta for updates.

    Args:
        attn_probs (torch.Tensor): Attention probabilities from the current attention layer (B, H, N, N).
        iteration (int): Current training iteration, used for UCB exploration term.
        count_score_buffer (torch.Tensor): Buffer tracking patch selection counts (H, N).
        keep_k (int): Number of patches to keep.
        beta (float): UCB exploration constant.
        exclude_cls (bool): Whether CLS token is excluded from patch selection.
        device (torch.device): The device the tensors are on.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - selected_indices (torch.Tensor): (B, keep_k) Indices of the selected patches.
            - score_delta (torch.Tensor): (H, N) Delta to update UCB count scores.
    """
    B, H, N, _ = attn_probs.shape

    # Calculate token importance scores: how much attention each patch token receives.
    # We differentiate between tokens (N) which include CLS, and image patches (N-1) which exclude CLS.
    if exclude_cls:
        # For image patch tokens (excluding CLS), average attention received from all query tokens.
        token_importance_scores = attn_probs[:, :, :, 1:].mean(dim=2)  # (B, H, num_image_patches)
        relevant_counts = count_score_buffer[:, 1:]  # (H, num_image_patches) for exploration
    else:
        # If CLS is not excluded, consider all tokens as potential patches.
        token_importance_scores = attn_probs.mean(dim=2)  # (B, H, N)
        relevant_counts = count_score_buffer  # (H, N) for exploration

    # UCB exploration term: Encourages selection of less-explored tokens.
    # The `+ 1.0` in `iteration + 1.0` prevents division by zero at the very start.
    # `relevant_counts + 1e-6` prevents division by zero for unvisited tokens.
    ucb_exploration = beta * torch.sqrt(
        torch.log(torch.tensor(iteration + 1.0, device=device)) / (relevant_counts + 1e-6)
    )

    # Combine actual importance scores with the UCB exploration bonus.
    ucb_scores = token_importance_scores + ucb_exploration.unsqueeze(0)  # (B, H, num_relevant_tokens)

    # Global selection: For each sample in the batch, select the `keep_k` tokens
    # with the highest UCB scores, averaged across all attention heads.
    global_ucb_scores = ucb_scores.mean(dim=1)  # (B, num_relevant_tokens)
    _, selected_indices = torch.topk(global_ucb_scores, k=keep_k, dim=-1)  # (B, keep_k)

    # Calculate score_delta: This tensor represents the updates for the UCB count buffer.
    # Each selected token contributes to its respective count.
    score_delta = torch.zeros_like(count_score_buffer, dtype=torch.float32)
    
    if exclude_cls:
        for b in range(B):
            # For each selected patch, increment its count.
            # We add 1 to `selected_indices` to convert them back to global token indices
            # if CLS token was excluded during scoring.
            indices_to_update = (selected_indices[b] + 1).long()
            score_delta[:, indices_to_update] += 1.0 / B # Normalize across batch
    else:
        for b in range(B):
            # If CLS was not excluded, `selected_indices` are already global token indices.
            indices_to_update = selected_indices[b].long()
            score_delta[:, indices_to_update] += 1.0 / B # Normalize across batch

    return selected_indices, score_delta