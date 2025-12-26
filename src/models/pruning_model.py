
import torch
import torch.nn as nn
import timm
import numpy as np

class UCBAttention(nn.Module):
    """
    Attention module with patch pruning capabilities (UCB or random).
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 k=None, keep_ratio=None, beta=1.0, exclude_cls=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.k = k
        self.keep_ratio = keep_ratio
        self.beta = beta
        self.exclude_cls = exclude_cls

    def _get_k(self, N):
        if self.exclude_cls:
            total_patches = N - 1
        else:
            total_patches = N

        if self.keep_ratio is not None:
            if self.keep_ratio <= 0:
                return 0
            return max(1, int(total_patches * self.keep_ratio))
        elif self.k is not None:
            return min(self.k, total_patches)
        return total_patches

    def random_pruning(self, attn_probs, v):
        """
        Applies random patch pruning.
        Returns context, None, and the indices of kept and removed tokens.
        """
        B, H, N, D = v.shape # D is head_dim
        device = v.device
        k_patches_to_keep = self._get_k(N)

        if k_patches_to_keep >= (N - 1 if self.exclude_cls else N):
            kept_token_indices_list = [torch.arange(1, N, device=device, dtype=torch.long) for _ in range(B)] if self.exclude_cls else [torch.arange(N, device=device, dtype=torch.long) for _ in range(B)]
            removed_token_indices_list = [torch.tensor([], device=device, dtype=torch.long) for _ in range(B)]
            return torch.matmul(self.attn_drop(attn_probs), v), None, kept_token_indices_list, removed_token_indices_list

        if k_patches_to_keep == 0 and self.exclude_cls:
            kept_token_indices_list = [torch.tensor([], device=device, dtype=torch.long) for _ in range(B)]
            removed_token_indices_list = [torch.arange(1, N, device=device, dtype=torch.long) for _ in range(B)]
            mask = torch.zeros(B, H, N, N, device=device, dtype=attn_probs.dtype)
            mask[:, :, :, 0] = 1.0
            mask[:, :, 0, :] = 1.0
            pruned_attn = attn_probs * mask
            pruned_attn = pruned_attn / (pruned_attn.sum(dim=-1, keepdim=True) + 1e-8)
            context = torch.matmul(self.attn_drop(pruned_attn), v)
            return context, None, kept_token_indices_list, removed_token_indices_list # No score delta for random pruning

        patch_indices_relative = np.arange(N - (1 if self.exclude_cls else 0))
        
        kept_token_indices_list = []
        removed_token_indices_list = []
        
        for b in range(B):
            selected_patch_indices_relative_b = np.random.choice(patch_indices_relative, k_patches_to_keep, replace=False)
            kept_token_indices_list.append(torch.tensor(selected_patch_indices_relative_b + (1 if self.exclude_cls else 0), device=device, dtype=torch.long))
            
            removed_patch_indices_relative_b = np.array([idx for idx in patch_indices_relative if idx not in selected_patch_indices_relative_b])
            removed_token_indices_list.append(torch.tensor(removed_patch_indices_relative_b + (1 if self.exclude_cls else 0), device=device, dtype=torch.long))

    
        mask = torch.zeros(B, H, N, N, device=device, dtype=attn_probs.dtype)
        if self.exclude_cls:
            mask[:, :, :, 0] = 1.0
            mask[:, :, 0, :] = 1.0
            for b in range(B):
                mask[b, :, :, kept_token_indices_list[b]] = 1.0
                mask[b, :, kept_token_indices_list[b], :] = 1.0
        else:
            for b in range(B):
                mask[b, :, :, kept_token_indices_list[b]] = 1.0
                mask[b, :, kept_token_indices_list[b], :] = 1.0
        
        pruned_attn = attn_probs * mask
        pruned_attn = pruned_attn / (pruned_attn.sum(dim=-1, keepdim=True) + 1e-8)
        context = torch.matmul(self.attn_drop(pruned_attn), v)
        return context, None, kept_token_indices_list, removed_token_indices_list # No score delta for random pruning

    def ucb_score_pruning(self, attn_scores, attn_probs, v, iteration, count_score_buffer):
        """
        Applies UCB-based patch pruning on attention probabilities.
        Returns context, score_delta, and the indices of kept and removed tokens.
        """
        B, H, N, D = v.shape # D is head_dim
        device = v.device
        
        k_patches_to_keep = self._get_k(N)

        # Handle cases where no pruning or full pruning happens
        if k_patches_to_keep >= (N - 1 if self.exclude_cls else N):
            # Keep all patches (excluding CLS if specified)
            kept_token_indices_list = [torch.arange(1, N, device=device, dtype=torch.long) for _ in range(B)] if self.exclude_cls else [torch.arange(N, device=device, dtype=torch.long) for _ in range(B)]
            removed_token_indices_list = [torch.tensor([], device=device, dtype=torch.long) for _ in range(B)]
            return torch.matmul(self.attn_drop(attn_probs), v), None, kept_token_indices_list, removed_token_indices_list

        if k_patches_to_keep == 0 and self.exclude_cls:
            # Keep only CLS token
            kept_token_indices_list = [torch.tensor([], device=device, dtype=torch.long) for _ in range(B)]
            removed_token_indices_list = [torch.arange(1, N, device=device, dtype=torch.long) for _ in range(B)]

            mask = torch.zeros(B, H, N, N, device=device, dtype=attn_probs.dtype)
            mask[:, :, :, 0] = 1.0
            mask[:, :, 0, :] = 1.0
            
            pruned_attn = attn_probs * mask
            pruned_attn = pruned_attn / (pruned_attn.sum(dim=-1, keepdim=True) + 1e-8)
            pruned_attn = self.attn_drop(pruned_attn)
            context = torch.matmul(pruned_attn, v)
            
            score_delta = torch.zeros_like(count_score_buffer, dtype=torch.float32)
            score_delta[:, 0] += 1.0 # Only CLS is implicitly "kept" for score purposes
            return context, score_delta, kept_token_indices_list, removed_token_indices_list

        # --- UCB score calculation to identify patches to keep ---
        if self.exclude_cls:
            # We are working with patch tokens only for UCB selection
            patch_scores = attn_probs[:, :, :, 1:].mean(dim=2) # Scores for non-CLS tokens
            relevant_counts = count_score_buffer[:, 1:]
        else:
            patch_scores = attn_probs.mean(dim=2)
            relevant_counts = count_score_buffer

        ucb_exploration = self.beta * torch.sqrt(
            torch.log(torch.tensor(iteration + 1.0, device=device)) / (relevant_counts + 1e-6)
        )

        ucb_scores = patch_scores + ucb_exploration.unsqueeze(0)
        global_ucb_scores = ucb_scores.mean(dim=1) # Average across heads
        
        # Get indices of top-k patches (0-indexed relative to patches)
        _, top_k_patch_indices_relative = torch.topk(global_ucb_scores, k=k_patches_to_keep, dim=-1)

        # --- Determine full token indices (including CLS if present) ---
        all_patch_indices_relative = torch.arange(N - (1 if self.exclude_cls else 0), device=device, dtype=torch.long)

        kept_token_indices_list = []
        removed_token_indices_list = []
        for b in range(B):
            kept_indices_b_relative_to_patches = top_k_patch_indices_relative[b]
            kept_token_indices_list.append(kept_indices_b_relative_to_patches + (1 if self.exclude_cls else 0))

            removed_indices_b_relative_to_patches = torch.tensor(
                [idx for idx in all_patch_indices_relative if idx not in kept_indices_b_relative_to_patches],
                device=device, dtype=torch.long
            )
            removed_token_indices_list.append(removed_indices_b_relative_to_patches + (1 if self.exclude_cls else 0))
        
        # Now, compute context using original attention masking, no actual token removal yet
        # This ensures residual connection works within UCBEncoderBlock
        mask = torch.zeros(B, H, N, N, device=device, dtype=attn_probs.dtype)
        if self.exclude_cls:
            mask[:, :, :, 0] = 1.0 # CLS token can attend to all
            mask[:, :, 0, :] = 1.0 # All tokens can attend to CLS
            for b in range(B):
                # Mask connections for kept tokens
                mask[b, :, :, kept_token_indices_list[b]] = 1.0
                mask[b, :, kept_token_indices_list[b], :] = 1.0
        else:
            for b in range(B):
                # Mask connections for kept tokens
                mask[b, :, :, kept_token_indices_list[b]] = 1.0
                mask[b, :, kept_token_indices_list[b], :] = 1.0

        pruned_attn = attn_probs * mask
        # Normalize after masking, avoiding division by zero with small epsilon
        pruned_attn = pruned_attn / (pruned_attn.sum(dim=-1, keepdim=True) + 1e-8)
        pruned_attn = self.attn_drop(pruned_attn)
        context = torch.matmul(pruned_attn, v) # B, H, N, D (N is original)

        # Adjust score_delta calculation (still counts which patches were selected)
        score_delta = torch.zeros_like(count_score_buffer, dtype=torch.float32)
        for b in range(B):
            # We need to update original 0-indexed patch indices for the buffer
            # This will be the indices relative to the full sequence (including CLS)
            indices_to_update = top_k_patch_indices_relative[b] + (1 if self.exclude_cls else 0)
            score_delta[:, indices_to_update] += 1.0 / B

        return context, score_delta, kept_token_indices_list, removed_token_indices_list
    
    def forward(self, x, counter, pruning_enabled, ucb_count_score, selection_mode='ucb', ucb_update_enabled=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_scores = (q @ k.transpose(-1, -2)) * self.scale
        attn_probs = attn_scores.softmax(dim=-1)

        score_delta = None
        kept_token_indices_list = [torch.arange(N, device=x.device, dtype=torch.long) for _ in range(B)] # Default to keeping all
        removed_token_indices_list = [torch.tensor([], device=x.device, dtype=torch.long) for _ in range(B)] # Default to removing none

        if pruning_enabled and counter > 50: # Warm-up period
            if selection_mode == 'ucb':
                context, score_delta_tmp, kept_token_indices_list, removed_token_indices_list = self.ucb_score_pruning(
                    attn_scores, attn_probs, v, iteration=counter, count_score_buffer=ucb_count_score
                )
                if ucb_update_enabled: # Only propagate score_delta if flag is True
                    score_delta = score_delta_tmp
                else:
                    score_delta = None # Do not update UCB scores
            elif selection_mode == 'random':
                context, score_delta, kept_token_indices_list, removed_token_indices_list = self.random_pruning(attn_probs, v)
            else:
                raise ValueError(f"Unknown selection mode: {selection_mode}")
        else:
            attn_probs = self.attn_drop(attn_probs)
            context = torch.matmul(attn_probs, v)
            # If pruning is not enabled, all non-CLS tokens are "kept" in terms of selection
            if self.exclude_cls:
                kept_token_indices_list = [torch.arange(1, N, device=x.device, dtype=torch.long) for _ in range(B)]
            else:
                kept_token_indices_list = [torch.arange(N, device=x.device, dtype=torch.long) for _ in range(B)]


        x = context.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, score_delta, kept_token_indices_list, removed_token_indices_list

class UCBEncoderBlock(nn.Module):
    def __init__(self, original_block: nn.Module, **ucb_kwargs):
        super().__init__()
        self.norm1 = original_block.norm1
        self.norm2 = original_block.norm2
        self.mlp = original_block.mlp
        
        self.ls1 = getattr(original_block, 'ls1', nn.Identity())
        self.ls2 = getattr(original_block, 'ls2', nn.Identity())
        self.drop_path1 = getattr(original_block, 'drop_path1', nn.Identity())
        self.drop_path2 = getattr(original_block, 'drop_path2', nn.Identity())

        orig_attn = original_block.attn
        self.attn = UCBAttention(
            dim=orig_attn.qkv.in_features,
            num_heads=orig_attn.num_heads,
            qkv_bias=orig_attn.qkv.bias is not None,
            attn_drop=orig_attn.attn_drop.p if hasattr(orig_attn.attn_drop, 'p') else 0.0,
            proj_drop=orig_attn.proj_drop.p if hasattr(orig_attn.proj_drop, 'p') else 0.0,
            **ucb_kwargs
        )
        
        state_dict = orig_attn.state_dict()
        self.attn.load_state_dict(state_dict, strict=False)

    def forward(self, x, counter, pruning_enabled, ucb_count_score, selection_mode, ucb_update_enabled=False):
        B, N_original, C = x.shape
        # Attention block
        h_attn, score_delta, kept_token_indices_list, removed_token_indices_list = self.attn(
            self.norm1(x), counter, pruning_enabled, ucb_count_score, selection_mode, ucb_update_enabled
        )
        # Apply first residual connection
        x_after_attn = x + self.drop_path1(self.ls1(h_attn.transpose(1, 2).reshape(B, N_original, C)))

        # --- Token merging/selection ---
        x_new_sequence = []
        for b in range(B):
            current_batch_tokens = []
            
            # 1. Add CLS token if exclude_cls is True (it's always index 0)
            if self.attn.exclude_cls:
                current_batch_tokens.append(x_after_attn[b, 0:1, :]) # (1, C)
            
            # 2. Add kept tokens
            kept_indices_b = kept_token_indices_list[b]
            if len(kept_indices_b) > 0:
                # Ensure kept_indices_b are 1D for index_select
                kept_tokens = torch.index_select(x_after_attn[b], dim=0, index=kept_indices_b) # (num_kept, C)
                current_batch_tokens.append(kept_tokens)
            
            # 3. Compute and add merged token from removed tokens
            removed_indices_b = removed_token_indices_list[b]
            if len(removed_indices_b) > 0:
                removed_tokens = torch.index_select(x_after_attn[b], dim=0, index=removed_indices_b) # (num_removed, C)
                merged_token = removed_tokens.mean(dim=0, keepdim=True) # (1, C)
                current_batch_tokens.append(merged_token)
            
            # Concatenate all tokens for this batch item
            x_new_sequence.append(torch.cat(current_batch_tokens, dim=0)) # (N_new, C)
        
        # Stack all batch items to form (B, N_new, C)
        x_new = torch.stack(x_new_sequence, dim=0)

        # MLP block (applied to the new, potentially shorter sequence)
        x_final = x_new + self.drop_path2(self.ls2(self.mlp(self.norm2(x_new))))
        return x_final, score_delta

class VisionTransformerUCB(nn.Module):
    def __init__(self, model_name="hf-hub:MahmoodLab/uni", pretrained=True, n_classes=None,
                 k=None, keep_ratio=None, beta=1.0, exclude_cls=True, selection_mode='ucb'):
        super().__init__()
        print(f"Loading source model '{model_name}'...")
        source_model = timm.create_model(model_name, pretrained=pretrained, init_values=1e-5)

        self.keep_ratio = keep_ratio
        self.selection_mode = selection_mode
        self.patch_embed = source_model.patch_embed
        self.cls_token = source_model.cls_token
        self.pos_embed = source_model.pos_embed
        self.pos_drop = source_model.pos_drop
        self.norm = source_model.norm
        
        if hasattr(source_model, 'head'):
            if hasattr(source_model.head, 'in_features'):
                head_dim = source_model.head.in_features
            else:
                head_dim = source_model.head.weight.shape[1] if hasattr(source_model.head, 'weight') else 1024
        else:
            head_dim = source_model.embed_dim if hasattr(source_model, 'embed_dim') else 1024
            
        self.head = nn.Linear(head_dim, n_classes if n_classes is not None else head_dim)
        self.n_classes = n_classes if n_classes is not None else head_dim

        self.blocks = nn.ModuleList([
            UCBEncoderBlock(block, k=k, keep_ratio=keep_ratio, beta=beta, exclude_cls=exclude_cls) 
            for block in source_model.blocks
        ])

        num_layers = len(self.blocks)
        # Check if blocks list is not empty to safely access self.blocks[0]
        if num_layers > 0:
            num_heads = self.blocks[0].attn.num_heads
        else:
            # Fallback if no blocks are created (e.g., in a very simplified test case)
            num_heads = 1 # Default or a sensible fallback
        
        # Determine num_patches based on pos_embed shape
        # pos_embed shape is (1, num_patches_with_cls_token, embed_dim)
        num_patches_with_cls_token = self.pos_embed.shape[1]

        self.register_buffer("ucb_count_scores", torch.ones(num_layers, num_heads, num_patches_with_cls_token))

    def forward(self, pixel_values: torch.Tensor, counter: int = 0, pruning_enabled: bool = True, labels: torch.Tensor = None, top_k_indices: torch.Tensor = None, ucb_update_enabled: bool = False):
        if top_k_indices is not None:
             # This path is for static pruning (token dropping), which user clarified is not desired for new pipeline.
             # However, it remains for compatibility if it's used elsewhere.
             return self.forward_pruned(pixel_values, top_k_indices, labels)
        else:
             return self.forward_dynamic(pixel_values, counter, pruning_enabled, labels, ucb_update_enabled)

    def forward_dynamic(self, pixel_values: torch.Tensor, counter: int = 0, pruning_enabled: bool = True, labels: torch.Tensor = None, ucb_update_enabled: bool = False):
        x = pixel_values
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for i, block in enumerate(self.blocks):
            x, score_delta = block(
                x, 
                counter=counter, 
                pruning_enabled=pruning_enabled,
                ucb_count_score=self.ucb_count_scores[i],
                selection_mode=self.selection_mode,
                ucb_update_enabled=ucb_update_enabled # Pass the new flag
            )
            
            if score_delta is not None and self.selection_mode == 'ucb':
                # Update ucb_count_scores only if score_delta is propagated (ucb_update_enabled was True)
                self.ucb_count_scores[i].data += score_delta.data

        x = self.norm(x)
        logits = self.head(x[:, 0])

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits
    
    def get_top_k_patch_indices(self, keep_ratio: float):
        """
        Analyzes the learned ucb_count_scores to find the most important patch indices.
        """
        patch_scores = self.ucb_count_scores[:, :, 1:].mean(dim=(0, 1))
        
        if keep_ratio <= 0:
            return torch.tensor([0], device=patch_scores.device)
        
        num_patches_to_keep = max(1, int(patch_scores.shape[0] * keep_ratio))
        
        top_patch_indices = torch.topk(patch_scores, k=num_patches_to_keep, dim=-1).indices
        
        token_indices = top_patch_indices + 1
        
        cls_token_index = torch.tensor([0], device=token_indices.device)
        
        final_indices = torch.cat([cls_token_index, token_indices]).sort().values
        
        return final_indices

    def forward_pruned(self, pixel_values: torch.Tensor, top_k_indices: torch.Tensor, labels: torch.Tensor = None):
        x = pixel_values
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        x = torch.index_select(x, dim=1, index=top_k_indices.to(x.device))

        for i, block in enumerate(self.blocks):
            x, _ = block(
                x, 
                counter=99999, # High counter to ensure no warm-up
                pruning_enabled=False,
                ucb_count_score=None, # Not used in this path
                selection_mode=self.selection_mode,
                ucb_update_enabled=False # No UCB score update
            )
            
        x = self.norm(x)
        logits = self.head(x[:, 0])
        
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits
