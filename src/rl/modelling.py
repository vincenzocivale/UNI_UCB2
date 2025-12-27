import torch
import torch.nn as nn
import timm

class UCBAttention(nn.Module):
    """
    Corrected UCB Attention module with proper patch pruning.
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

    def ucb_score_pruning(self, attn_scores, attn_probs, v, iteration, count_score_buffer):
        """
        Applies UCB-based patch pruning on attention probabilities.
        Fixed to avoid NaN issues while maintaining correct behavior.
        """
        B, H, N, _ = attn_scores.shape
        device = v.device

        # Determine dynamic k (number of patches to keep)
        if self.exclude_cls:
            total_patches = N - 1
            patch_start_idx = 1
        else:
            total_patches = N
            patch_start_idx = 0

        if self.keep_ratio is not None:
            # FIX: Gestisci correttamente keep_ratio=0
            if self.keep_ratio <= 0:
                k = 0  # Prune TUTTE le patch (mantieni solo CLS)
            else:
                k = max(1, int(total_patches * self.keep_ratio))
        elif self.k is not None:
            k = min(self.k, total_patches)
        else:
            # If no pruning specified, return original attention
            return torch.matmul(self.attn_drop(attn_probs), v), None

        # Se k=0, restituisci solo l'attenzione sul CLS token
        if k == 0 and self.exclude_cls:
            # Crea una maschera che mantiene solo il CLS token
            mask = torch.zeros(B, H, N, N, device=device, dtype=attn_probs.dtype)
            mask[:, :, :, 0] = 1.0  # Tutti possono attendere al CLS
            mask[:, :, 0, :] = 1.0  # CLS può attendere a tutti
            
            pruned_attn = attn_probs * mask
            pruned_attn = pruned_attn / (pruned_attn.sum(dim=-1, keepdim=True) + 1e-8)
            pruned_attn = self.attn_drop(pruned_attn)
            context = torch.matmul(pruned_attn, v)
            
            # FIX: Nessun aggiornamento degli score perché nessuna patch è selezionata
            # Ma incrementa comunque il CLS token per tracking
            score_delta = torch.zeros_like(count_score_buffer, dtype=torch.float32)
            score_delta[:, 0] += 1.0  # Solo CLS viene "selezionato"
            return context, score_delta

        # Calculate patch importance scores - using attention received by each patch
        if self.exclude_cls:
            # Average attention each patch receives (excluding CLS)
            patch_scores = attn_probs[:, :, :, 1:].mean(dim=2)  # (B, H, N-1)
            relevant_counts = count_score_buffer[:, 1:]  # (H, N-1)
        else:
            patch_scores = attn_probs.mean(dim=2)  # (B, H, N)
            relevant_counts = count_score_buffer  # (H, N)

        # UCB exploration term
        ucb_exploration = self.beta * torch.sqrt(
            torch.log(torch.tensor(iteration + 1.0, device=device)) / (relevant_counts + 1e-6)
        )

        # Combine scores with exploration bonus
        ucb_scores = patch_scores + ucb_exploration.unsqueeze(0)  # (B, H, num_patches)

        # Global selection (same patches for all heads in each sample)
        global_ucb_scores = ucb_scores.mean(dim=1)  # (B, num_patches)
        _, selected_indices = torch.topk(global_ucb_scores, k=k, dim=-1)  # (B, k)

        # Create attention mask
        mask = torch.zeros(B, H, N, N, device=device, dtype=attn_probs.dtype)
        
        if self.exclude_cls:
            # Always preserve CLS token interactions
            mask[:, :, :, 0] = 1.0  # All tokens can attend to CLS
            mask[:, :, 0, :] = 1.0  # CLS can attend to all tokens
            
            # Convert patch indices to token indices (add 1 for CLS offset)
            token_indices = selected_indices + 1  # (B, k)
            
            for b in range(B):
                mask[b, :, :, token_indices[b]] = 1.0  # All can attend TO selected
                mask[b, :, token_indices[b], :] = 1.0  # Selected can attend to ALL
        else:
            for b in range(B):
                mask[b, :, :, selected_indices[b]] = 1.0  # All can attend TO selected
                mask[b, :, selected_indices[b], :] = 1.0  # Selected can attend to ALL

        pruned_attn = attn_probs * mask
        
        # Renormalize with epsilon to avoid division by zero
        pruned_attn = pruned_attn / (pruned_attn.sum(dim=-1, keepdim=True) + 1e-8)
        
        pruned_attn = self.attn_drop(pruned_attn)

        context = torch.matmul(pruned_attn, v)
        
        # Update counts for selected patches
        # FIX: Assicurati che score_delta sia un tensor float, non int
        score_delta = torch.zeros_like(count_score_buffer, dtype=torch.float32)
        
        if self.exclude_cls:
            for b in range(B):
                # Each selected patch gets +1/B to normalize across batch
                # FIX: Converti selected_indices in long per indexing sicuro
                indices_to_update = (selected_indices[b] + 1).long()
                score_delta[:, indices_to_update] += 1.0 / B
        else:
            for b in range(B):
                indices_to_update = selected_indices[b].long()
                score_delta[:, indices_to_update] += 1.0 / B

        return context, score_delta
    
    
    def forward(self, x, counter, ucb_enabled, ucb_count_score, return_attn_probs=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_scores = (q @ k.transpose(-1, -2)) * self.scale
        attn_probs = attn_scores.softmax(dim=-1)

        score_delta = None
        # FIX: Applica pruning SOLO se ucb_enabled=True E dopo warm-up
        if ucb_enabled and counter > 50:
            context, score_delta = self.ucb_score_pruning(
                attn_scores, attn_probs, v, iteration=counter, count_score_buffer=ucb_count_score
            )
        else:
            # Nessun pruning - attention standard
            attn_probs_processed = self.attn_drop(attn_probs)
            context = torch.matmul(attn_probs_processed, v)

        x = context.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if return_attn_probs:
            return x, score_delta, attn_probs
        return x, score_delta


class UCBBlock(nn.Module):
    def __init__(self, original_block: nn.Module, **ucb_kwargs):
        super().__init__()
        # Copy all components from original block
        self.norm1 = original_block.norm1
        self.norm2 = original_block.norm2
        self.mlp = original_block.mlp
        
        # Handle different timm versions
        self.ls1 = getattr(original_block, 'ls1', nn.Identity())
        self.ls2 = getattr(original_block, 'ls2', nn.Identity())
        self.drop_path1 = getattr(original_block, 'drop_path1', nn.Identity())
        self.drop_path2 = getattr(original_block, 'drop_path2', nn.Identity())

        # Replace attention with UCB version
        orig_attn = original_block.attn
        self.attn = UCBAttention(
            dim=orig_attn.qkv.in_features,
            num_heads=orig_attn.num_heads,
            qkv_bias=orig_attn.qkv.bias is not None,
            attn_drop=orig_attn.attn_drop.p if hasattr(orig_attn.attn_drop, 'p') else 0.0,
            proj_drop=orig_attn.proj_drop.p if hasattr(orig_attn.proj_drop, 'p') else 0.0,
            **ucb_kwargs
        )
        
        # Load weights from original attention
        state_dict = orig_attn.state_dict()
        self.attn.load_state_dict(state_dict, strict=False)

    def forward(self, x, counter, ucb_enabled, ucb_count_score, return_attn_probs=False):
        # Standard transformer block with residual connections
        h_out = self.attn(self.norm1(x), counter, ucb_enabled, ucb_count_score, return_attn_probs)
        
        if return_attn_probs:
            h, score_delta, attn_probs = h_out
        else:
            h, score_delta = h_out
            attn_probs = None # Explicitly set to None if not returned

        x = x + self.drop_path1(self.ls1(h))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        
        if return_attn_probs:
            return x, score_delta, attn_probs
        return x, score_delta


class ViT_UCB_Pruning(nn.Module):
    def __init__(self, model_name="hf-hub:MahmoodLab/uni", pretrained=True, n_classes=None,
                 k=None, keep_ratio=None, beta=1.0, exclude_cls=True, input_aware_extra_tokens: int = 0):
        super().__init__()
        print(f"Loading source model '{model_name}'...")
        source_model = timm.create_model(model_name, pretrained=pretrained, init_values=1e-5)

        self.keep_ratio = keep_ratio

        # Copy patch embedding and positional components
        self.patch_embed = source_model.patch_embed
        self.cls_token = source_model.cls_token
        self.pos_embed = source_model.pos_embed
        self.pos_drop = source_model.pos_drop
        self.norm = source_model.norm
        
        # Setup classification head
        if hasattr(source_model, 'head'):
            if hasattr(source_model.head, 'in_features'):
                head_dim = source_model.head.in_features
            else:
                head_dim = source_model.head.weight.shape[1] if hasattr(source_model.head, 'weight') else 1024
        else:
            head_dim = source_model.embed_dim if hasattr(source_model, 'embed_dim') else 1024
            
        self.head = nn.Linear(head_dim, n_classes if n_classes is not None else head_dim)
        self.n_classes = n_classes if n_classes is not None else head_dim

        # Replace blocks with UCB versions
        self.blocks = nn.ModuleList([
            UCBBlock(block, k=k, keep_ratio=keep_ratio, beta=beta, exclude_cls=exclude_cls) 
            for block in source_model.blocks
        ])

        # Initialize UCB count buffer
        num_layers = len(self.blocks)
        num_heads = self.blocks[0].attn.num_heads
        num_patches = self.pos_embed.shape[1]  # Total tokens including CLS

        # Buffer to track patch selection counts per layer and head
        self.register_buffer("ucb_count_scores", torch.ones(num_layers, num_heads, num_patches))

    def forward(self, x: torch.Tensor, counter: int = 0, ucb_enabled: bool = True, labels: torch.Tensor = None):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        # Initial tokens before any pruning
        initial_x = x
        B, N_full, C = initial_x.shape # N_full includes CLS and all patches
        
        # Determine the set of tokens to process
        # This mechanism is active ONLY during inference and if input_aware_extra_tokens > 0
        # It corrects the global pruning based on input-specific attention.
        # It does NOT update UCB counts.
        if not self.training and self.input_aware_extra_tokens > 0:
            # 1. Perform a single full-token attention pass using ONLY the first transformer block.
            # We need the attention probabilities from the first block's CLS token to patches.
            # No UCB pruning here, no UCB count updates.
            with torch.no_grad(): # Ensure no gradients are tracked for this temporary pass
                # Create a dummy counter and ucb_count_score for this one-off forward call
                # These won't be used as ucb_enabled is False, and no_grad() is active.
                # Use a copy of initial_x to avoid modifying it
                _, _, attn_probs_first_block = self.blocks[0](
                    initial_x, 
                    counter=0, # Counter doesn't matter as ucb_enabled=False
                    ucb_enabled=False, # Disable UCB pruning for this pass
                    ucb_count_score=self.ucb_count_scores[0], # Dummy, won't be updated
                    return_attn_probs=True # Request attention probabilities
                )
            
            # attn_probs_first_block shape: (B, H, N_full, N_full)
            # 2. Use the CLS-to-patch attention from this first block to identify M image-specific patches.
            # CLS token is at index 0, patches start from index 1.
            cls_to_patch_attn = attn_probs_first_block[:, :, 0, 1:] # (B, H, N_full - 1)
            
            # Average attention over heads for each sample in the batch
            avg_cls_to_patch_attn = cls_to_patch_attn.mean(dim=1) # (B, N_full - 1)
            
            # Select top-M patches per image
            M = min(self.input_aware_extra_tokens, N_full - 1) # Ensure M doesn't exceed available patches
            _, input_aware_patch_indices_local = torch.topk(avg_cls_to_patch_attn, k=M, dim=-1) # (B, M)
            
            # Convert local patch indices to global token indices (add 1 for CLS offset)
            input_aware_token_indices = input_aware_patch_indices_local + 1 # (B, M)
            
            # 3. Merge these M patches with the K globally selected patches.
            # Get globally selected patches based on learned UCB scores
            # Use the model's overall keep_ratio to determine K, or a sensible default if not set.
            # We assume a global keep_ratio is set for the model.
            global_keep_ratio = self.keep_ratio if self.keep_ratio is not None else 0.5 # Default global ratio
            globally_pruned_tokens = self.get_top_k_patch_indices(global_keep_ratio) # (K,)
            
            # Initialize final token indices with CLS token for each sample
            final_token_indices_batch = []
            
            for b_idx in range(B):
                # Start with global tokens, ensuring CLS (index 0) is always there
                current_sample_tokens = globally_pruned_tokens.to(x.device).clone()
                
                # Add input-aware tokens for this sample
                current_sample_tokens = torch.cat([current_sample_tokens, input_aware_token_indices[b_idx]])
                
                # Remove duplicates and sort to preserve deterministic order
                current_sample_tokens = torch.unique(current_sample_tokens).sort().values
                
                final_token_indices_batch.append(current_sample_tokens)

            # Pad or truncate to ensure all samples have the same number of tokens
            # For simplicity, we'll assume a fixed size for the output of this process
            # or handle variable length by creating a ragged tensor or iterating,
            # but batch-wise operation is requested.
            # For now, let's select the union for each batch element
            # and then pad them to the max length in the batch.
            max_len = max(idx.numel() for idx in final_token_indices_batch)
            padded_final_token_indices_batch = torch.zeros(B, max_len, dtype=torch.long, device=x.device)
            
            # We fill the tokens and use -1 for padding.
            # Then we can filter these -1s later or ensure our index_select handles it.
            # A simpler way: we'll perform index_select iteratively per batch element
            # and then stack them, if sizes differ. If sizes are uniform, it's easier.
            # The prompt requests "batch-wise operation (no Python loops over batch)".
            # This implies `torch.index_select` which needs uniform indices across batch.
            # So, we should determine a single set of indices to apply to the whole batch.
            # This contradicts "top-M patches *per image*".

            # Re-evaluating: "Merge these M patches with the K globally selected patches."
            # "The final token set = union(global_pruned_tokens, input_aware_tokens)"
            # This implies a single set of tokens to be used for the entire batch.
            # This is simpler and aligns with the batch-wise operation constraint.
            # So, instead of per-image M, I'll select M patches from the *mean* attention over the batch.

            # Re-calculation for single set of tokens across batch
            avg_cls_to_patch_attn_across_batch = avg_cls_to_patch_attn.mean(dim=0) # (N_full - 1)
            _, input_aware_patch_indices_global_single = torch.topk(
                avg_cls_to_patch_attn_across_batch, k=M, dim=-1
            ) # (M,)
            input_aware_token_indices_global_single = input_aware_patch_indices_global_single + 1 # (M,)
            
            # Merge global UCB tokens with global input-aware tokens
            final_token_indices = torch.cat([
                globally_pruned_tokens.to(x.device),
                input_aware_token_indices_global_single
            ])
            final_token_indices = torch.unique(final_token_indices).sort().values
            
            # 4. Continue the forward pass using only the merged token set.
            x = torch.index_select(x, dim=1, index=final_token_indices)
            
            # When input-aware pruning is active, UCB pruning inside blocks is not needed
            # as physical pruning is already applied.
            ucb_enabled_for_blocks = False
            
            # Add comment explaining why this is inference-only
            # This input-aware correction is applied only during inference/evaluation
            # to provide a lightweight, input-specific refinement to the
            # globally learned UCB pruning. It avoids modifying UCB counts or
            # interfering with training dynamics.
        else:
            # If not in evaluation or input_aware_extra_tokens is 0,
            # proceed with standard UCB pruning as configured.
            final_token_indices = None # No initial physical pruning
            ucb_enabled_for_blocks = ucb_enabled # Respect original ucb_enabled flag
        
        # Process through blocks with potentially pruned tokens or standard full tokens
        for i, block in enumerate(self.blocks):
            # If input-aware pruning happened, ucb_enabled_for_blocks is False.
            # Otherwise, it uses the original ucb_enabled flag.
            # In the main loop, we never need to return attn_probs.
            x, score_delta = block(
                x, 
                counter=counter, 
                ucb_enabled=ucb_enabled_for_blocks, # Use the determined ucb_enabled flag
                ucb_count_score=self.ucb_count_scores[i],
                return_attn_probs=False # Always False for the main loop blocks
            )
            
            if score_delta is not None:
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

        Args:
            keep_ratio (float): The ratio of patches to keep (e.g., 0.7 for 70%).

        Returns:
            torch.Tensor: A 1D tensor containing the indices of tokens to keep,
                          including the CLS token.
        """
        # Exclude CLS token from importance calculation
        # Scores are averaged across all layers and heads
        patch_scores = self.ucb_count_scores[:, :, 1:].mean(dim=(0, 1))
        
        # FIX: Gestisci keep_ratio=0
        if keep_ratio <= 0:
            # Ritorna solo il CLS token
            return torch.tensor([0], device=patch_scores.device)
        
        num_patches_to_keep = max(1, int(patch_scores.shape[0] * keep_ratio))
        
        # Get the indices of the patches with the highest scores
        top_patch_indices = torch.topk(patch_scores, k=num_patches_to_keep, dim=-1).indices
        
        # Add 1 to offset for the CLS token we excluded
        token_indices = top_patch_indices + 1
        
        # Always include the CLS token (index 0)
        cls_token_index = torch.tensor([0], device=token_indices.device)
        
        # Concatenate and sort for predictable order
        final_indices = torch.cat([cls_token_index, token_indices]).sort().values
        
        return final_indices

    def forward_pruned(self, x: torch.Tensor, top_k_indices: torch.Tensor):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        print(f"PRUNING: Tensor shape before: {x.shape[1]} tokens")
        x = torch.index_select(x, dim=1, index=top_k_indices.to(x.device))
        print(f"PRUNING: Tensor shape after: {x.shape[1]} tokens")

        # Process the reduced set of tokens through the transformer blocks
        # We disable UCB logic since pruning is already done
        for i, block in enumerate(self.blocks):
            x, _ = block(
                x, 
                counter=99999,
                ucb_enabled=False,  # Disabilita UCB durante inferenza pruned
                ucb_count_score=None
            )
            
        # Final norm and classification (CLS token is still at index 0)
        x = self.norm(x)
        logits = self.head(x[:, 0])
        
        return logits

    def get_pruning_stats(self):
        """Returns statistics about current pruning configuration and state."""
        stats = {
            'num_layers': len(self.blocks),
            'num_heads': self.blocks[0].attn.num_heads,
            'total_patches': self.pos_embed.shape[1],
        }
        
        # Add per-layer pruning configuration
        for i, block in enumerate(self.blocks):
            if hasattr(block.attn, 'keep_ratio') and block.attn.keep_ratio is not None:
                stats[f'layer_{i}_keep_ratio'] = block.attn.keep_ratio
                stats[f'layer_{i}_kept_patches'] = int(
                    (self.pos_embed.shape[1] - (1 if block.attn.exclude_cls else 0)) * block.attn.keep_ratio
                )
            elif hasattr(block.attn, 'k') and block.attn.k is not None:
                total_patches = self.pos_embed.shape[1] - (1 if block.attn.exclude_cls else 0)
                stats[f'layer_{i}_keep_ratio'] = block.attn.k / total_patches
                stats[f'layer_{i}_kept_patches'] = block.attn.k
                
        # Add UCB statistics
        stats['avg_patch_counts'] = self.ucb_count_scores.mean(dim=(0, 1)).cpu().numpy()
        stats['min_patch_count'] = self.ucb_count_scores.min().item()
        stats['max_patch_count'] = self.ucb_count_scores.max().item()
        
        return stats