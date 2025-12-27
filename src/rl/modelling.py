import torch
import torch.nn as nn
import timm
from src.models.vit.pruning import get_global_pruning_indices
from src.models.vit.input_aware import get_input_aware_and_global_token_indices
from src.ucb.scoring import calculate_ucb_selection

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
        Applies UCB-based token selection (pruning) on attention probabilities.
        This function determines which tokens to keep based on UCB scores and then masks
        the attention probabilities to only allow interactions with selected tokens.
        """
        B, H, N, _ = attn_scores.shape
        device = v.device

        # Determine the dynamic number of image tokens to keep (keep_k).
        # `N` is the total sequence length (CLS + image tokens).
        if self.exclude_cls:
            total_image_tokens = N - 1 # Exclude CLS token from the pool of prunable tokens
        else:
            total_image_tokens = N

        if self.keep_ratio is not None:
            if self.keep_ratio <= 0:
                keep_k = 0  # If keep_ratio is 0 or less, no image tokens are kept.
            else:
                keep_k = max(1, int(total_image_tokens * self.keep_ratio))
        elif self.k is not None:
            keep_k = min(self.k, total_image_tokens)
        else:
            # If no pruning configuration is specified, return original attention (no token selection).
            return torch.matmul(self.attn_drop(attn_probs), v), None

        # Special case: if `keep_k` is 0 and CLS token is excluded, only the CLS token remains.
        # Create a mask that allows all tokens to attend to CLS, and CLS to attend to all.
        # This effectively prunes all image tokens.
        if keep_k == 0 and self.exclude_cls:
            mask = torch.zeros(B, H, N, N, device=device, dtype=attn_probs.dtype)
            mask[:, :, :, 0] = 1.0  # All query tokens can attend to the CLS key/value.
            mask[:, :, 0, :] = 1.0  # The CLS query token can attend to all key/value tokens.
            
            pruned_attn = attn_probs * mask
            # Renormalize to ensure attention probabilities sum to 1.
            pruned_attn = pruned_attn / (pruned_attn.sum(dim=-1, keepdim=True) + 1e-8)
            pruned_attn = self.attn_drop(pruned_attn)
            context = torch.matmul(pruned_attn, v)
            
            # Since no image tokens are selected, the score_delta only reflects the CLS token's 'selection'.
            score_delta = torch.zeros_like(count_score_buffer, dtype=torch.float32)
            score_delta[:, 0] += 1.0  # CLS token is implicitly "selected" for tracking purposes.
            return context, score_delta

        # Calculate UCB selection and corresponding score delta using the external utility function.
        # This decouples the UCB scoring logic from attention masking.
        selected_indices, score_delta = calculate_ucb_selection(
            attn_probs=attn_probs,
            iteration=iteration,
            count_score_buffer=count_score_buffer,
            keep_k=keep_k,
            beta=self.beta,
            exclude_cls=self.exclude_cls,
            device=device
        )

        # Create the attention mask based on the `selected_indices` returned by UCB selection.
        mask = torch.zeros(B, H, N, N, device=device, dtype=attn_probs.dtype)
        
        if self.exclude_cls:
            # Always ensure the CLS token interacts with all other tokens it would normally attend to,
            # and that all tokens can attend to the CLS token.
            mask[:, :, :, 0] = 1.0  # All query tokens can attend to the CLS key/value.
            mask[:, :, 0, :] = 1.0  # The CLS query token can attend to all key/value tokens.
            
            for b in range(B):
                # For each sample, allow interactions with the dynamically selected image tokens.
                # `selected_indices` already contains global token indices (including CLS offset if exclude_cls).
                mask[b, :, :, selected_indices[b]] = 1.0  # All query tokens can attend TO selected image token key/values.
                mask[b, :, selected_indices[b], :] = 1.0  # Selected image tokens can attend to ALL key/value tokens.
        else:
            # If CLS is not excluded, selected_indices directly correspond to `N` tokens.
            for b in range(B):
                mask[b, :, :, selected_indices[b]] = 1.0  # All can attend TO selected
                mask[b, :, selected_indices[b], :] = 1.0  # Selected can attend to ALL

        pruned_attn = attn_probs * mask
        
        # Renormalize to ensure attention probabilities sum to 1 after masking.
        pruned_attn = pruned_attn / (pruned_attn.sum(dim=-1, keepdim=True) + 1e-8)
        
        pruned_attn = self.attn_drop(pruned_attn)

        context = torch.matmul(pruned_attn, v)
        
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
            # This input-aware correction is applied only during inference/evaluation
            # to provide a lightweight, input-specific refinement to the
            # globally learned UCB pruning. It avoids modifying UCB counts or
            # interfering with training dynamics.
            
            # Use the external utility function to get the merged token indices
            final_token_indices = get_input_aware_and_global_token_indices(
                initial_x=initial_x,
                first_block=self.blocks[0],
                ucb_count_scores=self.ucb_count_scores,
                pos_embed_shape=self.pos_embed.shape,
                global_keep_ratio=self.keep_ratio if self.keep_ratio is not None else 0.5,
                input_aware_extra_tokens=self.input_aware_extra_tokens,
                get_global_pruning_indices_fn=get_global_pruning_indices # Pass the function
            )
            
            # Apply physical pruning to the token sequence
            x = torch.index_select(x, dim=1, index=final_token_indices)
            
            # When input-aware pruning is active, UCB pruning inside blocks is not needed
            # as physical pruning is already applied.
            ucb_enabled_for_blocks = False
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



    def forward_pruned(self, x: torch.Tensor, top_k_indices: torch.Tensor):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        # Physically prune the tokens based on the provided top_k_indices.
        x = torch.index_select(x, dim=1, index=top_k_indices.to(x.device))

        # Process the reduced set of tokens through the transformer blocks.
        # UCB logic is disabled here since physical pruning has already occurred.
        for i, block in enumerate(self.blocks):
            x, _ = block(
                x, 
                counter=99999, # Dummy value, as ucb_enabled=False
                ucb_enabled=False,  # Disable UCB updates/pruning within blocks
                ucb_count_score=None # Not used when ucb_enabled=False
            )
            
        # Final normalization and classification (CLS token is always at index 0 in the pruned sequence).
        x = self.norm(x)
        logits = self.head(x[:, 0])
        
        return logits

    def get_pruning_stats(self):
        """
        Returns statistics about the current UCB pruning configuration and state.
        This provides insights into the learned token importances.
        """
        stats = {
            'num_layers': len(self.blocks),
            'num_heads': self.blocks[0].attn.num_heads,
            'total_tokens': self.pos_embed.shape[1], # Total tokens including CLS
        }
        
        # Add per-layer pruning configuration based on the attention modules.
        for i, block in enumerate(self.blocks):
            # The `keep_ratio` or `k` attributes determine the pruning configuration.
            if hasattr(block.attn, 'keep_ratio') and block.attn.keep_ratio is not None:
                stats[f'layer_{i}_keep_ratio'] = block.attn.keep_ratio
                # Calculate the number of tokens kept per layer (excluding CLS if specified).
                stats[f'layer_{i}_kept_tokens'] = int(
                    (self.pos_embed.shape[1] - (1 if block.attn.exclude_cls else 0)) * block.attn.keep_ratio
                )
            elif hasattr(block.attn, 'k') and block.attn.k is not None:
                total_tokens_for_k = self.pos_embed.shape[1] - (1 if block.attn.exclude_cls else 0)
                stats[f'layer_{i}_keep_ratio'] = block.attn.k / total_tokens_for_k
                stats[f'layer_{i}_kept_tokens'] = block.attn.k
                
        # Add UCB statistics from the accumulated count scores.
        stats['avg_token_counts'] = self.ucb_count_scores.mean(dim=(0, 1)).cpu().numpy()
        stats['min_token_count'] = self.ucb_count_scores.min().item()
        stats['max_token_count'] = self.ucb_count_scores.max().item()
        
        return stats