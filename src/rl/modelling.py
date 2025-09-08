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
            k = max(1, int(total_patches * self.keep_ratio))
        elif self.k is not None:
            k = min(self.k, total_patches)
        else:
            # If no pruning specified, return original attention
            return torch.matmul(self.attn_drop(attn_probs), v), None

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

        # Create attention mask - USE YOUR ORIGINAL APPROACH (more permissive)
        mask = torch.zeros(B, H, N, N, device=device, dtype=attn_probs.dtype)
        
        if self.exclude_cls:
            # Always preserve CLS token interactions
            mask[:, :, :, 0] = 1.0  # All tokens can attend to CLS
            mask[:, :, 0, :] = 1.0  # CLS can attend to all tokens
            
            # Convert patch indices to token indices (add 1 for CLS offset)
            token_indices = selected_indices + 1  # (B, k)
            
            # Use your original approach: all tokens can attend TO selected patches
            # and selected patches can attend to ALL tokens
            for b in range(B):
                mask[b, :, :, token_indices[b]] = 1.0  # All can attend TO selected
                mask[b, :, token_indices[b], :] = 1.0  # Selected can attend to ALL
        else:
            for b in range(B):
                mask[b, :, :, selected_indices[b]] = 1.0  # All can attend TO selected
                mask[b, :, selected_indices[b], :] = 1.0  # Selected can attend to ALL

        # Apply mask AFTER softmax (like your original) to avoid NaN
        # This is mathematically imperfect but avoids numerical issues
        pruned_attn = attn_probs * mask
        
        # Renormalize with epsilon to avoid division by zero
        pruned_attn = pruned_attn / (pruned_attn.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Apply dropout after pruning
        pruned_attn = self.attn_drop(pruned_attn)
        
        # Compute context with pruned attention
        context = torch.matmul(pruned_attn, v)
        
        # Update counts for selected patches
        score_delta = torch.zeros_like(count_score_buffer)
        
        if self.exclude_cls:
            # Increment counts for selected patches
            for b in range(B):
                # Each selected patch gets +1/B to normalize across batch
                score_delta[:, selected_indices[b] + 1] += 1.0 / B
        else:
            for b in range(B):
                score_delta[:, selected_indices[b]] += 1.0 / B

        return context, score_delta
    
    
    def forward(self, x, counter, ucb_enabled, ucb_count_score):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_scores = (q @ k.transpose(-1, -2)) * self.scale
        attn_probs = attn_scores.softmax(dim=-1)

        score_delta = None
        if ucb_enabled and counter > 50:  # Warm-up period
            context, score_delta = self.ucb_score_pruning(
                attn_scores, attn_probs, v, iteration=counter, count_score_buffer=ucb_count_score
            )
        else:
            attn_probs = self.attn_drop(attn_probs)
            context = torch.matmul(attn_probs, v)

        x = context.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
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

    def forward(self, x, counter, ucb_enabled, ucb_count_score):
        # Standard transformer block with residual connections
        h, score_delta = self.attn(self.norm1(x), counter, ucb_enabled, ucb_count_score)
        x = x + self.drop_path1(self.ls1(h))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, score_delta


class ViT_UCB_Pruning(nn.Module):
    def __init__(self, model_name="hf-hub:MahmoodLab/uni", pretrained=True, n_classes=None,
                 k=None, keep_ratio=None, beta=1.0, exclude_cls=True):
        super().__init__()
        print(f"Loading source model '{model_name}'...")
        source_model = timm.create_model(model_name, pretrained=pretrained, init_values=1e-5)

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
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = self.pos_drop(x + self.pos_embed)

        # Process through transformer blocks
        for i, block in enumerate(self.blocks):
            x, score_delta = block(
                x, 
                counter=counter, 
                ucb_enabled=ucb_enabled, 
                ucb_count_score=self.ucb_count_scores[i]
            )
            
            # Update counts if pruning was applied
            if score_delta is not None:
                self.ucb_count_scores[i] += score_delta

        # Final norm and classification
        x = self.norm(x)
        logits = self.head(x[:, 0])  # Use CLS token for classification

        # Compute loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
            
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