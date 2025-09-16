import torch
import torch.nn as nn
import timm
import random

class RandomAttention(nn.Module):
    """
    Random Attention module with patch pruning - selects patches randomly instead of using UCB.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 k=None, keep_ratio=None, exclude_cls=True, random_seed=None):
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
        self.exclude_cls = exclude_cls
        self.random_seed = random_seed
        
        # Initialize random number generator with seed for reproducibility
        if random_seed is not None:
            self.rng = torch.Generator(device='cuda')
            self.rng.manual_seed(random_seed)
        else:
            self.rng = None

    def random_score_pruning(self, attn_scores, attn_probs, v):
        """
        Applies random patch pruning on attention probabilities.
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

        # Create or move generator to correct device
        if self.rng is not None:
            if self.rng.device != device:
                # Create new generator on correct device with same seed
                self.rng = torch.Generator(device=device)
                if self.random_seed is not None:
                    self.rng.manual_seed(self.random_seed)
        else:
            # Fallback: create device-specific generator
            self.rng = torch.Generator(device=device)
            if self.random_seed is not None:
                self.rng.manual_seed(self.random_seed)

        # Randomly select patches to keep
        mask = torch.zeros(B, H, N, N, device=device, dtype=attn_probs.dtype)
        
        for b in range(B):
            if self.exclude_cls:
                # Randomly select k patches from the non-CLS tokens
                # MODIFIED: Added device=device argument
                patch_indices = torch.randperm(total_patches, generator=self.rng, device=device)[:k]
                token_indices = patch_indices + 1  # Add 1 for CLS offset
                
                # Always preserve CLS token interactions
                mask[b, :, :, 0] = 1.0  # All tokens can attend to CLS
                mask[b, :, 0, :] = 1.0  # CLS can attend to all tokens
                
                # Allow interactions with randomly selected patches
                mask[b, :, :, token_indices] = 1.0  # All can attend TO selected
                mask[b, :, token_indices, :] = 1.0  # Selected can attend to ALL
            else:
                # Randomly select k tokens from all available tokens
                # MODIFIED: Added device=device argument
                selected_indices = torch.randperm(N, generator=self.rng, device=device)[:k]
                mask[b, :, :, selected_indices] = 1.0  # All can attend TO selected
                mask[b, :, selected_indices, :] = 1.0  # Selected can attend to ALL

        pruned_attn = attn_probs * mask
        
        # Renormalize with epsilon to avoid division by zero
        pruned_attn = pruned_attn / (pruned_attn.sum(dim=-1, keepdim=True) + 1e-8)
        
        pruned_attn = self.attn_drop(pruned_attn)

        context = torch.matmul(pruned_attn, v)
        
        return context, None  # No score delta needed for random selection
    
    
    def forward(self, x, counter, random_enabled):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_scores = (q @ k.transpose(-1, -2)) * self.scale
        attn_probs = attn_scores.softmax(dim=-1)

        if random_enabled and counter > 50:  # Warm-up period (same as UCB model)
            context, _ = self.random_score_pruning(attn_scores, attn_probs, v)
        else:
            attn_probs = self.attn_drop(attn_probs)
            context = torch.matmul(attn_probs, v)

        x = context.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RandomBlock(nn.Module):
    def __init__(self, original_block: nn.Module, **random_kwargs):
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

        # Replace attention with Random version
        orig_attn = original_block.attn
        self.attn = RandomAttention(
            dim=orig_attn.qkv.in_features,
            num_heads=orig_attn.num_heads,
            qkv_bias=orig_attn.qkv.bias is not None,
            attn_drop=orig_attn.attn_drop.p if hasattr(orig_attn.attn_drop, 'p') else 0.0,
            proj_drop=orig_attn.proj_drop.p if hasattr(orig_attn.proj_drop, 'p') else 0.0,
            **random_kwargs
        )
        
        # Load weights from original attention
        state_dict = orig_attn.state_dict()
        self.attn.load_state_dict(state_dict, strict=False)

    def forward(self, x, counter, random_enabled):
        # Standard transformer block with residual connections
        h = self.attn(self.norm1(x), counter, random_enabled)
        x = x + self.drop_path1(self.ls1(h))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class ViT_Random_Pruning(nn.Module):
    def __init__(self, model_name="hf-hub:MahmoodLab/uni", pretrained=True, n_classes=None,
                 k=None, keep_ratio=None, exclude_cls=True, random_seed=42):
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

        # Replace blocks with Random versions
        self.blocks = nn.ModuleList([
            RandomBlock(block, k=k, keep_ratio=keep_ratio, exclude_cls=exclude_cls, random_seed=random_seed+i) 
            for i, block in enumerate(source_model.blocks)
        ])

        self.random_seed = random_seed

    def forward(self, x: torch.Tensor, counter: int = 0, random_enabled: bool = True, labels: torch.Tensor = None):
        
        x = self.patch_embed(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = self.pos_drop(x + self.pos_embed)

        # Process through transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(x, counter=counter, random_enabled=random_enabled)

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
        """Returns statistics about current pruning configuration."""
        stats = {
            'num_layers': len(self.blocks),
            'num_heads': self.blocks[0].attn.num_heads,
            'total_patches': self.pos_embed.shape[1],
            'pruning_method': 'random',
            'random_seed': self.random_seed
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
                
        return stats