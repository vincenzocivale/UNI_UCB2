import torch
import torch.nn as nn
import timm
import numpy as np

class UCBAttention(nn.Module):
    """
    Custom Attention module with UCB-based patch pruning.
    Corrected version.
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

    def ucb_score_pruning(self, attn_scores, v, iteration, count_score_buffer):
        """
        Applies UCB pruning to the attention scores.
        """
        B, H, N_q, N_k = attn_scores.shape

        # Determine dynamic k
        num_keys_to_consider = N_k - 1 if self.exclude_cls else N_k
        if self.keep_ratio is not None:
            k = max(1, int(num_keys_to_consider * self.keep_ratio))
        elif self.k is not None:
            k = min(self.k, num_keys_to_consider)
        else:
            k = num_keys_to_consider
        
        # Handle the case where k is 0 or less
        if k <= 0:
            # Return unpruned attention if there's nothing to keep
            return attn_scores.softmax(dim=-1), None

        # Exclude CLS token scores if required
        if self.exclude_cls:
            scores_for_topk = attn_scores[:, :, :, 1:]
            count_buffer_keys = count_score_buffer[:, 1:]
        else:
            scores_for_topk = attn_scores
            count_buffer_keys = count_score_buffer

        # UCB exploration term is calculated per-key
        ucb_exploration = self.beta * torch.sqrt(
            torch.log(torch.tensor(iteration + 1.0, device=v.device)) / (count_buffer_keys + 1e-6)
        )

        # Correctly broadcast the exploration bonus
        ucb_scores = scores_for_topk + ucb_exploration.unsqueeze(0).unsqueeze(2)

        # Top-k selection
        _, top_indices = torch.topk(ucb_scores, k=k, dim=-1, sorted=False)

        # Create a mask from the top-k indices
        mask = torch.zeros_like(scores_for_topk, dtype=torch.bool)
        mask.scatter_(-1, top_indices, True)

        # Re-attach the CLS token if it was excluded, ensuring it's never pruned
        if self.exclude_cls:
            cls_mask = torch.ones(B, H, N_q, 1, dtype=torch.bool, device=mask.device)
            mask_full = torch.cat([cls_mask, mask], dim=-1)
        else:
            mask_full = mask
            
        # Apply mask and renormalize
        pruned_attn_scores = torch.full_like(attn_scores, -torch.finfo(attn_scores.dtype).max)
        pruned_attn_scores[mask_full] = attn_scores[mask_full]
        pruned_attn_probs = pruned_attn_scores.softmax(dim=-1)

        # score_delta now reflects per-key counts
        score_delta = mask_full.sum(dim=(0, 2)).detach()

        return pruned_attn_probs, score_delta

    def forward(self, x, counter, ucb_enabled, ucb_count_score):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_scores = (q @ k.transpose(-1, -2)) * self.scale
        
        score_delta = None
        if ucb_enabled and counter > 50:
            attn_probs, score_delta = self.ucb_score_pruning(
                attn_scores, v, iteration=counter, count_score_buffer=ucb_count_score,
            )
        else:
            attn_probs = attn_scores.softmax(dim=-1)

        # Apply dropout consistently to attention probabilities
        attn_probs = self.attn_drop(attn_probs)
        context = torch.matmul(attn_probs, v)

        x = context.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, score_delta


class UCBBlock(nn.Module):
    def __init__(self, original_block: nn.Module, **ucb_kwargs):
        super().__init__()
        self.norm1 = original_block.norm1
        self.ls1 = getattr(original_block, 'ls1', nn.Identity())
        self.drop_path1 = original_block.drop_path1
        self.norm2 = original_block.norm2
        self.mlp = original_block.mlp
        self.ls2 = getattr(original_block, 'ls2', nn.Identity())
        self.drop_path2 = original_block.drop_path2

        orig_attn = original_block.attn
        self.attn = UCBAttention(
            dim=orig_attn.qkv.in_features,
            num_heads=orig_attn.num_heads,
            qkv_bias=orig_attn.qkv.bias is not None,
            attn_drop=orig_attn.attn_drop.p,
            proj_drop=orig_attn.proj_drop.p,
            **ucb_kwargs
        )
        self.attn.load_state_dict(orig_attn.state_dict(), strict=False)

    def forward(self, x, counter, ucb_enabled, ucb_count_score):
        h, score_delta = self.attn(self.norm1(x), counter, ucb_enabled, ucb_count_score)
        x = x + self.drop_path1(self.ls1(h))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, score_delta


class ViT_UCB_Pruning(nn.Module):
    def __init__(self, model_name="hf-hub:MahmoodLab/uni", pretrained=True, n_classes=None,
                 k=None, keep_ratio=None, beta=1.0, exclude_cls=True):
        super().__init__()
        print(f"Loading source model '{model_name}'...")
        
        source_model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            init_values=1e-5
        )

        self.patch_embed = source_model.patch_embed
        self.cls_token = source_model.cls_token
        self.pos_embed = source_model.pos_embed
        self.pos_drop = source_model.pos_drop
        self.norm = source_model.norm
        
        in_features = source_model.head.in_features if hasattr(source_model.head, 'in_features') else source_model.num_features
        self.head = nn.Linear(in_features, n_classes) if n_classes is not None else nn.Identity()
        self.n_classes = n_classes if n_classes is not None else in_features

        self.blocks = nn.ModuleList([
            UCBBlock(block, k=k, keep_ratio=keep_ratio, beta=beta, exclude_cls=exclude_cls) for block in source_model.blocks
        ])

        num_layers = len(self.blocks)
        num_heads = self.blocks[0].attn.num_heads
        num_patches = self.pos_embed.shape[1]

        self.register_buffer("ucb_count_scores", torch.zeros(num_layers, num_heads, num_patches))

    def forward(self, x: torch.Tensor, counter: int, ucb_enabled: bool = True, labels: torch.Tensor = None):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for i, block in enumerate(self.blocks):
            x, score_delta = block(x, counter=counter, ucb_enabled=ucb_enabled, ucb_count_score=self.ucb_count_scores[i])
            if score_delta is not None:
                self.ucb_count_scores[i].add_(score_delta)

        x = self.norm(x)
        logits = self.head(x[:, 0])

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.n_classes), labels.view(-1))
            return (loss, logits)
        return logits