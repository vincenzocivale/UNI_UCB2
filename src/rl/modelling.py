import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
import numpy as np
from typing import Tuple, Optional
import wandb

# Suppress timm warnings about dynamic image size
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Using 'dynamic_img_size' with a static model is not recommended.*")

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class UCBAttention(nn.Module):
    """
    Custom Attention module with UCB-based patch pruning (aggiornata).
    Compatibile con timm.
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
        Applica il pruning UCB sulle attention probabilities.
        """
        B, H, N, _ = attn_scores.shape

        # Determina k dinamico
        if self.keep_ratio is not None:
            k = max(1, int((N - 1 if self.exclude_cls else N) * self.keep_ratio))
        elif self.k is not None:
            k = min(self.k, N - (1 if self.exclude_cls else 0))
        else:
            k = N

        # Escludi CLS se richiesto
        if self.exclude_cls:
            scores_for_topk = attn_scores[:, :, :, 1:]
        else:
            scores_for_topk = attn_scores

        # Calcolo UCB
        ucb_exploration = self.beta * torch.sqrt(
        torch.log(torch.tensor(iteration + 1.0, device=v.device)) / (count_score_buffer + 1e-6)
        )  # shape (H, N, N)

        if self.exclude_cls:
            ucb_exploration = ucb_exploration[:, :, 1:]  

        ucb_scores = scores_for_topk + ucb_exploration.unsqueeze(0)  # broadcast su batch
        ucb_exploration = ucb_exploration[:, : scores_for_topk.shape[-1]]
        ucb_scores = scores_for_topk + ucb_exploration.unsqueeze(0).unsqueeze(0)

        # Top-k
        _, top_indices = torch.topk(ucb_scores, k=k, dim=-1, sorted=False)

        mask = torch.zeros_like(scores_for_topk, dtype=attn_scores.dtype)
        B, H, Q, _ = scores_for_topk.shape
        batch_indices = torch.arange(B, device=v.device).view(B, 1, 1, 1).expand(-1, H, Q, k)
        head_indices = torch.arange(H, device=v.device).view(1, H, 1, 1).expand(B, -1, Q, k)
        query_indices = torch.arange(Q, device=v.device).view(1, 1, Q, 1).expand(B, H, -1, k)

        mask[batch_indices, head_indices, query_indices, top_indices] = 1.0

        if self.exclude_cls:
            mask_full = torch.cat([torch.ones_like(mask[:, :, :, :1]), mask], dim=-1)
        else:
            mask_full = mask

        # Applica mask a attn_probs e rinormalizza
        pruned_attn = attn_probs * mask_full
        pruned_attn = pruned_attn / (pruned_attn.sum(dim=-1, keepdim=True) + 1e-8)

        context = torch.matmul(pruned_attn, v)
        score_delta = mask_full.sum(dim=0).detach()

        return context, score_delta

    def forward(self, x, counter, ucb_enabled, ucb_count_score):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_scores = (q @ k.transpose(-1, -2)) * self.scale
        attn_probs = attn_scores.softmax(dim=-1)

        score_delta = None
        if ucb_enabled and counter > 50:
            context, score_delta = self.ucb_score_pruning(
                attn_scores, attn_probs, v, iteration=counter, count_score_buffer=ucb_count_score, 
            )
            context = self.attn_drop(context)
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
        self.norm1 = original_block.norm1
        self.ls1 = original_block.ls1
        self.drop_path1 = original_block.drop_path1
        self.norm2 = original_block.norm2
        self.mlp = original_block.mlp
        self.ls2 = original_block.ls2
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
        source_model = timm.create_model(model_name, pretrained=pretrained, init_values=1e-5)

        self.patch_embed = source_model.patch_embed
        self.cls_token = source_model.cls_token
        self.pos_embed = source_model.pos_embed
        self.pos_drop = source_model.pos_drop
        self.norm = source_model.norm
        self.head = nn.Linear(source_model.head.in_features if hasattr(source_model.head, 'in_features') else 1024, n_classes)

        self.n_classes = n_classes if n_classes is not None else source_model.head.in_features
        self.blocks = nn.ModuleList([
            UCBBlock(block, k=k, keep_ratio=keep_ratio, beta=beta, exclude_cls=exclude_cls) for block in source_model.blocks
        ])

        num_layers = len(self.blocks)
        num_heads = self.blocks[0].attn.num_heads
        num_patches = self.pos_embed.shape[1]

        self.register_buffer("ucb_count_scores", torch.zeros(num_layers, num_heads, num_patches, num_patches))

    def forward(self, pixel_values: torch.Tensor, counter: int, ucb_enabled: bool = True, labels: torch.Tensor = None):
        # Il nome 'x' diventa 'pixel_values'
        x = self.patch_embed(pixel_values)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for i, block in enumerate(self.blocks):
            x, score_delta = block(x, counter=counter, ucb_enabled=ucb_enabled, ucb_count_score=self.ucb_count_scores[i])
            if score_delta is not None:
                self.ucb_count_scores[i].add_(score_delta.to(self.ucb_count_scores[i].device))

        x = self.norm(x)
        logits = self.head(x[:, 0])

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.n_classes), labels.view(-1))
            return (loss, logits)
        return logits
