import torch
import torch.nn as nn
import timm


# ---------------------------------------------------------
# UCB ATTENTION (OPTIMIZED)
# ---------------------------------------------------------
class UCBAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        keep_ratio=None,
        beta=1.0,
        exclude_cls=True,
        ucb_warmup_steps=50,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.exclude_cls = exclude_cls
        self.keep_ratio = keep_ratio
        self.beta = beta
        self.ucb_warmup_steps = ucb_warmup_steps

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, counter, ucb_enabled, ucb_count_scores, return_attn=False):
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_probs = attn.softmax(dim=-1)

        score_delta = None

        # -------------------------------------------------
        # UCB SOFT PRUNING (TRAINING ONLY) - OPTIMIZED
        # -------------------------------------------------
        if (
            ucb_enabled
            and counter >= self.ucb_warmup_steps
            and self.keep_ratio is not None
        ):
            patch_start = 1 if self.exclude_cls else 0
            num_patches = N - patch_start
            k_keep = max(1, int(num_patches * self.keep_ratio))

            patch_scores = attn_probs[:, :, :, patch_start:].mean(dim=2)
            counts = ucb_count_scores[:, patch_start:]

            ucb_bonus = self.beta * torch.sqrt(
                torch.log(torch.tensor(counter + 1.0, device=x.device))
                / (counts + 1e-6)
            )

            ucb_scores = patch_scores + ucb_bonus.unsqueeze(0)
            global_scores = ucb_scores.mean(dim=1)
            _, topk = torch.topk(global_scores, k=k_keep, dim=-1)

            # FIX #3: Vectorized masking (no Python loop)
            mask = torch.zeros_like(attn_probs)

            # CLS always preserved
            if self.exclude_cls:
                mask[:, :, :, 0] = 1.0
                mask[:, :, 0, :] = 1.0
                topk = topk + 1

            # Vectorized scatter instead of loop
            batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, k_keep)
            head_idx = torch.arange(self.num_heads, device=x.device).unsqueeze(0).unsqueeze(2).expand(B, -1, k_keep)
            
            # Fill mask for selected patches (both dimensions)
            for h in range(self.num_heads):
                mask[:, h, :, :].scatter_(2, topk.unsqueeze(1).expand(-1, N, -1), 1.0)
                mask[:, h, :, :].scatter_(1, topk.unsqueeze(2).expand(-1, -1, N), 1.0)

            attn_probs = attn_probs * mask
            attn_probs = attn_probs / (attn_probs.sum(dim=-1, keepdim=True) + 1e-8)

            score_delta = torch.zeros_like(ucb_count_scores)
            score_delta[:, topk.flatten()] += 1.0 / B

        attn_probs_dropped = self.attn_drop(attn_probs)
        x_out = (attn_probs_dropped @ v).transpose(1, 2).reshape(B, N, C)
        x_out = self.proj_drop(self.proj(x_out))

        if return_attn:
            return x_out, score_delta, attn_probs
        return x_out, score_delta


# ---------------------------------------------------------
# UCB BLOCK
# ---------------------------------------------------------
class UCBBlock(nn.Module):
    def __init__(self, block, **ucb_kwargs):
        super().__init__()
        self.norm1 = block.norm1
        self.norm2 = block.norm2
        self.mlp = block.mlp

        self.ls1 = getattr(block, "ls1", nn.Identity())
        self.ls2 = getattr(block, "ls2", nn.Identity())
        self.drop_path1 = getattr(block, "drop_path1", nn.Identity())
        self.drop_path2 = getattr(block, "drop_path2", nn.Identity())

        self.attn = UCBAttention(
            dim=block.attn.qkv.in_features,
            num_heads=block.attn.num_heads,
            qkv_bias=block.attn.qkv.bias is not None,
            attn_drop=block.attn.attn_drop.p,
            proj_drop=block.attn.proj_drop.p,
            **ucb_kwargs,
        )
        self.attn.load_state_dict(block.attn.state_dict(), strict=False)

    def forward(self, x, counter, ucb_enabled, ucb_count_scores, return_attn=False):
        if return_attn:
            h, score_delta, attn_probs = self.attn(
                self.norm1(x), counter, ucb_enabled, ucb_count_scores, return_attn=True
            )
            x = x + self.drop_path1(self.ls1(h))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            return x, score_delta, attn_probs
        else:
            h, score_delta = self.attn(
                self.norm1(x), counter, ucb_enabled, ucb_count_scores, return_attn=False
            )
            x = x + self.drop_path1(self.ls1(h))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            return x, score_delta


# ---------------------------------------------------------
# ViT WITH UCB (OPTIMIZED)
# ---------------------------------------------------------
class ViT_UCB_Pruning(nn.Module):
    def __init__(
        self,
        model_name,
        pretrained,
        n_classes,
        keep_ratio,
        beta=1.0,
        exclude_cls=True,
        input_aware_weight=0.7,
    ):
        super().__init__()

        backbone = timm.create_model(model_name, pretrained=pretrained, init_values=1e-5)
        self.keep_ratio = keep_ratio
        self.input_aware_weight = input_aware_weight

        self.patch_embed = backbone.patch_embed
        self.cls_token = backbone.cls_token
        self.pos_embed = backbone.pos_embed
        self.pos_drop = backbone.pos_drop
        self.norm = backbone.norm

        self.blocks = nn.ModuleList([
            UCBBlock(
                blk,
                keep_ratio=keep_ratio,
                beta=beta,
                exclude_cls=exclude_cls,
            )
            for blk in backbone.blocks
        ])

        embed_dim = backbone.embed_dim
        self.head = nn.Linear(embed_dim, n_classes)

        L = len(self.blocks)
        H = self.blocks[0].attn.num_heads
        N = self.pos_embed.shape[1]

        self.register_buffer(
            "ucb_count_scores", torch.ones(L, H, N)
        )

    def get_top_k_patch_indices(self, keep_ratio):
        """Get top-k patches based on UCB statistics."""
        scores = self.ucb_count_scores[:, :, 1:].mean(dim=(0, 1))
        k = max(1, int(scores.numel() * keep_ratio))
        topk = torch.topk(scores, k).indices + 1
        return torch.cat([torch.tensor([0], device=topk.device), topk]).sort().values

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True
        for block in self.blocks:
            block.gradient_checkpointing = True

    # FIX #2: Optimized with softmax normalization
    def get_input_aware_indices_fast(self, attn_probs, device):
        """Fast input-aware pruning with softmax normalization."""
        # Extract CLS attention (already computed)
        cls_attn = attn_probs[:, :, 0, 1:].mean(dim=(0, 1))
        
        # Get UCB global scores
        global_scores = self.ucb_count_scores[:, :, 1:].mean(dim=(0, 1))
        
        # FIX #2: Softmax instead of min-max (faster + numerically stable)
        cls_attn_norm = torch.softmax(cls_attn, dim=0)
        global_scores_norm = torch.softmax(global_scores, dim=0)
        
        # Combine scores
        alpha = self.input_aware_weight
        combined_scores = alpha * cls_attn_norm + (1 - alpha) * global_scores_norm
        
        # Select top-k
        k_keep = max(1, int(combined_scores.numel() * self.keep_ratio))
        _, topk = torch.topk(combined_scores, k_keep)
        
        return torch.cat([torch.tensor([0], device=device), topk + 1]).sort().values

    def forward(self, x, counter=0, ucb_enabled=True, labels=None):
        # Patch embedding
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        # INFERENCE: INPUT-AWARE PRUNING (OPTIMIZED)
        if not self.training:
            if self.input_aware_weight > 0.0:
                # FIX #1: Single pass through first block
                x, delta, attn_probs = self.blocks[0](
                    x, 
                    counter=counter, 
                    ucb_enabled=False,
                    ucb_count_scores=self.ucb_count_scores[0],
                    return_attn=True  # Get attention for pruning
                )
                
                # Use attention from first pass for pruning
                final_idx = self.get_input_aware_indices_fast(attn_probs, x.device)
                
                # Prune x (already processed by block 0)
                x = torch.index_select(x, dim=1, index=final_idx)
                
                # Process remaining blocks (1-11) with pruned patches
                for i in range(1, len(self.blocks)):
                    x, delta = self.blocks[i](
                        x, 
                        counter=counter, 
                        ucb_enabled=False,
                        ucb_count_scores=self.ucb_count_scores[i]
                    )
                    if delta is not None:
                        with torch.no_grad():
                            self.ucb_count_scores[i] += delta
            else:
                # Pure UCB mode
                final_idx = self.get_top_k_patch_indices(self.keep_ratio)
                x = torch.index_select(x, dim=1, index=final_idx)
                
                for i, blk in enumerate(self.blocks):
                    x, delta = blk(
                        x, 
                        counter=counter, 
                        ucb_enabled=False,
                        ucb_count_scores=self.ucb_count_scores[i]
                    )

        # TRAINING: UCB SOFT PRUNING
        else:
            for i, blk in enumerate(self.blocks):
                x, delta = blk(x, counter, ucb_enabled, self.ucb_count_scores[i])
                if delta is not None:
                    with torch.no_grad():
                        self.ucb_count_scores[i] += delta

        # Classification head
        x = self.norm(x)
        logits = self.head(x[:, 0])

        if labels is not None:
            return nn.CrossEntropyLoss()(logits, labels), logits
        return logits