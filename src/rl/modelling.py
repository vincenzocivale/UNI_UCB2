import torch
import torch.nn as nn
import timm


# ---------------------------------------------------------
# UCB ATTENTION
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
        # UCB SOFT PRUNING (TRAINING ONLY)
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

            mask = torch.zeros_like(attn_probs)

            # CLS always preserved
            if self.exclude_cls:
                mask[:, :, :, 0] = 1.0
                mask[:, :, 0, :] = 1.0
                topk = topk + 1

            for b in range(B):
                mask[b, :, :, topk[b]] = 1.0
                mask[b, :, topk[b], :] = 1.0

            attn_probs = attn_probs * mask
            attn_probs = attn_probs / (attn_probs.sum(dim=-1, keepdim=True) + 1e-8)

            score_delta = torch.zeros_like(ucb_count_scores)
            for b in range(B):
                score_delta[:, topk[b]] += 1.0 / B

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
# ViT WITH UCB + INPUT-AWARE INFERENCE
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
        input_aware_weight=0.7,  # Weight for input-aware scores (0=pure UCB, 1=pure input)
    ):
        """
        Args:
            input_aware_weight: Balance between input-aware and UCB scores
                               0.0 = pure UCB (fastest, no first-block overhead)
                               1.0 = pure input-aware (slowest, most adaptive)
                               0.7 = balanced (recommended)
        """
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

    # -----------------------------------------------------
    # GLOBAL PRUNING (POST-TRAIN) - UCB Statistics Only
    # -----------------------------------------------------
    def get_top_k_patch_indices(self, keep_ratio):
        """Get top-k patches based on UCB statistics accumulated during training."""
        scores = self.ucb_count_scores[:, :, 1:].mean(dim=(0, 1))
        k = max(1, int(scores.numel() * keep_ratio))
        topk = torch.topk(scores, k).indices + 1
        return torch.cat([torch.tensor([0], device=topk.device), topk]).sort().values

    # -----------------------------------------------------
    # INPUT-AWARE PRUNING (OPTION A)
    # -----------------------------------------------------
    def get_input_aware_indices(self, x):
        """
        Run first block to get input-specific attention, combine with UCB statistics.
        
        Cost: One full forward pass through first block.
        Benefit: Input-adaptive patch selection for remaining 11 blocks.
        """
        with torch.no_grad():
            # Run first block with all patches to get attention weights
            _, _, attn_probs = self.blocks[0](
                x,
                counter=0,
                ucb_enabled=False,
                ucb_count_scores=self.ucb_count_scores[0],
                return_attn=True
            )
            
            # Extract CLS attention to all patches (average over batch and heads)
            # attn_probs: [B, H, N, N]
            cls_attn = attn_probs[:, :, 0, 1:].mean(dim=(0, 1))  # [N-1]
        
        # Get UCB global scores
        global_scores = self.ucb_count_scores[:, :, 1:].mean(dim=(0, 1))  # [N-1]
        
        # Normalize scores to [0, 1] for fair weighting
        cls_attn_norm = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
        global_scores_norm = (global_scores - global_scores.min()) / (global_scores.max() - global_scores.min() + 1e-8)
        
        # Combine: weighted sum
        alpha = self.input_aware_weight
        combined_scores = alpha * cls_attn_norm + (1 - alpha) * global_scores_norm
        
        # Select top-k patches
        k_keep = max(1, int(combined_scores.numel() * self.keep_ratio))
        _, topk = torch.topk(combined_scores, k_keep)
        
        # Return indices (CLS + selected patches)
        final_idx = torch.cat([torch.tensor([0], device=x.device), topk + 1]).sort().values
        return final_idx

    # -----------------------------------------------------
    # FORWARD
    # -----------------------------------------------------
    def forward(self, x, counter=0, ucb_enabled=True, labels=None):
        # Patch embedding
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        # ---------------------------------------------
        # INFERENCE: INPUT-AWARE PRUNING (OPTION A)
        # ---------------------------------------------
        if not self.training:
            if self.input_aware_weight > 0.0:
                # Option A: Run first block, then prune
                # Cost: First block runs on all patches
                # Benefit: Remaining blocks run on pruned patches
                
                # Step 1: Process first block with all patches
                x, delta = self.blocks[0](
                    x, 
                    counter=counter, 
                    ucb_enabled=False,  # No UCB masking during inference
                    ucb_count_scores=self.ucb_count_scores[0]
                )
                
                # Step 2: Get input-aware patch selection
                final_idx = self.get_input_aware_indices(x)
                
                # Step 3: Prune patches
                x = torch.index_select(x, dim=1, index=final_idx)
                
                # Step 4: Process remaining blocks (2-12) with pruned patches
                for i in range(1, len(self.blocks)):
                    x, delta = self.blocks[i](
                        x, 
                        counter=counter, 
                        ucb_enabled=False,  # Already pruned
                        ucb_count_scores=self.ucb_count_scores[i]
                    )
                    if delta is not None:
                        with torch.no_grad():
                            self.ucb_count_scores[i] += delta
            else:
                # Pure UCB mode (input_aware_weight=0): Maximum speed
                final_idx = self.get_top_k_patch_indices(self.keep_ratio)
                x = torch.index_select(x, dim=1, index=final_idx)
                
                # Process all blocks with pruned patches
                for i, blk in enumerate(self.blocks):
                    x, delta = blk(
                        x, 
                        counter=counter, 
                        ucb_enabled=False,
                        ucb_count_scores=self.ucb_count_scores[i]
                    )

        # ---------------------------------------------
        # TRAINING: UCB SOFT PRUNING
        # ---------------------------------------------
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
