import torch
import torch.nn as nn
from src.modelling import ViT_UCB_Pruning


class ViT_AttentionPruning(ViT_UCB_Pruning):
    """Pruning based only on first block CLS attention"""
    
    def forward(self, x, labels=None, counter=0, **kwargs):
        # Patch embedding
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        # Attention-based pruning
        if self.keep_ratio < 1.0:
            with torch.no_grad():
                _, _, attn = self.blocks[0](
                    x, counter=0, ucb_enabled=False,
                    ucb_count_scores=self.ucb_count_scores[0],
                    return_attn=True
                )
            
            # CLS attention scores
            cls_attn = attn[:, :, 0, 1:].mean(dim=(0, 1))
            k_keep = max(1, int(cls_attn.numel() * self.keep_ratio))
            _, topk = torch.topk(cls_attn, k_keep)
            keep_idx = torch.cat([torch.tensor([0], device=x.device), topk + 1]).sort().values
            
            x = torch.index_select(x, dim=1, index=keep_idx)
        
        # Process blocks
        for i, blk in enumerate(self.blocks):
            x, _ = blk(x, counter=0, ucb_enabled=False, ucb_count_scores=self.ucb_count_scores[i])
        
        x = self.norm(x)
        logits = self.head(x[:, 0])
        
        if labels is not None:
            return nn.CrossEntropyLoss()(logits, labels), logits
        return logits
