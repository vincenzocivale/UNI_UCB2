from src.modelling import ViT_UCB_Pruning
import torch
import numpy as np

class ViT_RandomPruning(ViT_UCB_Pruning):
    """Random pruning inheriting UCB structure"""
    
    def __init__(self, model_name, pretrained, n_classes, keep_ratio, exclude_cls=True, seed=42):
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            n_classes=n_classes,
            keep_ratio=keep_ratio,
            beta=1.0,
            exclude_cls=exclude_cls,
            input_aware_weight=0.0
        )
        self.seed = seed
    
    def get_top_k_patch_indices(self, keep_ratio):
        """Override: Random selection instead of UCB"""
        N = self.pos_embed.shape[1]
        num_patches = N - 1
        k_keep = max(1, int(num_patches * keep_ratio))
        
        rng = np.random.RandomState(self.seed)
        indices = rng.choice(num_patches, k_keep, replace=False) + 1
        topk = torch.from_numpy(indices)
        
        return torch.cat([torch.tensor([0]), topk]).sort().values
    
    def get_input_aware_indices_fast(self, attn_probs, device):
        """Override: Random selection (ignore attention)"""
        return self.get_top_k_patch_indices(self.keep_ratio).to(device)