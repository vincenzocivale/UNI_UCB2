
import torch
import torch.nn as nn
import timm
import numpy as np

class UCBAttention(nn.Module):
    """
    Attention module with patch pruning capabilities (UCB or random).
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

    def _get_k(self, N):
        if self.exclude_cls:
            total_patches = N - 1
        else:
            total_patches = N

        if self.keep_ratio is not None:
            if self.keep_ratio <= 0:
                return 0
            return max(1, int(total_patches * self.keep_ratio))
        elif self.k is not None:
            return min(self.k, total_patches)
        return total_patches

    def random_pruning(self, attn_probs, v):
        """
        Applies random patch pruning.
        """
        B, H, N, _ = attn_probs.shape
        device = v.device
        k = self._get_k(N)

        if k >= (N - 1 if self.exclude_cls else N):
            return torch.matmul(self.attn_drop(attn_probs), v), None

        # Randomly select indices for each item in the batch
        patch_indices = np.arange(1, N) if self.exclude_cls else np.arange(N)
        selected_indices = torch.tensor(
            [np.random.choice(patch_indices, k, replace=False) for _ in range(B)],
            device=device,
            dtype=torch.long
        )

        mask = torch.zeros(B, H, N, N, device=device, dtype=attn_probs.dtype)
        if self.exclude_cls:
            mask[:, :, :, 0] = 1.0
            mask[:, :, 0, :] = 1.0
            token_indices = selected_indices
            for b in range(B):
                mask[b, :, :, token_indices[b]] = 1.0
                mask[b, :, token_indices[b], :] = 1.0
        else:
            for b in range(B):
                mask[b, :, :, selected_indices[b]] = 1.0
                mask[b, :, selected_indices[b], :] = 1.0
        
        pruned_attn = attn_probs * mask
        pruned_attn = pruned_attn / (pruned_attn.sum(dim=-1, keepdim=True) + 1e-8)
        context = torch.matmul(self.attn_drop(pruned_attn), v)
        return context, None # No score delta for random pruning

    def ucb_score_pruning(self, attn_scores, attn_probs, v, iteration, count_score_buffer):
        """
        Applies UCB-based patch pruning on attention probabilities.
        """
        B, H, N, _ = attn_scores.shape
        device = v.device
        k = self._get_k(N)

        if k >= (N - 1 if self.exclude_cls else N):
            return torch.matmul(self.attn_drop(attn_probs), v), None

        if k == 0 and self.exclude_cls:
            mask = torch.zeros(B, H, N, N, device=device, dtype=attn_probs.dtype)
            mask[:, :, :, 0] = 1.0
            mask[:, :, 0, :] = 1.0
            
            pruned_attn = attn_probs * mask
            pruned_attn = pruned_attn / (pruned_attn.sum(dim=-1, keepdim=True) + 1e-8)
            pruned_attn = self.attn_drop(pruned_attn)
            context = torch.matmul(pruned_attn, v)
            
            score_delta = torch.zeros_like(count_score_buffer, dtype=torch.float32)
            score_delta[:, 0] += 1.0
            return context, score_delta

        if self.exclude_cls:
            patch_scores = attn_probs[:, :, :, 1:].mean(dim=2)
            relevant_counts = count_score_buffer[:, 1:]
        else:
            patch_scores = attn_probs.mean(dim=2)
            relevant_counts = count_score_buffer

        ucb_exploration = self.beta * torch.sqrt(
            torch.log(torch.tensor(iteration + 1.0, device=device)) / (relevant_counts + 1e-6)
        )

        ucb_scores = patch_scores + ucb_exploration.unsqueeze(0)
        global_ucb_scores = ucb_scores.mean(dim=1)
        _, selected_indices = torch.topk(global_ucb_scores, k=k, dim=-1)

        mask = torch.zeros(B, H, N, N, device=device, dtype=attn_probs.dtype)
        
        if self.exclude_cls:
            mask[:, :, :, 0] = 1.0
            mask[:, :, 0, :] = 1.0
            token_indices = selected_indices + 1
            for b in range(B):
                mask[b, :, :, token_indices[b]] = 1.0
                mask[b, :, token_indices[b], :] = 1.0
        else:
            for b in range(B):
                mask[b, :, :, selected_indices[b]] = 1.0
                mask[b, :, selected_indices[b], :] = 1.0

        pruned_attn = attn_probs * mask
        pruned_attn = pruned_attn / (pruned_attn.sum(dim=-1, keepdim=True) + 1e-8)
        pruned_attn = self.attn_drop(pruned_attn)
        context = torch.matmul(pruned_attn, v)
        
        score_delta = torch.zeros_like(count_score_buffer, dtype=torch.float32)
        
        if self.exclude_cls:
            for b in range(B):
                indices_to_update = (selected_indices[b] + 1).long()
                score_delta[:, indices_to_update] += 1.0 / B
        else:
            for b in range(B):
                indices_to_update = selected_indices[b].long()
                score_delta[:, indices_to_update] += 1.0 / B

        return context, score_delta
    
    def forward(self, x, counter, pruning_enabled, ucb_count_score, selection_mode='ucb'):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_scores = (q @ k.transpose(-1, -2)) * self.scale
        attn_probs = attn_scores.softmax(dim=-1)

        score_delta = None
        if pruning_enabled and counter > 50: # Warm-up period
            if selection_mode == 'ucb':
                context, score_delta = self.ucb_score_pruning(
                    attn_scores, attn_probs, v, iteration=counter, count_score_buffer=ucb_count_score
                )
            elif selection_mode == 'random':
                context, score_delta = self.random_pruning(attn_probs, v)
            else:
                raise ValueError(f"Unknown selection mode: {selection_mode}")
        else:
            attn_probs = self.attn_drop(attn_probs)
            context = torch.matmul(attn_probs, v)

        x = context.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, score_delta

class UCBEncoderBlock(nn.Module):
    def __init__(self, original_block: nn.Module, **ucb_kwargs):
        super().__init__()
        self.norm1 = original_block.norm1
        self.norm2 = original_block.norm2
        self.mlp = original_block.mlp
        
        self.ls1 = getattr(original_block, 'ls1', nn.Identity())
        self.ls2 = getattr(original_block, 'ls2', nn.Identity())
        self.drop_path1 = getattr(original_block, 'drop_path1', nn.Identity())
        self.drop_path2 = getattr(original_block, 'drop_path2', nn.Identity())

        orig_attn = original_block.attn
        self.attn = UCBAttention(
            dim=orig_attn.qkv.in_features,
            num_heads=orig_attn.num_heads,
            qkv_bias=orig_attn.qkv.bias is not None,
            attn_drop=orig_attn.attn_drop.p if hasattr(orig_attn.attn_drop, 'p') else 0.0,
            proj_drop=orig_attn.proj_drop.p if hasattr(orig_attn.proj_drop, 'p') else 0.0,
            **ucb_kwargs
        )
        
        state_dict = orig_attn.state_dict()
        self.attn.load_state_dict(state_dict, strict=False)

    def forward(self, x, counter, pruning_enabled, ucb_count_score, selection_mode):
        h, score_delta = self.attn(self.norm1(x), counter, pruning_enabled, ucb_count_score, selection_mode)
        x = x + self.drop_path1(self.ls1(h))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, score_delta

class VisionTransformerUCB(nn.Module):
    def __init__(self, model_name="hf-hub:MahmoodLab/uni", pretrained=True, n_classes=None,
                 k=None, keep_ratio=None, beta=1.0, exclude_cls=True, selection_mode='ucb'):
        super().__init__()
        print(f"Loading source model '{model_name}'...")
        source_model = timm.create_model(model_name, pretrained=pretrained, init_values=1e-5)

        self.keep_ratio = keep_ratio
        self.selection_mode = selection_mode
        self.patch_embed = source_model.patch_embed
        self.cls_token = source_model.cls_token
        self.pos_embed = source_model.pos_embed
        self.pos_drop = source_model.pos_drop
        self.norm = source_model.norm
        
        if hasattr(source_model, 'head'):
            if hasattr(source_model.head, 'in_features'):
                head_dim = source_model.head.in_features
            else:
                head_dim = source_model.head.weight.shape[1] if hasattr(source_model.head, 'weight') else 1024
        else:
            head_dim = source_model.embed_dim if hasattr(source_model, 'embed_dim') else 1024
            
        self.head = nn.Linear(head_dim, n_classes if n_classes is not None else head_dim)
        self.n_classes = n_classes if n_classes is not None else head_dim

        self.blocks = nn.ModuleList([
            UCBEncoderBlock(block, k=k, keep_ratio=keep_ratio, beta=beta, exclude_cls=exclude_cls) 
            for block in source_model.blocks
        ])

        num_layers = len(self.blocks)
        num_heads = self.blocks[0].attn.num_heads
        num_patches = self.pos_embed.shape[1]

        self.register_buffer("ucb_count_scores", torch.ones(num_layers, num_heads, num_patches))

    def forward(self, pixel_values: torch.Tensor, counter: int = 0, pruning_enabled: bool = True, labels: torch.Tensor = None, top_k_indices: torch.Tensor = None):
        if top_k_indices is not None:
             return self.forward_pruned(pixel_values, top_k_indices, labels)
        else:
             return self.forward_dynamic(pixel_values, counter, pruning_enabled, labels)

    def forward_dynamic(self, pixel_values: torch.Tensor, counter: int = 0, pruning_enabled: bool = True, labels: torch.Tensor = None):
        x = pixel_values
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for i, block in enumerate(self.blocks):
            x, score_delta = block(
                x, 
                counter=counter, 
                pruning_enabled=pruning_enabled,
                ucb_count_score=self.ucb_count_scores[i],
                selection_mode=self.selection_mode
            )
            
            if score_delta is not None and self.selection_mode == 'ucb':
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
        """
        patch_scores = self.ucb_count_scores[:, :, 1:].mean(dim=(0, 1))
        
        if keep_ratio <= 0:
            return torch.tensor([0], device=patch_scores.device)
        
        num_patches_to_keep = max(1, int(patch_scores.shape[0] * keep_ratio))
        
        top_patch_indices = torch.topk(patch_scores, k=num_patches_to_keep, dim=-1).indices
        
        token_indices = top_patch_indices + 1
        
        cls_token_index = torch.tensor([0], device=token_indices.device)
        
        final_indices = torch.cat([cls_token_index, token_indices]).sort().values
        
        return final_indices

    def forward_pruned(self, pixel_values: torch.Tensor, top_k_indices: torch.Tensor, labels: torch.Tensor = None):
        x = pixel_values
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        x = torch.index_select(x, dim=1, index=top_k_indices.to(x.device))

        for i, block in enumerate(self.blocks):
            x, _ = block(
                x, 
                counter=99999,
                pruning_enabled=False,
                ucb_count_score=None,
                selection_mode=self.selection_mode
            )
            
        x = self.norm(x)
        logits = self.head(x[:, 0])
        
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits
