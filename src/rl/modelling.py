import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
import numpy as np
from typing import Tuple, Optional

# Suppress timm warnings about dynamic image size
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Using 'dynamic_img_size' with a static model is not recommended.*")

class UCBAttention(nn.Module):
    """
    Custom Attention module with UCB-based patch pruning.
    It is designed to be a drop-in replacement for timm's Attention module,
    allowing weight loading from a pre-trained model.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def ucb_score_pruning(self, attention_scores, v, beta, iteration, count_score_buffer, k=25):
        """
        Applies UCB-based pruning to the attention scores.
        
        Args:
            attention_scores: Raw attention scores (B, H, N, N).
            v: Value tensor (B, H, N, D_head).
            beta: Exploration factor for UCB.
            iteration: Current training iteration/step.
            count_score_buffer: Accumulated selection counts for each attention head (H, N, N).
            k: The number of top patches to keep.

        Returns:
            Tuple of (pruned context layer, score delta to update buffer).
        """
        # UCB calculation
        # Add a small epsilon to avoid division by zero or log(0)
        ucb_exploration = beta * torch.sqrt(
        torch.log(torch.tensor(iteration, device=v.device)) / (count_score_buffer.unsqueeze(0) + 1e-6)
        )
        ucb_scores = attention_scores + ucb_exploration

        # Select top-k patches based on UCB scores
        _, top_indices = torch.topk(ucb_scores, k=k, dim=-1, sorted=False) # shape: (B, H, N, k)


        mask = torch.zeros(
            attention_scores.shape,
            dtype=attention_scores.dtype, # O torch.bool, se preferisci, poi converti a float
            device=attention_scores.device
        )

        # 1. Crea i tensori di indici per B, H, N
        B, H, N, _ = attention_scores.shape
        batch_indices = torch.arange(B, device=attention_scores.device).view(B, 1, 1, 1).expand(-1, H, N, k)
        head_indices = torch.arange(H, device=attention_scores.device).view(1, H, 1, 1).expand(B, -1, N, k)
        query_indices = torch.arange(N, device=attention_scores.device).view(1, 1, N, 1).expand(B, H, -1, k)

        # Usa i top_indices per la dimensione finale
        # mask ha shape (B, H, N, N)
        # Stiamo impostando 1.0 per mask[batch_idx, head_idx, query_idx, top_patch_idx]
        mask[batch_indices, head_indices, query_indices, top_indices] = 1.0

        # Apply mask to original attention scores and re-normalize
        pruned_attn = attention_scores * mask
        # Aggiusta la rinormalizzazione per gestire righe completamente mascherate (tutti zeri)
        # Somma lungo l'ultima dimensione (N) e mantieni la dimensione per broadcasting
        sum_pruned_attn = pruned_attn.sum(dim=-1, keepdim=True)

        pruned_attn = pruned_attn / (sum_pruned_attn + 1e-8)

        context = torch.matmul(pruned_attn, v)

        # Calculate the update for the count_score_buffer (detached from graph)
        # We sum over the batch dimension to get the total counts for this step
        score_delta = mask.sum(dim=0).detach() # mask è (B, H, N, N) -> sum(dim=0) -> (H, N, N)

        return context, score_delta


    def forward(self, x, counter, ucb_enabled, ucb_count_score):
        B, N, C = x.shape
        # Get Q, K, V from a single linear layer
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Calculate raw attention scores
        attn_scores = (q @ k.transpose(-1, -2)) * self.scale
        
        score_delta = None
        # Conditionally apply UCB pruning
        if ucb_enabled and counter > 500: # Pruning starts after 500 iterations
            # NOTE: You can adjust the `beta` and `k` parameters here
            context, score_delta = self.ucb_score_pruning(
                attn_scores, v, beta=1.0, iteration=counter, count_score_buffer=ucb_count_score, k=100
            )
            # Apply standard dropout to the pruned attention-derived context
            context = self.attn_drop(context) 
        else:
            # Standard attention path
            attn_probs = attn_scores.softmax(dim=-1)
            attn_probs = self.attn_drop(attn_probs)
            context = torch.matmul(attn_probs, v)

        # Reshape and project output
        x = context.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, score_delta

class UCBBlock(nn.Module):
    """
    Custom Transformer Block that uses the UCBAttention module.
    """
    def __init__(self, original_block: nn.Module):
        super().__init__()
        # Copy layers from the original pre-trained block
        self.norm1 = original_block.norm1
        self.ls1 = original_block.ls1
        self.drop_path1 = original_block.drop_path1
        self.norm2 = original_block.norm2
        self.mlp = original_block.mlp
        self.ls2 = original_block.ls2
        self.drop_path2 = original_block.drop_path2
        
        # Create and load weights into the custom UCBAttention module
        orig_attn = original_block.attn
        self.attn = UCBAttention(
            dim=orig_attn.qkv.in_features,
            num_heads=orig_attn.num_heads,
            qkv_bias=orig_attn.qkv.bias is not None,
            attn_drop=orig_attn.attn_drop.p,
            proj_drop=orig_attn.proj_drop.p
        )
        self.attn.load_state_dict(orig_attn.state_dict())

    def forward(self, x, counter, ucb_enabled, ucb_count_score):
        # Attention path
        h, score_delta = self.attn(self.norm1(x), counter, ucb_enabled, ucb_count_score)
        x = x + self.drop_path1(self.ls1(h))
        
        # MLP path
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, score_delta

class ViT_UCB_Pruning(nn.Module):
    """
    Main Vision Transformer model with UCB pruning capability.
    It builds the model with custom blocks and loads weights from a specified
    timm model from Hugging Face.
    """
    def __init__(self, model_name="hf-hub:MahmoodLab/uni", pretrained=True, n_classes=None):
        super().__init__()
        
        # 1. Load the pre-trained model to source weights and config
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
        self.head = nn.Linear(1024, n_classes)

        print(f'Patch: {self.patch_embed}',
              'CLS Token: {self.cls_token.shape}, '
              f'Pos Embed: {self.pos_embed.shape}, Norm: {self.norm}, Head: {self.head}')

        self.n_classes = n_classes if n_classes is not None else source_model.head.in_features
        
        
        self.blocks = nn.ModuleList([UCBBlock(block) for block in source_model.blocks])
        
       
        num_layers = len(self.blocks)
        num_heads = self.blocks[0].attn.num_heads
        num_patches = self.pos_embed.shape[1] 
        
        self.register_buffer(
            "ucb_count_scores", 
            torch.zeros(num_layers, num_heads, num_patches, num_patches)
        )
        
        
    def forward(self, x: torch.Tensor, counter: int, ucb_enabled: bool = True, labels: torch.Tensor = None):
        # (The existing forward logic remains the same)
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        
        for i, block in enumerate(self.blocks):
            x, score_delta = block(
                x, 
                counter=counter, 
                ucb_enabled=ucb_enabled,
                ucb_count_score=self.ucb_count_scores[i]
            )
            if score_delta is not None:
                self.ucb_count_scores[i].add_(score_delta.to(self.ucb_count_scores[i].device))

        
        x = self.norm(x)
        logits = self.head(x[:, 0])
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(logits.reshape(-1, self.n_classes), labels.view(-1))
            return (loss, logits) # Return loss and logits as a tuple

        return logits



class StandardViT(nn.Module):
    """
    Standard Vision Transformer classifier.

    This model loads a pretrained Vision Transformer from timm, replaces the
    head with a new classifier, and uses the standard attention mechanism
    without UCB pruning for direct performance comparison.
    """
    def __init__(self, model_name: str = "hf-hub:MahmoodLab/uni", pretrained: bool = True, n_classes: int = 2):
        """
        Initializes the StandardViT model.

        Args:
            model_name (str): The model name to load from timm's hub.
            pretrained (bool): Whether to load pretrained weights.
            n_classes (int): The number of output classes for the new head.
        """
        super().__init__()
        
        # Load the specified pretrained model from timm
        source_model = timm.create_model(
            model_name, 
            pretrained=pretrained,
            init_values=1e-5  # LayerScale initialization value
        )
        
        # --- Inherit core layers from the pretrained model ---
        self.patch_embed = source_model.patch_embed
        self.cls_token = source_model.cls_token
        self.pos_embed = source_model.pos_embed
        self.pos_drop = source_model.pos_drop
        self.norm = source_model.norm
        
        # The transformer blocks are used directly from the source model
        self.blocks = source_model.blocks
        
        # --- Create a new classification head ---
        # Get the number of input features from the original model's head
        num_in_features = source_model.embed_dim 
        self.head = nn.Linear(num_in_features, n_classes)
        self.n_classes = n_classes

        print(f"Model '{model_name}' loaded. New classification head with {n_classes} classes created.")

    def forward(self, x, labels=None, counter=None, ucb_enabled=False):
        """
        Forward pass for the standard Vision Transformer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            labels (Optional[torch.Tensor]): Optional labels for loss calculation.

        Returns:
            If labels are provided, returns a tuple of (loss, logits).
            Otherwise, returns only the logits tensor.
        """
        # 1. Get patch embeddings and add CLS token and positional embeddings
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        
        # 2. Pass through the transformer blocks
        x = self.blocks(x)
        
        # 3. Final normalization and classification head
        x = self.norm(x)
        # Use the output corresponding to the CLS token for classification
        logits = self.head(x[:, 0])
        
        # 4. Calculate loss if labels are provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.n_classes), labels.view(-1))
            return (loss, logits)

        return logits
