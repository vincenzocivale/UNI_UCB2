import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
import numpy as np
from typing import Tuple

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

        # -- THE FIX IS HERE --
        # Add init_values to enable LayerScale, matching the pretrained model's architecture.
        # The value 1e-5 is a common default for LayerScale initialization.
        source_model = timm.create_model(
            model_name, 
            pretrained=pretrained,
            init_values=1e-5  # <--- ADD THIS ARGUMENT
        )
        
        # 2. Copy essential non-block layers
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
        
        # 3. Create custom UCB blocks and load weights
        self.blocks = nn.ModuleList([UCBBlock(block) for block in source_model.blocks])
        
        # 4. Initialize UCB count score buffers
        num_layers = len(self.blocks)
        num_heads = self.blocks[0].attn.num_heads
        num_patches = self.pos_embed.shape[1] 
        
        self.register_buffer(
            "ucb_count_scores", 
            torch.zeros(num_layers, num_heads, num_patches, num_patches)
        )
        
        print("✅ ViT with UCB Pruning successfully initialized.")
        print(f"   - Layers: {num_layers}, Heads: {num_heads}, Patches: {num_patches}")
        
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
        
        # NEW: Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(logits.reshape(-1, self.n_classes), labels.view(-1))
            return (loss, logits) # Return loss and logits as a tuple

        return logits




# # coding=utf-8
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import copy
# import logging
# import math

# from os.path import join as pjoin
# import torch.nn.functional as F
# import torch
# import torch.nn as nn
# import numpy as np

# from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
# from torch.nn.modules.utils import _pair
# from scipy import ndimage

# from .modeling_resnet import ResNetV2

# logger = logging.getLogger(__name__)

# ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
# ATTENTION_K = "MultiHeadDotProductAttention_1/key"
# ATTENTION_V = "MultiHeadDotProductAttention_1/value"
# ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
# FC_0 = "MlpBlock_3/Dense_0"
# FC_1 = "MlpBlock_3/Dense_1"
# ATTENTION_NORM = "LayerNorm_0"
# MLP_NORM = "LayerNorm_2"

# def np2th(weights, conv=False):
#     """Possibly convert HWIO to OIHW."""
#     if conv:
#         weights = weights.transpose([3, 2, 0, 1])
#     return torch.from_numpy(weights)

# def swish(x):
#     return x * torch.sigmoid(x)

# ACT2FN = {
#     "gelu": torch.nn.functional.gelu,
#     "relu": torch.nn.functional.relu,
#     "swish": swish,
# }

# block_size = (224 * 224) // (16 * 16) + 1

# class Attention(nn.Module):
#     def __init__(self, config, vis):
#         super(Attention, self).__init__()
#         self.vis = vis
#         self.num_attention_heads = config.transformer["num_heads"]
#         self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size

#         self.query = Linear(config.hidden_size, self.all_head_size)
#         self.key = Linear(config.hidden_size, self.all_head_size)
#         self.value = Linear(config.hidden_size, self.all_head_size)
#         self.out = Linear(config.hidden_size, config.hidden_size)
        
#         self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
#         self.proj_dropout = Dropout(config.transformer["dropout_rate"])
#         self.softmax = Softmax(dim=-1)
        
#         # Parametri UCB
#         self.ucb_beta = 1.0  # Valore fisso come nell'articolo
#         self.ucb_activation_step = 1000000
#         self.ucb_top_k = 80  # Top-k fisso
        
#         # Per-batch count buffers
#         self.count_buffer_size = 512  # Max sequence length
#         self.count_decay = 0.95  # Decay factor for count buffer

#     def initialize_count_buffer(self, batch_size, seq_len, device):
#         """Initialize per-batch count buffers"""
#         return torch.ones(batch_size, seq_len, seq_len, device=device) * 0.1

#     def update_count_buffer(self, count_buffer, selection_mask):
#         """Update count buffer with decay"""
#         # Decay existing counts
#         count_buffer.mul_(self.count_decay)
#         # Add new selections
#         count_buffer.add_(selection_mask.mean(dim=1))  # Average across heads
#         return count_buffer

#     def compute_ucb_scores(self, attention_scores, count_buffer, step):
#         """Compute UCB scores with proper exploration-exploitation balance"""
#         batch_size, num_heads, seq_len, _ = attention_scores.shape
        
#         # Decaying beta for reducing exploration over time
#         current_beta = self.ucb_beta_initial * (self.ucb_beta_decay ** step)
        
#         # Compute confidence intervals
#         log_t = torch.log(torch.tensor(float(step), device=attention_scores.device))
#         confidence = current_beta * torch.sqrt(log_t / (count_buffer + 1e-8))
        
#         # Expand confidence to match attention dimensions
#         confidence = confidence.unsqueeze(1).expand(-1, num_heads, -1, -1)
        
#         # Add confidence to raw attention scores (before softmax)
#         ucb_scores = attention_scores + confidence
        
#         return ucb_scores, current_beta

#     def apply_ucb_selection(self, attention_scores, count_buffer, step):
#         """Apply UCB-based patch selection"""
#         batch_size, num_heads, seq_len, _ = attention_scores.shape
        
#         # Compute UCB scores
#         ucb_scores, current_beta = self.compute_ucb_scores(attention_scores, count_buffer, step)
        
#         # Apply softmax to UCB scores
#         ucb_probs = self.softmax(ucb_scores)
        
#         # Top-p selection
#         sorted_probs, sorted_indices = torch.sort(ucb_probs, dim=-1, descending=True)
#         cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
#         # Gradually increase selectivity
#         current_top_p = self.ucb_top_p + (1.0 - self.ucb_top_p) * torch.exp(-step / 5000.0)
        
#         # Create cutoff mask
#         cutoff_mask = cumsum_probs <= current_top_p
#         cutoff_mask = torch.cat([cutoff_mask[:, :, :, :1], cutoff_mask[:, :, :, :-1]], dim=-1)
        
#         # Create selection mask
#         selection_mask = torch.zeros_like(ucb_probs)
#         selection_mask.scatter_(-1, sorted_indices, cutoff_mask.float())
        
#         # Apply selection and renormalize
#         selected_probs = ucb_probs * selection_mask
#         selected_probs = selected_probs / (selected_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
#         # Update count buffer
#         count_buffer = self.update_count_buffer(count_buffer, selection_mask)
        
#         return selected_probs, count_buffer, current_beta

#     def forward(self, hidden_states, counter, ucb_count_score, ucb):
#         B, T, C = hidden_states.size()
        
#         mixed_query_layer = self.query(hidden_states)
#         mixed_key_layer = self.key(hidden_states)
#         mixed_value_layer = self.value(hidden_states)

#         query_layer = self.transpose_for_scores(mixed_query_layer)
#         key_layer = self.transpose_for_scores(mixed_key_layer)
#         value_layer = self.transpose_for_scores(mixed_value_layer)

#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
#         # Applica softmax prima della selezione UCB
#         att = self.softmax(attention_scores)
        
#         if ucb and counter >= self.ucb_activation_step:
#             # Calcola UCB scores
#             log_t = math.log(counter)
#             ucb_scores = att + self.ucb_beta * torch.sqrt(log_t / (ucb_count_score.unsqueeze(1) + 1e-8))
            
#             # Top-k selection
#             _, top_indices = torch.topk(ucb_scores, k=self.ucb_top_k, dim=-1)
#             mask = torch.zeros_like(att)
#             mask.scatter_(-1, top_indices, 1.0)
            
#             # Aggiorna lo stato
#             updated_ucb_count_score = ucb_count_score + mask.sum(dim=1) 
            
#             # Applia maschera e normalizza
#             selected_att = att * mask
#             selected_att = selected_att / (selected_att.sum(dim=-1, keepdim=True) + 1e-8)
#             attention_probs = self.attn_dropout(selected_att)
#         else:
#             attention_probs = self.attn_dropout(att)
#             updated_ucb_count_score = ucb_count_score  # Mantieni lo stato invariato
            
           
#         # Apply attention to values
#         context_layer = torch.matmul(attention_probs, value_layer)
        
#         # Reshape context
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(*new_context_layer_shape)
        
#         # Apply output projection
#         attention_output = self.out(context_layer)
#         attention_output = self.proj_dropout(attention_output)

#         return attention_output, attention_probs, updated_ucb_count_score


    
#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)

# class Block(nn.Module):
#     def __init__(self, config, vis):
#         super(Block, self).__init__()
#         self.hidden_size = config.hidden_size
#         self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
#         self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
#         self.ffn = Mlp(config)
#         self.attn = Attention(config, vis)

#     def forward(self, x, counter=None, ucb_count_score=None, ucb=False):
#         # Pre-norm with residual connection
#         h = x
#         x = self.attention_norm(x)
#         x, weights, count_score = self.attn(x, counter=counter, ucb=ucb, ucb_count_score=ucb_count_score)
#         x = x + h

#         # FFN with residual connection
#         h = x
#         x = self.ffn_norm(x)
#         x = self.ffn(x)
#         x = x + h

#         return x, weights, count_score

# class Encoder(nn.Module):
#     def __init__(self, config, vis):
#         super(Encoder, self).__init__()
#         self.vis = vis
#         self.layer = nn.ModuleList()
#         self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        
#         for _ in range(config.transformer["num_layers"]):
#             layer = Block(config, vis)
#             self.layer.append(copy.deepcopy(layer))

#     def forward(self, hidden_states, counter=None, ucb_count_score=None, ucb=False):
#         attn_weights = []

#         # 'ucb_count_score' ora è lo stato per il layer corrente
#         current_layer_ucb_score = ucb_count_score 

#         for layer_block in self.layer:
#             hidden_states, weights, updated_ucb_score = layer_block( # Rinominato per chiarezza
#                 hidden_states,
#                 counter=counter,
#                 ucb_count_score=current_layer_ucb_score, # Passa lo stato corrente
#                 ucb=ucb
#             )

#             if self.vis:
#                 attn_weights.append(weights)

#         encoded = self.encoder_norm(hidden_states)

#         # Restituisce lo stato dell'ultimo layer
#         final_count_score = updated_ucb_score if ucb else torch.tensor(-1.0, device=hidden_states.device)

#         return encoded, attn_weights, final_count_score


# class Mlp(nn.Module):
#     def __init__(self, config):
#         super(Mlp, self).__init__()
#         self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
#         self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
#         self.act_fn = ACT2FN["gelu"]
#         self.dropout = Dropout(config.transformer["dropout_rate"])
#         self._init_weights()

#     def _init_weights(self):
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.xavier_uniform_(self.fc2.weight)
#         nn.init.normal_(self.fc1.bias, std=1e-6)
#         nn.init.normal_(self.fc2.bias, std=1e-6)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act_fn(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.dropout(x)
#         return x

# class Embeddings(nn.Module):
#     """Construct the embeddings from patch, position embeddings."""
#     def __init__(self, config, img_size, in_channels=3):
#         super(Embeddings, self).__init__()
#         self.hybrid = None
#         img_size = _pair(img_size)

#         if config.patches.get("grid") is not None:
#             grid_size = config.patches["grid"]
#             patch_size = (
#                 img_size[0] // 16 // grid_size[0],
#                 img_size[1] // 16 // grid_size[1],
#             )
#             n_patches = (img_size[0] // 16) * (img_size[1] // 16)
#             self.hybrid = True
#         else:
#             patch_size = _pair(config.patches["size"])
#             n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
#             self.hybrid = False

#         if self.hybrid:
#             self.hybrid_model = ResNetV2(
#                 block_units=config.resnet.num_layers,
#                 width_factor=config.resnet.width_factor,
#             )
#             in_channels = self.hybrid_model.width * 16

#         self.patch_embeddings = Conv2d(
#             in_channels=in_channels,
#             out_channels=config.hidden_size,
#             kernel_size=patch_size,
#             stride=patch_size,
#         )
#         self.position_embeddings = nn.Parameter(
#             torch.zeros(1, n_patches + 1, config.hidden_size)
#         )
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
#         self.dropout = Dropout(config.transformer["dropout_rate"])

#     def forward(self, x):
#         B = x.shape[0]
#         cls_tokens = self.cls_token.expand(B, -1, -1)

#         if self.hybrid:
#             x = self.hybrid_model(x)
#         x = self.patch_embeddings(x)
#         x = x.flatten(2)
#         x = x.transpose(-1, -2)
#         x = torch.cat((cls_tokens, x), dim=1)

#         embeddings = x + self.position_embeddings
#         embeddings = self.dropout(embeddings)
#         return embeddings

# class Transformer(nn.Module):
#     def __init__(self, config, img_size, vis):
#         super(Transformer, self).__init__()
#         self.embeddings = Embeddings(config, img_size=img_size)
#         self.encoder = Encoder(config, vis)

#     def forward(self, input_ids, counter, ucb_count_score, ucb=False):  # Aggiungi default per ucb
#         embedding_output = self.embeddings(input_ids)
#         encoded, attn_weights, updated_ucb_count_score = self.encoder(
#             hidden_states=embedding_output,
#             counter=counter,
#             ucb_count_score=ucb_count_score,
#             ucb=ucb
#         )
#         return encoded, attn_weights, updated_ucb_count_score

# class MlpClassifier(nn.Module):
#     def __init__(self, config, in_features, out_features): # Aggiungi in_features, out_features
#         super(MlpClassifier, self).__init__()
#         self.fc1 = Linear(in_features, config.transformer["mlp_dim"]) # in_features è ora dinamico
#         self.fc2 = Linear(config.transformer["mlp_dim"], out_features) # out_features è ora dinamico
#         self.act_fn = ACT2FN["gelu"]
#         self.dropout = Dropout(config.transformer["dropout_rate"])
#         self._init_weights() # Assicurati che _init_weights inizializzi correttamente con le nuove dimensioni

#     def _init_weights(self):
#         # Assicurati che queste inizializzazioni siano appropriate per le tue dimensioni
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.xavier_uniform_(self.fc2.weight)
#         nn.init.normal_(self.fc1.bias, std=1e-6)
#         nn.init.normal_(self.fc2.bias, std=1e-6)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act_fn(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.dropout(x)
#         return x
    
# class VisionTransformer(nn.Module):
#     def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
#         super(VisionTransformer, self).__init__()
#         self.num_classes = num_classes
#         self.zero_head = zero_head
#         self.classifier = config.classifier
#         self.transformer = Transformer(config, img_size, vis)
#         self.head = MlpClassifier(config, in_features=config.hidden_size, out_features=self.num_classes)

#          # Calcola e memorizza la dimensione del blocco
#         if config.patches.get("grid"):
#             grid_size = config.patches["grid"]
#             patch_size = (img_size // 16 // grid_size[0], img_size // 16 // grid_size[1])
#             n_patches = (img_size // 16) * (img_size // 16)
#         else:
#             patch_size = _pair(config.patches["size"])
#             n_patches = (img_size // patch_size[0]) * (img_size // patch_size[1])
#         self.block_size = n_patches + 1  # +1 per il token [CLS]

#     def forward(self, x, counter, ucb_count_score, ucb=True, labels=None):  # Aggiungi default per ucb
#         x, attn_weights, updated_ucb_count_score = self.transformer(
#             input_ids=x,
#             counter=counter,
#             ucb_count_score=ucb_count_score,
#             ucb=ucb
#         )
#         logits = self.head(x[:, 0])
#         # print(f"DEBUG - VisionTransformer: logits - Mean: {logits.mean().item():.6f}, Std: {logits.std().item():.6f}, Min: {logits.min().item():.6f}, Max: {logits.max().item():.6f}")
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
#             return loss, updated_ucb_count_score
#         else:
#             return logits, updated_ucb_count_score


#     def load_from(self, weights):
#         # Carica gli embeddings di patch e posizione
#         self.transformer.embeddings.patch_embeddings.weight.copy_(
#             np2th(weights["embedding/kernel"], conv=True)
#         )
#         self.transformer.embeddings.position_embeddings.copy_(
#             np2th(weights["Transformer/posembed_input/pos_embedding"])
#         )
#         self.cls_token.copy_(np2th(weights["cls"]))

#         # Carica i blocchi dell'encoder
#         for bname, block in self.transformer.encoder.named_children():
#             if bname.startswith("encoder_norm"): # l'ultimo LayerNorm dell'encoder
#                 block.weight.copy_(np2th(weights[f"Transformer/encoder_norm/{'scale'}"]))
#                 block.bias.copy_(np2th(weights[f"Transformer/encoder_norm/{'bias'}"]))
#                 continue

#             for mname, module in block.named_children():
#                 if mname == "attention_norm":
#                     module.weight.copy_(np2th(weights[f"Transformer/encoderblock_{bname}/LayerNorm_0/{'scale'}"]))
#                     module.bias.copy_(np2th(weights[f"Transformer/encoderblock_{bname}/LayerNorm_0/{'bias'}"]))
#                 elif mname == "ffn_norm":
#                     module.weight.copy_(np2th(weights[f"Transformer/encoderblock_{bname}/LayerNorm_1/{'scale'}"]))
#                     module.bias.copy_(np2th(weights[f"Transformer/encoderblock_{bname}/LayerNorm_1/{'bias'}"]))
#                 elif mname == "attn":
#                     # Carica i pesi di query, key, value e out projection per l'attenzione
#                     # Questi sono i layer che vuoi che siano addestrabili per il pruning UCB
#                     module.query.weight.copy_(np2th(weights[f"Transformer/encoderblock_{bname}/MultiHeadDotProductAttention_1/query/kernel"]).t())
#                     module.query.bias.copy_(np2th(weights[f"Transformer/encoderblock_{bname}/MultiHeadDotProductAttention_1/query/bias"]).t())
#                     module.key.weight.copy_(np2th(weights[f"Transformer/encoderblock_{bname}/MultiHeadDotProductAttention_1/key/kernel"]).t())
#                     module.key.bias.copy_(np2th(weights[f"Transformer/encoderblock_{bname}/MultiHeadDotProductAttention_1/key/bias"]).t())
#                     module.value.weight.copy_(np2th(weights[f"Transformer/encoderblock_{bname}/MultiHeadDotProductAttention_1/value/kernel"]).t())
#                     module.value.bias.copy_(np2th(weights[f"Transformer/encoderblock_{bname}/MultiHeadDotProductAttention_1/value/bias"]).t())
#                     module.out.weight.copy_(np2th(weights[f"Transformer/encoderblock_{bname}/MultiHeadDotProductAttention_1/out/kernel"]).t())
#                     module.out.bias.copy_(np2th(weights[f"Transformer/encoderblock_{bname}/MultiHeadDotProductAttention_1/out/bias"]).t())
#                 elif mname == "ffn":
#                     # Carica i pesi della Feed Forward Network
#                     module.fc1.weight.copy_(np2th(weights[f"Transformer/encoderblock_{bname}/MlpBlock_3/Dense_0/kernel"]).t())
#                     module.fc1.bias.copy_(np2th(weights[f"Transformer/encoderblock_{bname}/MlpBlock_3/Dense_0/bias"]).t())
#                     module.fc2.weight.copy_(np2th(weights[f"Transformer/encoderblock_{bname}/MlpBlock_3/Dense_1/kernel"]).t())
#                     module.fc2.bias.copy_(np2th(weights[f"Transformer/encoderblock_{bname}/MlpBlock_3/Dense_1/bias"]).t())

    
#             if "head/kernel" in weights and "head/bias" in weights:
#                 # Se il modello pre-addestrato include una testa E le dimensioni corrispondono, caricala.
#                 # Altrimenti, ignora e usa l'inizializzazione di default della tua testa.
#                 if self.head.weight.shape[0] == weights["head/kernel"].shape[0]:
#                     self.head.weight.copy_(np2th(weights["head/kernel"]).t())
#                     self.head.bias.copy_(np2th(weights["head/bias"]).t())
#                 else:
#                     # Stampa un avviso se le dimensioni della testa non corrispondono
#                     print(f"Warning: Head dimensions mismatch. Not loading pre-trained head weights. "
#                         f"Expected: {self.head.weight.shape[0]}, Found: {weights['head/kernel'].shape[0]}")
#             else:
#                 print("Info: 'head/kernel' or 'head/bias' not found in pre-trained weights. Not loading pre-trained head weights.")