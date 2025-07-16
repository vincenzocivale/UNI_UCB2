# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

from .modeling_resnet import ResNetV2

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {
    "gelu": torch.nn.functional.gelu,
    "relu": torch.nn.functional.relu,
    "swish": swish,
}

block_size = (224 * 224) // (16 * 16) + 1

class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)
        self.out = Linear(config.hidden_size, config.hidden_size)
        
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["dropout_rate"])
        self.softmax = Softmax(dim=-1)
        
        # UCB parameters - more conservative
        self.ucb_beta_initial = 2.0
        self.ucb_beta_decay = 0.999
        self.ucb_top_p = 0.9  # Keep 90% of patches initially
        self.ucb_activation_step = 1000  # Start UCB much later
        
        # Per-batch count buffers
        self.count_buffer_size = 512  # Max sequence length
        self.count_decay = 0.95  # Decay factor for count buffer

    def initialize_count_buffer(self, batch_size, seq_len, device):
        """Initialize per-batch count buffers"""
        return torch.ones(batch_size, seq_len, seq_len, device=device) * 0.1

    def update_count_buffer(self, count_buffer, selection_mask):
        """Update count buffer with decay"""
        # Decay existing counts
        count_buffer.mul_(self.count_decay)
        # Add new selections
        count_buffer.add_(selection_mask.mean(dim=1))  # Average across heads
        return count_buffer

    def compute_ucb_scores(self, attention_scores, count_buffer, step):
        """Compute UCB scores with proper exploration-exploitation balance"""
        batch_size, num_heads, seq_len, _ = attention_scores.shape
        
        # Decaying beta for reducing exploration over time
        current_beta = self.ucb_beta_initial * (self.ucb_beta_decay ** step)
        
        # Compute confidence intervals
        log_t = torch.log(torch.tensor(float(step), device=attention_scores.device))
        confidence = current_beta * torch.sqrt(log_t / (count_buffer + 1e-8))
        
        # Expand confidence to match attention dimensions
        confidence = confidence.unsqueeze(1).expand(-1, num_heads, -1, -1)
        
        # Add confidence to raw attention scores (before softmax)
        ucb_scores = attention_scores + confidence
        
        return ucb_scores, current_beta

    def apply_ucb_selection(self, attention_scores, count_buffer, step):
        """Apply UCB-based patch selection"""
        batch_size, num_heads, seq_len, _ = attention_scores.shape
        
        # Compute UCB scores
        ucb_scores, current_beta = self.compute_ucb_scores(attention_scores, count_buffer, step)
        
        # Apply softmax to UCB scores
        ucb_probs = self.softmax(ucb_scores)
        
        # Top-p selection
        sorted_probs, sorted_indices = torch.sort(ucb_probs, dim=-1, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Gradually increase selectivity
        current_top_p = self.ucb_top_p + (1.0 - self.ucb_top_p) * torch.exp(-step / 5000.0)
        
        # Create cutoff mask
        cutoff_mask = cumsum_probs <= current_top_p
        cutoff_mask = torch.cat([cutoff_mask[:, :, :, :1], cutoff_mask[:, :, :, :-1]], dim=-1)
        
        # Create selection mask
        selection_mask = torch.zeros_like(ucb_probs)
        selection_mask.scatter_(-1, sorted_indices, cutoff_mask.float())
        
        # Apply selection and renormalize
        selected_probs = ucb_probs * selection_mask
        selected_probs = selected_probs / (selected_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Update count buffer
        count_buffer = self.update_count_buffer(count_buffer, selection_mask)
        
        return selected_probs, count_buffer, current_beta

    def forward(self, hidden_states, counter, ucb):
        B, T, C = hidden_states.size()
        
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Raw attention scores (before softmax)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Store weights for visualization (without detaching)
        weights = None
        final_count_score = None
        
        # Apply UCB if enabled and after activation step
        if ucb and counter >= self.ucb_activation_step:
            # Initialize count buffer for this batch
            count_buffer = self.initialize_count_buffer(B, T, attention_scores.device)
            
            # Apply UCB selection
            attention_probs, count_buffer, current_beta = self.apply_ucb_selection(
                attention_scores, count_buffer, counter
            )
            
            final_count_score = count_buffer.mean(dim=0)  # Average across batch
            
            if self.vis:
                weights = attention_probs.clone()
        else:
            # Standard attention
            attention_probs = self.softmax(attention_scores)
            attention_probs = self.attn_dropout(attention_probs)
            
            if self.vis:
                weights = attention_probs.clone()
            
            final_count_score = torch.tensor(-1.0, device=attention_scores.device)

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape context
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Apply output projection
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output, weights, final_count_score

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x, counter, ucb):
        # Pre-norm with residual connection
        h = x
        x = self.attention_norm(x)
        x, weights, count_score = self.attn(x, counter, ucb)
        x = x + h

        # FFN with residual connection
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x, weights, count_score

class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states, counter, ucb):
        attn_weights = []
        all_count_scores = []
        
        for layer_block in self.layer:
            hidden_states, weights, count_score = layer_block(hidden_states, counter, ucb)
            
            if self.vis:
                attn_weights.append(weights)
            
            # Collect valid count scores
            if isinstance(count_score, torch.Tensor) and count_score.numel() > 1:
                all_count_scores.append(count_score)
        
        encoded = self.encoder_norm(hidden_states)
        
        # Return the last valid count score or -1
        final_count_score = all_count_scores[-1] if all_count_scores else torch.tensor(-1.0)
        
        return encoded, attn_weights, final_count_score

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings."""
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (
                img_size[0] // 16 // grid_size[0],
                img_size[1] // 16 // grid_size[1],
            )
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(
                block_units=config.resnet.num_layers,
                width_factor=config.resnet.width_factor,
            )
            in_channels = self.hybrid_model.width * 16

        self.patch_embeddings = Conv2d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches + 1, config.hidden_size)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids, counter, ucb):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights, count_score = self.encoder(embedding_output, counter, ucb=ucb)
        return encoded, attn_weights, count_score

class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, x, counter, ucb, labels=None):
        x, attn_weights, count_score = self.transformer(x, counter, ucb=ucb)
        logits = self.head(x[:, 0])
        return logits, count_score

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(
                np2th(weights["embedding/kernel"], conv=True)
            )
            self.transformer.embeddings.patch_embeddings.bias.copy_(
                np2th(weights["embedding/bias"])
            )
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(
                np2th(weights["Transformer/encoder_norm/scale"])
            )
            self.transformer.encoder.encoder_norm.bias.copy_(
                np2th(weights["Transformer/encoder_norm/bias"])
            )

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print("load_pretrained: grid-size from %s to %s" % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(weights["conv_root/kernel"], conv=True)
                )
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)