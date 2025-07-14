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

block_size = (224 * 224) // (
    16 * 16
) + 1



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
        # Assuming proj_dropout uses a different dropout rate than attention_dropout_rate
        self.proj_dropout = Dropout(config.transformer["dropout_rate"]) 
        
        self.softmax = Softmax(dim=-1)
        
        # Calculate block_size dynamically or ensure it's available from config
        # For a ViT, block_size is typically (IMG_SIZE // PATCH_SIZE)**2 + 1 (for CLS token)
        # Using a placeholder here based on common ViT configs, adjust if your setup differs
        calculated_block_size = (224 // 16) * (224 // 16) + 1 # Example for 224x224, 16x16 patches
        self.register_buffer("bias", torch.tril(torch.ones(calculated_block_size, calculated_block_size))
                                            .view(1, 1, calculated_block_size, calculated_block_size))
        
        self.register_buffer("ucb_count_score_buffer", None)
        self.current_block_size = None # Per tenere traccia del block_size corrente


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def UCB_Score(self, Attention_Score, beta, iter_val, v, current_batch_size, current_block_size):
      
        if self.ucb_count_score_buffer is None or \
           self.ucb_count_score_buffer.shape[0] != current_batch_size or \
           self.ucb_count_score_buffer.shape[2] != current_block_size:
            # logger.info(f"Re-initializing UCB Count Score buffer for batch_size={current_batch_size}, block_size={current_block_size}")
            self.ucb_count_score_buffer = torch.ones(
                current_batch_size,
                self.num_heads,
                current_block_size,
                current_block_size,
                requires_grad=False,
            ).to(Attention_Score.device)
            self.current_block_size = current_block_size # Aggiorna la dimensione memorizzata

        # Ora usa il buffer persistente interno invece di un parametro esterno
        Count_score_internal = self.ucb_count_score_buffer

        # Gestione di iter_val per evitare log(0)
        # Se iter_val è il counter (che può essere 0 all'inizio), aggiungi un epsilon o gestiscilo
        num = torch.tensor(np.log(iter_val) if iter_val > 0 else 0.0,
                           dtype=torch.float32,
                           device=Attention_Score.device) # Usa attention_score.device per coerenza

        # Modifica il calcolo di score_sum per riflettere le dimensioni di Count_score_internal
        # Se UCB_score è calcolato per ogni coppia (query_patch, key_patch) all'interno di ogni testa:
        score_sum = Count_score_internal # In questo caso, ogni elemento di Count_score_internal è un conteggio.
                                        # L'errore originale "score_sum = torch.sum(Count_score, dim=0)" implicava
                                        # che la dimensione 0 dovesse scomparire, ma la tua UCB_score ha la stessa
                                        # dimensione di Attention_Score. Se `Count_score` ha le dimensioni [B, N_H, S, S],
                                        # allora `score_sum` deve mantenere quelle dimensioni per essere aggiunto correttamente
                                        # ad Attention_Score o essere broadcastabile.

        # Se il tuo UCB_Score è per ogni elemento dell'Attention_Score, allora `score_sum` deve avere quella forma.
        # L'interpretazione più logica è che `Count_score_internal` stesso rappresenti i conteggi per ogni elemento.
        # Quindi, `score_sum` dovrebbe essere semplicemente `Count_score_internal`.
        UCB_score = Attention_Score + beta * torch.sqrt(num / (Count_score_internal + 1e-6)) # Usiamo Count_score_internal direttamente

        # Attenzione al dim=3 per torch.topk
        # Se UCB_score è [B, N_H, S, S], topk(..., dim=3) è corretto per prendere i k migliori lungo la dimensione delle key-patches.
        _, Max_Indiices = torch.topk(UCB_score, dim=3, k=10)

        # num_classes per one_hot_vector deve essere la dimensione della dimensione 3 di UCB_score (S)
        # Cioè, current_block_size
        one_hot_vector = F.one_hot(Max_Indiices, num_classes=current_block_size).float()
    
        summed_score = one_hot_vector.sum(dim=3) # Questo dovrebbe produrre [B, N_H, S, S]

        # È importante che `summed_score` abbia la stessa forma di `Count_score_internal`
        Count_score_new = Count_score_internal + summed_score # Ho rinominato la variabile locale

        # Aggiorna il buffer persistente per il prossimo passo
        self.ucb_count_score_buffer = Count_score_new.clone().detach() # Clona e detach per impedire backprop infinite

        newatt = Attention_Score * summed_score
        newatt = newatt / (torch.sum(newatt, dim=3, keepdim=True) + 1e-6)

        update_attn = torch.matmul(newatt.float(), v)
        # Ritorna il risultato dell'attenzione aggiornata e il conteggio aggiornato
        return update_attn, self.ucb_count_score_buffer # Ritorna il buffer aggiornato
    
    def forward(self, hidden_states, counter, ucb):
        B, T, C = hidden_states.size() 

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply causal mask if enabled (original line was commented out)
        attention_scores = attention_scores.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        
        att = self.softmax(attention_scores)
        
        batch_size, seq_len, _ = hidden_states.shape
        current_block_size = seq_len

        

        if self.vis:
            weights = att.detach().clone()
        else:
            weights = torch.zeros_like(att).detach() # Dummy tensor if not visualizing

        if counter > 500 and ucb:
            attention_probs = self.attn_dropout(att)
            attention_output_ucb, final_count_score = self.UCB_Score(
                attention_probs,          # Attention_Score (your attention probabilities)
                self.beta,                # beta (assuming self.beta is defined in __init__)
                counter,                  # iter_val (your counter for UCB)
                value_layer,              # v (the value tensor)
                batch_size,               # current_batch_size
                current_block_size        # current_block_size
            )
            context_layer = attention_output_ucb
            
        else:
            final_count_score = 0 
            attention_probs = self.attn_dropout(att)
            context_layer = torch.matmul(attention_probs, value_layer)
            
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output, weights, final_count_score


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


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config) # Assicurati che Mlp sia definito
        self.attn = Attention(config, vis) # Assicurati che Attention sia definito

    def forward(self, x, counter, ucb, layer_id=0):
        h = x
        x = self.attention_norm(x)
        
        # Modifica la chiamata a self.attn: rimuovi UCB_Count_Score
        x, weights, Count_Score = self.attn(x, counter, ucb=ucb) 
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights, Count_Score

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = (
                np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")])
                .view(self.hidden_size, self.hidden_size)
                .t()
            )
            key_weight = (
                np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")])
                .view(self.hidden_size, self.hidden_size)
                .t()
            )
            value_weight = (
                np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")])
                .view(self.hidden_size, self.hidden_size)
                .t()
            )
            out_weight = (
                np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")])
                .view(self.hidden_size, self.hidden_size)
                .t()
            )

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(
                np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")])
            )
            self.attention_norm.bias.copy_(
                np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")])
            )
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


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
        final_count_score_from_last_layer = None # Inizializza per catturare l'ultimo Count_Score
        for i, layer_block in enumerate(self.layer):
            # Modifica la chiamata a layer_block: rimuovi UCB_Count_Score
            hidden_states, weights, current_layer_count_score = layer_block(
                hidden_states, counter, ucb=ucb, layer_id=i
            )
            if self.vis:
                attn_weights.append(weights)
            # Memorizza il Count_Score dell'ultimo strato (o aggregalo se necessario)
            final_count_score_from_last_layer = current_layer_count_score 
            
        encoded = self.encoder_norm(hidden_states)
        
        # Restituisci l'ultimo Count_Score calcolato
        return encoded, attn_weights, final_count_score_from_last_layer


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size) # Assicurati che Embeddings sia definito
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids, counter, ucb):
        embedding_output = self.embeddings(input_ids)
        # Modifica la chiamata a self.encoder: rimuovi UCB_Count_Score
        encoded, attn_weights, Count_Score = self.encoder(embedding_output, counter, ucb=ucb)
        return encoded, attn_weights, Count_Score # Count_Score proviene da self.encoder


class VisionTransformer(nn.Module):
    def __init__(
        self, config, img_size=224, num_classes=21843, zero_head=False, vis=False
    ):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis)
        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, x, counter, ucb, labels=None): # Rimosso UCB_Count_Score
        # Modifica la chiamata a self.transformer: rimuovi UCB_Count_Score
        x, attn_weights, Count_Score = self.transformer(x, counter, ucb=ucb)
        logits = self.head(x[:, 0])

        return logits, Count_Score # Count_Score ora proviene da self.transformer
            

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
                logger.info(
                    "load_pretrained: resized variant: %s to %s"
                    % (posemb.size(), posemb_new.size())
                )
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

                for (
                    bname,
                    block,
                ) in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


