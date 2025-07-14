
import os
from huggingface_hub import hf_hub_download
from typing import Optional, OrderedDict

def download_weights(output_dir="/equilibrium/datasets/TCGA-histological-data/vit_weights_cache") -> str:
    os.makedirs(output_dir, exist_ok=True)
        
    filename = "pytorch_model.bin"
    hf_hub_download(
            "MahmoodLab/UNI",
            filename=filename,
            local_dir=output_dir,
            force_download=False # Imposta a True se vuoi sempre riscaricare
        )
    weights_path = os.path.join(output_dir, filename)
    return weights_path


def refactor_hf_weight(timm_state_dict, vit_config):
    """
    Mappa i nomi delle chiavi dallo state_dict di un modello timm ViT
    ai nomi delle chiavi attesi dal tuo VisionTransformer (basato su JAX/Flax).
    """
    new_state_dict = OrderedDict()

    # --- Embeddings ---
    new_state_dict["transformer.embeddings.cls_token"] = timm_state_dict["cls_token"]
    new_state_dict["transformer.embeddings.position_embeddings"] = timm_state_dict["pos_embed"]
    new_state_dict["transformer.embeddings.patch_embeddings.weight"] = timm_state_dict["patch_embed.proj.weight"]
    new_state_dict["transformer.embeddings.patch_embeddings.bias"] = timm_state_dict["patch_embed.proj.bias"]

    # --- Encoder Layers ---
    num_layers = vit_config.transformer["num_layers"]
    for i in range(num_layers):
        # Normalizzazioni
        new_state_dict[f"transformer.encoder.layer.{i}.attention_norm.weight"] = timm_state_dict[f"blocks.{i}.norm1.weight"]
        new_state_dict[f"transformer.encoder.layer.{i}.attention_norm.bias"] = timm_state_dict[f"blocks.{i}.norm1.bias"]
        new_state_dict[f"transformer.encoder.layer.{i}.ffn_norm.weight"] = timm_state_dict[f"blocks.{i}.norm2.weight"] # timm's norm2 is ffn_norm
        new_state_dict[f"transformer.encoder.layer.{i}.ffn_norm.bias"] = timm_state_dict[f"blocks.{i}.norm2.bias"] # timm's norm2 is ffn_norm

        # Attenzione (qkv split)
        qkv_weight = timm_state_dict[f"blocks.{i}.attn.qkv.weight"]
        qkv_bias = timm_state_dict[f"blocks.{i}.attn.qkv.bias"]
        
        # Determine the dimension for splitting based on the output dimension of qkv
        # The qkv_weight's first dimension is (3 * hidden_size), qkv_bias is (3 * hidden_size)
        hidden_size = qkv_weight.shape[0] // 3
        
        # Split QKV weights
        new_state_dict[f"transformer.encoder.layer.{i}.attn.query.weight"] = qkv_weight[0:hidden_size, :]
        new_state_dict[f"transformer.encoder.layer.{i}.attn.key.weight"] = qkv_weight[hidden_size:2*hidden_size, :]
        new_state_dict[f"transformer.encoder.layer.{i}.attn.value.weight"] = qkv_weight[2*hidden_size:3*hidden_size, :]

        # Split QKV biases
        new_state_dict[f"transformer.encoder.layer.{i}.attn.query.bias"] = qkv_bias[0:hidden_size]
        new_state_dict[f"transformer.encoder.layer.{i}.attn.key.bias"] = qkv_bias[hidden_size:2*hidden_size]
        new_state_dict[f"transformer.encoder.layer.{i}.attn.value.bias"] = qkv_bias[2*hidden_size:3*hidden_size]

        # Output projection
        new_state_dict[f"transformer.encoder.layer.{i}.attn.out.weight"] = timm_state_dict[f"blocks.{i}.attn.proj.weight"]
        new_state_dict[f"transformer.encoder.layer.{i}.attn.out.bias"] = timm_state_dict[f"blocks.{i}.attn.proj.bias"]
        
        # Note: transformer.encoder.layer.{i}.attn.bias in your model has no direct timm counterpart here.
        # It will be initialized by the VisionTransformer constructor and left untouched by load_state_dict(strict=False).

        # MLP / FFN
        new_state_dict[f"transformer.encoder.layer.{i}.ffn.fc1.weight"] = timm_state_dict[f"blocks.{i}.mlp.fc1.weight"]
        new_state_dict[f"transformer.encoder.layer.{i}.ffn.fc1.bias"] = timm_state_dict[f"blocks.{i}.mlp.fc1.bias"]
        new_state_dict[f"transformer.encoder.layer.{i}.ffn.fc2.weight"] = timm_state_dict[f"blocks.{i}.mlp.fc2.weight"]
        new_state_dict[f"transformer.encoder.layer.{i}.ffn.fc2.bias"] = timm_state_dict[f"blocks.{i}.mlp.fc2.bias"]

        # LayerScale parameters (ls1.gamma, ls2.gamma) from timm are not present in your ViT keys, they will be ignored.

    # --- Final Encoder Normalization ---
    new_state_dict["transformer.encoder.encoder_norm.weight"] = timm_state_dict["norm.weight"]
    new_state_dict["transformer.encoder.encoder_norm.bias"] = timm_state_dict["norm.bias"]

    # --- Head (if applicable) ---
    # Only map if the 'head' keys exist in the timm state_dict AND you want to use them.
    # Your JSON specified num_classes: 0, so the head might not be trained/present in the timm weights you downloaded.
    # If present, map them. Otherwise, your model's head will be initialized from scratch.
    if "head.weight" in timm_state_dict:
        new_state_dict["head.weight"] = timm_state_dict["head.weight"]
    if "head.bias" in timm_state_dict:
        new_state_dict["head.bias"] = timm_state_dict["head.bias"]

    return new_state_dict
