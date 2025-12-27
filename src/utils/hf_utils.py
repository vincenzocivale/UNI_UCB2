
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
            force_download=False # Set to True if you always want to re-download
        )
    weights_path = os.path.join(output_dir, filename)
    return weights_path


def refactor_hf_weight(timm_state_dict, vit_config):
    """
    Maps key names from a timm ViT model's state_dict
    to the key names expected by your VisionTransformer.

    Args:
        timm_state_dict (dict): The state_dict loaded from a timm ViT checkpoint.
        vit_config (ml_collections.ConfigDict): The config of your VisionTransformer,
                                                used to determine num_layers.

    Returns:
        OrderedDict: A new state_dict with key names mapped
                     to match your VisionTransformer model.
    """
    new_state_dict = OrderedDict()

    # --- Embeddings ---
    # The CLS token in your model is under transformer.embeddings
    new_state_dict["transformer.embeddings.cls_token"] = timm_state_dict["cls_token"]
    # The positional embeddings in your model are under transformer.embeddings
    new_state_dict["transformer.embeddings.position_embeddings"] = timm_state_dict["pos_embed"]
    # The patch embeddings (proj) in your model are under transformer.embeddings
    new_state_dict["transformer.embeddings.patch_embeddings.weight"] = timm_state_dict["patch_embed.proj.weight"]
    new_state_dict["transformer.embeddings.patch_embeddings.bias"] = timm_state_dict["patch_embed.proj.bias"]

    # --- Encoder Layers ---
    # `num_layers` is extracted from the provided config
    num_layers = vit_config.transformer["num_layers"]
    # It's assumed that `hidden_size` is consistent throughout the model
    # and is the input/output dimension of the linear layers
    hidden_size = vit_config.hidden_size 

    for i in range(num_layers):
        # Normalizations (LayerNorm)
        # timm's `norm1` corresponds to `attention_norm` in your model
        new_state_dict[f"transformer.encoder.layer.{i}.attention_norm.weight"] = timm_state_dict[f"blocks.{i}.norm1.weight"]
        new_state_dict[f"transformer.encoder.layer.{i}.attention_norm.bias"] = timm_state_dict[f"blocks.{i}.norm1.bias"]
        # timm's `norm2` corresponds to `ffn_norm` (LayerNorm before the MLP) in your model
        new_state_dict[f"transformer.encoder.layer.{i}.ffn_norm.weight"] = timm_state_dict[f"blocks.{i}.norm2.weight"]
        new_state_dict[f"transformer.encoder.layer.{i}.ffn_norm.bias"] = timm_state_dict[f"blocks.{i}.norm2.bias"]

        # Attention (QKV split and output projection)
        # QKV weights are combined in timm and need to be split
        qkv_weight = timm_state_dict[f"blocks.{i}.attn.qkv.weight"] # Shape: [3 * hidden_size, hidden_size]
        qkv_bias = timm_state_dict[f"blocks.{i}.attn.qkv.bias"]     # Shape: [3 * hidden_size]
        
        # Split the weights and biases into three equal parts for Query, Key, Value
        # `.chunk(3, dim=0)` is the most robust way to split PyTorch tensors
        q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)
        
        new_state_dict[f"transformer.encoder.layer.{i}.attn.query.weight"] = q_weight
        new_state_dict[f"transformer.encoder.layer.{i}.attn.query.bias"] = q_bias
        new_state_dict[f"transformer.encoder.layer.{i}.attn.key.weight"] = k_weight
        new_state_dict[f"transformer.encoder.layer.{i}.attn.key.bias"] = k_bias
        new_state_dict[f"transformer.encoder.layer.{i}.attn.value.weight"] = v_weight
        new_state_dict[f"transformer.encoder.layer.{i}.attn.value.bias"] = v_bias

        # Output projection of the attention (timm's `proj`)
        new_state_dict[f"transformer.encoder.layer.{i}.attn.out.weight"] = timm_state_dict[f"blocks.{i}.attn.proj.weight"]
        new_state_dict[f"transformer.encoder.layer.{i}.attn.out.bias"] = timm_state_dict[f"blocks.{i}.attn.proj.bias"]
        
        # Note: If your VisionTransformer model has a separate `attn.bias` not mapped here,
        # it will be initialized by default and ignored by `load_state_dict(strict=False)`.

        # MLP / FFN (Feed Forward Network)
        new_state_dict[f"transformer.encoder.layer.{i}.ffn.fc1.weight"] = timm_state_dict[f"blocks.{i}.mlp.fc1.weight"]
        new_state_dict[f"transformer.encoder.layer.{i}.ffn.fc1.bias"] = timm_state_dict[f"blocks.{i}.mlp.fc1.bias"]
        new_state_dict[f"transformer.encoder.layer.{i}.ffn.fc2.weight"] = timm_state_dict[f"blocks.{i}.mlp.fc2.weight"]
        new_state_dict[f"transformer.encoder.layer.{i}.ffn.fc2.bias"] = timm_state_dict[f"blocks.{i}.mlp.fc2.bias"]

        # LayerScale parameters (ls1.gamma, ls2.gamma)
        # These parameters are usually not present in the original Google/JAX ViT models,
        # but are common in `timm` implementations. If your model doesn't have them,
        # they won't be mapped here, and `load_state_dict(strict=False)` will ignore them.
        # If your model has LayerScale and you want to load them, you need to adjust the key names in your model.
        # For example: new_state_dict[f"transformer.encoder.layer.{i}.ls1.gamma"] = timm_state_dict[f"blocks.{i}.ls1.gamma"]
        # Be aware that the LayerScale modules (`ls1`, `ls2`) must exist in your implementation.
        
        # Example for LayerScale (make sure your VisionTransformer model has these modules!)
        # if f"blocks.{i}.ls1.gamma" in timm_state_dict:
        #     new_state_dict[f"transformer.encoder.layer.{i}.ls1.gamma"] = timm_state_dict[f"blocks.{i}.ls1.gamma"]
        # if f"blocks.{i}.ls2.gamma" in timm_state_dict:
        #     new_state_dict[f"transformer.encoder.layer.{i}.ls2.gamma"] = timm_state_dict[f"blocks.{i}.ls2.gamma"]


    # --- Final Encoder Normalization ---
    # timm's `norm` corresponds to the final `encoder_norm` in your model
    new_state_dict["transformer.encoder.encoder_norm.weight"] = timm_state_dict["norm.weight"]
    new_state_dict["transformer.encoder.encoder_norm.bias"] = timm_state_dict["norm.bias"]

    # --- Head (Classification Head) ---
    # Your configuration indicated `num_classes: 0` and the previous `KeyError: 'head/kernel'` error.
    # This suggests that the timm model might not have a classification head or
    # that you don't want to load it for your downstream task.
    # If the head is present in the timm_state_dict AND you want to use it, map it here.
    # Otherwise, your model's head will be initialized from scratch.
    if "head.weight" in timm_state_dict:
        new_state_dict["head.weight"] = timm_state_dict["head.weight"]
    if "head.bias" in timm_state_dict:
        new_state_dict["head.bias"] = timm_state_dict["head.bias"]
    
    return new_state_dict
