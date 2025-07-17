
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
    ai nomi delle chiavi attesi dal tuo VisionTransformer.

    Args:
        timm_state_dict (dict): Lo state_dict caricato da un checkpoint timm ViT.
        vit_config (ml_collections.ConfigDict): Il config del tuo VisionTransformer,
                                                usato per determinare num_layers.

    Returns:
        OrderedDict: Un nuovo state_dict con i nomi delle chiavi mappati
                     per corrispondere al tuo modello VisionTransformer.
    """
    new_state_dict = OrderedDict()

    # --- Embeddings ---
    # Il CLS token nel tuo modello si trova sotto transformer.embeddings
    new_state_dict["transformer.embeddings.cls_token"] = timm_state_dict["cls_token"]
    # I positional embeddings nel tuo modello si trovano sotto transformer.embeddings
    new_state_dict["transformer.embeddings.position_embeddings"] = timm_state_dict["pos_embed"]
    # I patch embeddings (proj) nel tuo modello si trovano sotto transformer.embeddings
    new_state_dict["transformer.embeddings.patch_embeddings.weight"] = timm_state_dict["patch_embed.proj.weight"]
    new_state_dict["transformer.embeddings.patch_embeddings.bias"] = timm_state_dict["patch_embed.proj.bias"]

    # --- Encoder Layers ---
    # `num_layers` è estratto dal config fornito
    num_layers = vit_config.transformer["num_layers"]
    # Si assume che il `hidden_size` sia coerente in tutto il modello
    # e che sia la dimensione dell'input/output dei layer lineari
    hidden_size = vit_config.hidden_size 

    for i in range(num_layers):
        # Normalizzazioni (LayerNorm)
        # `norm1` di timm corrisponde a `attention_norm` nel tuo modello
        new_state_dict[f"transformer.encoder.layer.{i}.attention_norm.weight"] = timm_state_dict[f"blocks.{i}.norm1.weight"]
        new_state_dict[f"transformer.encoder.layer.{i}.attention_norm.bias"] = timm_state_dict[f"blocks.{i}.norm1.bias"]
        # `norm2` di timm corrisponde a `ffn_norm` (LayerNorm prima della MLP) nel tuo modello
        new_state_dict[f"transformer.encoder.layer.{i}.ffn_norm.weight"] = timm_state_dict[f"blocks.{i}.norm2.weight"]
        new_state_dict[f"transformer.encoder.layer.{i}.ffn_norm.bias"] = timm_state_dict[f"blocks.{i}.norm2.bias"]

        # Attenzione (QKV split e output projection)
        # I pesi QKV sono combinati in timm e devono essere splittati
        qkv_weight = timm_state_dict[f"blocks.{i}.attn.qkv.weight"] # Forma: [3 * hidden_size, hidden_size]
        qkv_bias = timm_state_dict[f"blocks.{i}.attn.qkv.bias"]     # Forma: [3 * hidden_size]
        
        # Splitta i pesi e i bias in tre parti uguali per Query, Key, Value
        # `.chunk(3, dim=0)` è il modo più robusto per splittare i tensori PyTorch
        q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)
        
        new_state_dict[f"transformer.encoder.layer.{i}.attn.query.weight"] = q_weight
        new_state_dict[f"transformer.encoder.layer.{i}.attn.query.bias"] = q_bias
        new_state_dict[f"transformer.encoder.layer.{i}.attn.key.weight"] = k_weight
        new_state_dict[f"transformer.encoder.layer.{i}.attn.key.bias"] = k_bias
        new_state_dict[f"transformer.encoder.layer.{i}.attn.value.weight"] = v_weight
        new_state_dict[f"transformer.encoder.layer.{i}.attn.value.bias"] = v_bias

        # Output projection dell'attenzione (`proj` di timm)
        new_state_dict[f"transformer.encoder.layer.{i}.attn.out.weight"] = timm_state_dict[f"blocks.{i}.attn.proj.weight"]
        new_state_dict[f"transformer.encoder.layer.{i}.attn.out.bias"] = timm_state_dict[f"blocks.{i}.attn.proj.bias"]
        
        # Nota: Se il tuo modello VisionTransformer ha un `attn.bias` separato non mappato qui,
        # sarà inizializzato di default e ignorato da `load_state_dict(strict=False)`.

        # MLP / FFN (Feed Forward Network)
        new_state_dict[f"transformer.encoder.layer.{i}.ffn.fc1.weight"] = timm_state_dict[f"blocks.{i}.mlp.fc1.weight"]
        new_state_dict[f"transformer.encoder.layer.{i}.ffn.fc1.bias"] = timm_state_dict[f"blocks.{i}.mlp.fc1.bias"]
        new_state_dict[f"transformer.encoder.layer.{i}.ffn.fc2.weight"] = timm_state_dict[f"blocks.{i}.mlp.fc2.weight"]
        new_state_dict[f"transformer.encoder.layer.{i}.ffn.fc2.bias"] = timm_state_dict[f"blocks.{i}.mlp.fc2.bias"]

        # Parametri LayerScale (ls1.gamma, ls2.gamma)
        # Questi parametri di solito non sono presenti nei modelli ViT originali di Google/JAX,
        # ma sono comuni nelle implementazioni `timm`. Se il tuo modello non li ha,
        # non verranno mappati qui, e `load_state_dict(strict=False)` li ignorerà.
        # Se il tuo modello ha LayerScale e vuoi caricarli, devi aggiustare i nomi delle chiavi nel tuo modello.
        # Ad esempio: new_state_dict[f"transformer.encoder.layer.{i}.ls1.gamma"] = timm_state_dict[f"blocks.{i}.ls1.gamma"]
        # Fai attenzione che i moduli LayerScale (`ls1`, `ls2`) devono esistere nella tua implementazione.
        
        # Esempio per LayerScale (assicurati che il tuo modello VisionTransformer abbia questi moduli!)
        # if f"blocks.{i}.ls1.gamma" in timm_state_dict:
        #     new_state_dict[f"transformer.encoder.layer.{i}.ls1.gamma"] = timm_state_dict[f"blocks.{i}.ls1.gamma"]
        # if f"blocks.{i}.ls2.gamma" in timm_state_dict:
        #     new_state_dict[f"transformer.encoder.layer.{i}.ls2.gamma"] = timm_state_dict[f"blocks.{i}.ls2.gamma"]


    # --- Final Encoder Normalization ---
    # `norm` di timm corrisponde a `encoder_norm` finale nel tuo modello
    new_state_dict["transformer.encoder.encoder_norm.weight"] = timm_state_dict["norm.weight"]
    new_state_dict["transformer.encoder.encoder_norm.bias"] = timm_state_dict["norm.bias"]

    # --- Head (Testa di classificazione) ---
    # La tua configurazione indicava `num_classes: 0` e l'errore `KeyError: 'head/kernel'` precedente.
    # Questo suggerisce che il modello timm potrebbe non avere una testa di classificazione o
    # che non vuoi caricarla per la tua task downstream.
    # Se la testa è presente nel timm_state_dict E la vuoi usare, mappala qui.
    # Altrimenti, la testa del tuo modello verrà inizializzata da zero.
    if "head.weight" in timm_state_dict:
        new_state_dict["head.weight"] = timm_state_dict["head.weight"]
    if "head.bias" in timm_state_dict:
        new_state_dict["head.bias"] = timm_state_dict["head.bias"]
    
    return new_state_dict
