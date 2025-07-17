import ml_collections
from src.rl.modelling import VisionTransformer
from src.utils.hf_utils import refactor_hf_weight
from typing import Optional

def get_uni_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def inizialize_model(timm_pretrained_state_dict, num_classes: Optional[int] = 0):
    
    config = get_uni_config()

    model = VisionTransformer(config=config, img_size=224, num_classes=num_classes, zero_head=False, vis=True)

    mapped_state_dict = refactor_hf_weight(timm_pretrained_state_dict, config)

    model.load_state_dict(mapped_state_dict, strict=False)

    return model
