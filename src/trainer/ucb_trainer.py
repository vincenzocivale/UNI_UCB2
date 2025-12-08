
from transformers import Trainer
import torch

class UcbTrainer(Trainer):
    def __init__(self, *args, top_k_indices=None, pruning_enabled=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_k_indices = top_k_indices
        self.pruning_enabled = pruning_enabled

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation to pass the training step counter and pruning flag to the model.
        """
        labels = inputs.pop("labels", None)
        
        outputs = model(
            **inputs,
            counter=self.state.global_step,
            pruning_enabled=self.pruning_enabled,
            top_k_indices=self.top_k_indices,
            labels=labels
        )

        if labels is not None:
            loss, logits = outputs
        else:
            loss = None
            logits = outputs

        return (loss, outputs) if return_outputs else loss
