
from transformers import Trainer, EvalPrediction
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def compute_metrics(p: EvalPrediction):
    """
    Computes macro F1, precision, and recall for a given set of predictions.
    """
    preds = np.argmax(p.predictions, axis=1)
    return {
        'f1_macro': f1_score(p.label_ids, preds, average='macro'),
        'precision_macro': precision_score(p.label_ids, preds, average='macro'),
        'recall_macro': recall_score(p.label_ids, preds, average='macro'),
    }

class UcbTrainer(Trainer):
    def __init__(self, *args, top_k_indices=None, pruning_enabled=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_k_indices = top_k_indices
        self.pruning_enabled = pruning_enabled

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom loss computation to pass the training step counter and pruning flag to the model.
        """
        labels = inputs.pop("labels", None)
        
        outputs = model(
            **inputs, # Now includes 'pixel_values' directly
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
