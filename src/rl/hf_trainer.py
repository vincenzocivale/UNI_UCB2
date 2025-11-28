"""
This module contains a custom Hugging Face Trainer for ViT pruning, 
including specific logic for UCB and random pruning strategies.
"""

import torch
import torch.nn as nn
from transformers import Trainer
from typing import Dict, Union, Any, Tuple, List, Optional

# --- Custom Data Collator ---
def custom_data_collator(features: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Manually stack the features from the dataset (tuples) and return a dictionary.
    """
    if not isinstance(features[0], tuple):
        return Trainer.data_collator(features)

    pixel_values = torch.stack([f[0] for f in features])
    labels = torch.stack([f[1] for f in features])
    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = self._detect_model_type()
        # Override data collator if the dataset returns tuples
        if self.train_dataset and isinstance(self.train_dataset[0], tuple):
            self.data_collator = custom_data_collator
        
        print(f"[CustomTrainer] Detected model type: {self.model_type}")

    def _detect_model_type(self) -> str:
        """Automatically detect the model type based on available attributes."""
        # Assumes model is already on the correct device
        model = self.model
        if hasattr(model, 'ucb_count_scores'):
            return "ucb"
        elif hasattr(model, 'random_seed') or any(hasattr(block.attn, 'rng') for block in getattr(model, 'blocks', [])):
            return "random"
        else:
            return "baseline"

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss for UCB, random, or baseline models.
        This is the main training step.
        """
        pixel_values = inputs.get("pixel_values")
        labels = inputs.get("labels")
        counter = self.state.global_step

        if self.model_type == "ucb":
            # UCB model expects 'x' as input key
            loss, logits = model(x=pixel_values, labels=labels, counter=counter, ucb_enabled=True)
        elif self.model_type == "random":
            # Random pruning model expects 'x' and 'random_enabled'
            loss, logits = model(x=pixel_values, labels=labels, counter=counter, random_enabled=True)
        else: # baseline
            # Standard model forward pass
            outputs = model(pixel_values)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, {"logits": logits}) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Custom prediction step to handle deterministic evaluation for UCB models.
        """
        pixel_values = inputs.get("pixel_values").to(self.args.device)
        labels = inputs.get("labels").to(self.args.device)

        with torch.no_grad():
            if self.model_type == "ucb":
                keep_ratio = getattr(model, 'keep_ratio', 1.0)
                
                # Use autocast for mixed precision
                with torch.cuda.amp.autocast(enabled=self.args.fp16):
                    if keep_ratio < 1.0:
                        top_k_indices = model.get_top_k_patch_indices(keep_ratio=keep_ratio)
                        logits = model.forward_pruned(pixel_values, top_k_indices.to(self.args.device))
                    else:
                        # Standard forward pass with UCB disabled
                        output = model(x=pixel_values, labels=None, ucb_enabled=False)
                        logits = output.logits if hasattr(output, 'logits') else output
                
                loss = nn.CrossEntropyLoss()(logits, labels)
            
            elif self.model_type == "random":
                 with torch.cuda.amp.autocast(enabled=self.args.fp16):
                    loss, logits = model(x=pixel_values, labels=labels, random_enabled=False)
            else: # baseline
                with torch.cuda.amp.autocast(enabled=self.args.fp16):
                    outputs = model(pixel_values)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss = nn.CrossEntropyLoss()(logits, labels)

        return (loss, logits, labels)


from transformers import TrainerCallback

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class PruningMetricsCallback(TrainerCallback):
    """
    A TrainerCallback that logs pruning-specific metrics during training.
    This replicates the logic from the old trainer's _log_pruning_metrics.
    """

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """
        Event called after logging. We check for 'loss' to ensure this is a training log.
        """
        if not state.is_world_process_zero or logs is None:
            return

        # We only want to log these metrics during training steps
        is_training_log = 'loss' in logs and 'eval_loss' not in logs
        if not is_training_log:
            return

        model_type = getattr(kwargs.get('trainer'), 'model_type', None)
        if model_type == 'ucb':
            self._log_ucb_metrics(model, logs)
        elif model_type == 'random':
            self._log_random_metrics(model, logs)

    def _log_ucb_metrics(self, model, logs: Dict):
        """Log UCB-specific metrics."""
        if not hasattr(model, 'ucb_count_scores'):
            return
            
        with torch.no_grad():
            ucb_scores = model.ucb_count_scores.detach().float()
            logs["pruning/ucb_sparsity"] = (ucb_scores == 0).sum().item() / ucb_scores.numel()
            logs["pruning/ucb_mean_selection_count"] = ucb_scores.mean().item()
            logs["pruning/ucb_max_selection_count"] = ucb_scores.max().item()
            logs["pruning/ucb_min_selection_count"] = ucb_scores.min().item()
            logs["pruning/ucb_selection_std"] = ucb_scores.std().item()

            if _WANDB_AVAILABLE and wandb.run is not None:
                # wandb.Histogram needs to be logged directly as a value in the dict
                logs["pruning/ucb_selection_distribution"] = wandb.Histogram(ucb_scores.cpu().numpy())

    def _log_random_metrics(self, model, logs: Dict):
        """Log Random pruning-specific metrics."""
        if not hasattr(model, 'get_pruning_stats'):
            return

        stats = model.get_pruning_stats()
        logs["pruning/method"] = "random"
        logs["pruning/random_seed"] = stats.get('random_seed', 'unknown')
        
        # Log keep ratios and other stats
        for key, value in stats.items():
            if 'keep_ratio' in key or 'kept_patches' in key:
                logs[f"pruning/{key}"] = value


import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from transformers import EvalPrediction
from typing import Dict

def compute_metrics(p: EvalPrediction) -> Dict:
    """
    Compute accuracy and F1 score from predictions.
    The Trainer calls this function during evaluation.
    """
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    
    return {
        "eval_accuracy": accuracy,
        "eval_f1": f1,
    }

