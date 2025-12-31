import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from torch.cuda.amp import GradScaler, autocast

import os
import math
import logging
import shutil
from collections import deque
from dataclasses import dataclass
from tqdm.auto import tqdm
from typing import Optional, Dict, Tuple, List, Union
from sklearn.metrics import f1_score
from src.models.vit.pruning import get_global_pruning_indices
from src.training.phases import TrainingPhase

# --- Logging Configuration ---
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --- Weights & Biases (W&B) Integration ---
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

@dataclass
class TrainingArguments:
    """Arguments defining the training process."""
    output_dir: str
    run_name: str = "vit-pruning-run"
    num_train_epochs: float = 3.0
    max_steps: int = -1
    learning_rate: float = 3e-2
    weight_decay: float = 0.0
    warmup_steps: int = 500
    train_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    eval_batch_size: int = 64
    logging_strategy: str = "steps"
    logging_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: Optional[int] = 2
    seed: int = 42
    fp16: bool = True
    report_to: str = "wandb"
    early_stopping_patience: int = 0
    early_stopping_metric: str = "eval/loss"
    early_stopping_threshold: float = 0.0001
    save_best_model: bool = True
    # Model type parameter
    model_type: str = "ucb"  # "ucb", "random", or "baseline"
    freeze_backbone: bool = False
    # Input-aware weight for Option A (replaces input_aware_extra_tokens)
    input_aware_weight: float = 0.7  # Balance between input-aware (1.0) and UCB (0.0)
    log_checkpoints_to_wandb: bool = True

class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        test_dataloader: Optional[DataLoader] = None,
        class_names: Optional[List[str]] = None,
        optimizers: Tuple[Optional[Optimizer], Optional[_LRScheduler]] = (None, None),
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.class_names = class_names

        # Detect model type automatically if not specified
        self.model_type = self._detect_model_type()
        if self.args.model_type != "auto":
            self.model_type = self.args.model_type
        
        logger.info(f"Detected/Using model type: {self.model_type}")

        # Calculate total number of training steps
        num_update_steps_per_epoch = len(self.train_dataloader) // self.args.gradient_accumulation_steps
        if self.args.max_steps <= 0:
            self.args.max_steps = int(self.args.num_train_epochs * num_update_steps_per_epoch)
        else:
            self.args.num_train_epochs = math.ceil(self.args.max_steps / num_update_steps_per_epoch)

        self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model.to(self.device)


        # Initialize optimizer and scheduler
        optimizer, scheduler = optimizers
        self.optimizer = optimizer if optimizer is not None else self.create_optimizer()
        self.scheduler = scheduler if scheduler is not None else self.create_scheduler()

        self.scaler = GradScaler(enabled=self.args.fp16)
        
        # Initialize W&B if requested
        if self.args.report_to == "wandb" and _WANDB_AVAILABLE:
            # Add model type to wandb config
            config = vars(self.args).copy()
            config['model_type'] = self.model_type
            
            wandb.init(project="vit-ucb-dynamic-pruning", name=self.args.run_name, config=config)
            wandb.watch(self.model, log_freq=self.args.logging_steps)

        # Checkpoint management
        self._checkpoints = [] if self.args.save_total_limit is not None and self.args.save_total_limit > 0 else None

        # Early stopping initialization
        self.patience_counter = 0
        self.metric_greater_is_better = "loss" not in self.args.early_stopping_metric
        self.best_metric = None
        if self.args.early_stopping_patience > 0 or self.args.save_best_model:
            self.best_metric = float('-inf') if self.metric_greater_is_better else float('inf')
            logger.info(f"Monitoraggio della metrica '{self.args.early_stopping_metric}' per early stopping e/o salvataggio del modello migliore.")

        # Set input_aware_weight for Option A models
        if hasattr(self.model, "input_aware_weight"):
            self.model.input_aware_weight = self.args.input_aware_weight
            logger.info(f"Set model input_aware_weight to {self.args.input_aware_weight}")


    def _detect_model_type(self) -> str:
        """Automatically detect the model type based on available attributes."""
        if hasattr(self.model, 'ucb_count_scores'):
            return "ucb"
        elif hasattr(self.model, 'random_seed') or any(hasattr(block.attn, 'rng') for block in getattr(self.model, 'blocks', [])):
            return "random"
        else:
            return "baseline"

    def create_optimizer(self) -> Optimizer:
        logger.info("Nessun optimizer fornito; creazione di un optimizer SGD di default.")
        return SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9, weight_decay=self.args.weight_decay)

    def create_scheduler(self) -> _LRScheduler:
        logger.info("Nessuno scheduler fornito; creazione di uno scheduler Warmup-Cosine di default.")
        def lr_lambda(current_step: int):
            if current_step < self.args.warmup_steps:
                return float(current_step) / float(max(1, self.args.warmup_steps))
            progress = float(current_step - self.args.warmup_steps) / float(max(1, self.args.max_steps - self.args.warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return LambdaLR(self.optimizer, lr_lambda)

    def train(self, pruning_logging: bool = True):
        logger.info("***** Starting Training *****")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Total optimization steps = {self.args.max_steps}")
        logger.info(f"  Model type = {self.model_type}")
        if hasattr(self.model, 'input_aware_weight'):
            logger.info(f"  Input-aware weight = {self.model.input_aware_weight}")

        global_step = 0
        train_loss = 0.0
        self.model.train()

        best_model_path = None
        
        # Main training loop
        for epoch in range(int(math.ceil(self.args.num_train_epochs))):
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{int(self.args.num_train_epochs)}", leave=False)
            
            for step, batch in enumerate(progress_bar):
                # Pass global_step as counter for UCB
                loss = self._training_step(batch, counter=global_step)
                train_loss += loss.item()

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Log global_step for debugging
                    if global_step % 100 == 0:
                        logger.info(f"[DEBUG TRAIN] Global step: {global_step}")
                    
                    global_step += 1

                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}", "step": global_step})

                    # Logging
                    if global_step % self.args.logging_steps == 0:
                        metrics_to_log = {
                            "train/loss": train_loss / self.args.logging_steps,
                            "train/learning_rate": self.scheduler.get_last_lr()[0],
                            "train/global_step": global_step
                        }
                        if pruning_logging:
                            self._log_pruning_metrics(metrics_to_log)
                        self._log(metrics_to_log, step=global_step)
                        train_loss = 0.0

                    # Evaluation
                    if self.args.evaluation_strategy == "steps" and global_step % self.args.eval_steps == 0:
                        metrics = self.evaluate(counter=global_step)
                        if self.args.save_best_model and self._check_for_improvement(metrics):
                            self._save_checkpoint(global_step, is_best=True)
                            best_model_path = os.path.join(self.args.output_dir, "best_model", f"{self.args.run_name}.bin")
                        
                        self._log(metrics, step=global_step)
                        
                        if self._check_early_stopping():
                            logger.info("Early stopping triggered. Stopping training.")
                            self.finalize_training(global_step, best_model_path=best_model_path)
                            return

                    # Save checkpoint
                    if self.args.save_strategy == "steps" and global_step % self.args.save_steps == 0:
                        self._save_checkpoint(global_step)

            if self.args.evaluation_strategy == "epoch":
                metrics = self.evaluate(counter=global_step)
                self._log(metrics, step=global_step)
                if self._check_early_stopping():
                    logger.info("Early stopping triggered. Stopping training.")
                    self.finalize_training(global_step, best_model_path=best_model_path)
                    return
            
            if global_step >= self.args.max_steps:
                break
        
        self.finalize_training(global_step, best_model_path=best_model_path)


    def finalize_training(self, global_step: int, best_model_path: Optional[str] = None):
        """Executes final steps after training completion."""
        logger.info("Training completed. Performing final evaluation.")
        
        # Load the best model if it exists
        if best_model_path and os.path.exists(best_model_path):
            logger.info(f"Loading best model from {best_model_path} for final evaluation.")
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        
        metrics, val_preds, val_labels = self.evaluate(counter=global_step, return_preds=True)
        self._log(metrics, step=global_step)

        if self.args.report_to == "wandb" and _WANDB_AVAILABLE:
            logger.info("Logging validation set confusion matrix to W&B.")
            self._log_confusion_matrix("val", val_preds, val_labels, step=global_step)

        if self.test_dataloader is not None:
            self.predict(step=global_step)
        
        self._save_checkpoint(global_step, final=True)
        if self.args.report_to == "wandb" and _WANDB_AVAILABLE:
            wandb.finish()

    def _training_step(self, batch: tuple, counter: int) -> torch.Tensor:
        self.model.train()
        inputs, labels = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        
        with autocast(enabled=self.args.fp16):
            # Universal forward pass handling
            output = self._forward_pass(inputs, labels, counter, phase=TrainingPhase.UCB_ESTIMATION)
            loss = output[0] if isinstance(output, tuple) else output

        scaled_loss = self.scaler.scale(loss / self.args.gradient_accumulation_steps)
        scaled_loss.backward()
        return loss.detach()

    def _forward_pass(self, inputs: torch.Tensor, labels: Optional[torch.Tensor], counter: int, phase: TrainingPhase):
        """Universal forward pass that works with all model types, explicitly controlled by TrainingPhase."""
        if self.model_type == "ucb":
            keep_ratio = getattr(self.model, 'keep_ratio', 1.0)
            
            if phase == TrainingPhase.UCB_ESTIMATION:
                # Training: usa UCB per esplorare e selezionare dinamicamente le patch
                return self.model(x=inputs, labels=labels, counter=counter, ucb_enabled=True)
            else:
                # Validation/Inference: Uses input-aware pruning (Option A)
                # First block runs on all patches, then prunes based on input+UCB
                with torch.no_grad():
                    output = self.model(x=inputs, labels=None, ucb_enabled=False)
                    logits = output.logits if hasattr(output, 'logits') else output

                    loss = None
                    if labels is not None:
                        loss_fct = nn.CrossEntropyLoss()
                        loss = loss_fct(logits, labels)
                    return (loss, logits) if loss is not None else (torch.tensor(0.0), logits)
        
        elif self.model_type == "random":
            # For random model, `random_enabled=True` during training for dynamic random pruning,
            # `random_enabled=False` during inference for deterministic random pruning.
            random_enabled = (phase == TrainingPhase.UCB_ESTIMATION) # Re-using the phase for random_enabled
            return self.model(x=inputs, labels=labels, counter=counter, random_enabled=random_enabled)
        else: # Baseline
            output = self.model(inputs)
            logits = output.logits if hasattr(output, 'logits') else output
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            return (loss, logits) if loss is not None else (torch.tensor(0.0), logits)

    def evaluate(self, counter: int, return_preds: bool = False) -> Union[Dict[str, float], Tuple[Dict[str, float], list, list]]:
        logger.info(f"***** Running Evaluation at Step {counter} *****")
        self.model.eval()

        total_loss, total_correct, total_samples = 0, 0, 0
        all_preds, all_labels = [], []

        for batch in tqdm(self.eval_dataloader, desc="Evaluation", leave=False):
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            with torch.no_grad(), autocast(enabled=self.args.fp16):
                output = self._forward_pass(inputs, labels, counter, phase=TrainingPhase.PRUNING_INFERENCE)
                loss, logits = output if isinstance(output, tuple) else (torch.tensor(0.0), output)
            
            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += inputs.size(0)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        metrics = {"eval/loss": avg_loss, "eval/accuracy": accuracy, "eval/f1": f1}
        logger.info(f"  Evaluation results: {metrics}")
        
        self.model.train()
        return (metrics, all_preds, all_labels) if return_preds else metrics
    
    def predict(self, step: Optional[int] = None):
        if self.test_dataloader is None:
            logger.warning("Test dataloader not provided. Skipping prediction on test set.")
            return

        logger.info("***** Running Prediction on Test Set *****")
        self.model.eval()

        all_preds, all_labels = [], []
        for batch in tqdm(self.test_dataloader, desc="Prediction", leave=False):
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            with torch.no_grad(), autocast(enabled=self.args.fp16):
                output = self._forward_pass(inputs, None, counter=99999, phase=TrainingPhase.PRUNING_INFERENCE)
                logits = output[1] if isinstance(output, tuple) and len(output) > 1 else output

            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        accuracy = (all_preds == all_labels).sum() / all_labels.shape[0]
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        metrics = {"test/accuracy": accuracy, "test/f1": f1}
        self._log(metrics, step=step)
        logger.info(f"  Test set results: {metrics}")
        
        self.model.train()
    
    def _log_pruning_metrics(self, metrics_dict: Dict):
        """Log metrics specific to the pruning method."""
        if self.model_type == "ucb" and hasattr(self.model, 'ucb_count_scores'):
            self._log_ucb_metrics(metrics_dict)
        elif self.model_type == "random":
            self._log_random_metrics(metrics_dict)
        # For baseline models, no special metrics to log

    def _log_ucb_metrics(self, metrics_dict: Dict):
        """Log UCB-specific metrics."""
        with torch.no_grad():
            ucb_scores = self.model.ucb_count_scores.detach().float()
            
            # Basic UCB statistics
            metrics_dict.update({
                "pruning/ucb_sparsity": (ucb_scores == 0).sum().item() / ucb_scores.numel(),
                "pruning/ucb_mean_selection_count": ucb_scores.mean().item(),
                "pruning/ucb_max_selection_count": ucb_scores.max().item(),
                "pruning/ucb_min_selection_count": ucb_scores.min().item(),
                "pruning/ucb_selection_std": ucb_scores.std().item(),
            })
            
            # Input-aware specific metrics (for Option A)
            if hasattr(self.model, 'input_aware_weight'):
                metrics_dict["pruning/input_aware_weight"] = self.model.input_aware_weight
            
            # Per-layer statistics
            layer_means = ucb_scores.mean(dim=(1, 2))  # Average over heads and patches per layer
            for layer_idx, layer_mean in enumerate(layer_means):
                metrics_dict[f"pruning/ucb_layer_{layer_idx}_mean"] = layer_mean.item()
            

            if _WANDB_AVAILABLE:
                metrics_dict["pruning/ucb_selection_distribution"] = wandb.Histogram(ucb_scores.cpu().numpy())

    def _log_random_metrics(self, metrics_dict: Dict):
        """Log Random pruning-specific metrics."""
        # For random pruning, we can log the pruning configuration
        if hasattr(self.model, 'get_pruning_stats'):
            stats = self.model.get_pruning_stats()
            metrics_dict.update({
                "pruning/method": "random",
                "pruning/random_seed": stats.get('random_seed', 'unknown'),
                "pruning/total_tokens": stats.get('total_tokens', 0),
            })
            
            # Log keep ratios for each layer
            for key, value in stats.items():
                if 'keep_ratio' in key or 'kept_tokens' in key:
                    metrics_dict[f"pruning/{key}"] = value

    def _log(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to W&B."""
        if self.args.report_to == "wandb" and _WANDB_AVAILABLE:
            wandb.log(metrics, step=step)
    
    def _log_confusion_matrix(self, prefix: str, preds: list, labels: list, step: Optional[int] = None):
        """Log confusion matrix to W&B."""
        if _WANDB_AVAILABLE:
            wandb.log({
                f"{prefix}/confusion_matrix": wandb.plot.confusion_matrix(
                    preds=preds, y_true=labels, class_names=self.class_names
                )
            }, step=step)

    def _save_checkpoint(self, step: int, final: bool = False, is_best: bool = False):
        if is_best:
            name = "best_model"
        elif final:
            name = "final_checkpoint"
        else:
            name = f"checkpoint-{step}"
        
        output_path = os.path.join(self.args.output_dir, name)
        os.makedirs(output_path, exist_ok=True)
        ckpt_path = os.path.join(output_path, f"{self.args.run_name}.bin")
        
        logger.info(f"Saving model checkpoint to {output_path}")
        torch.save(self.model.state_dict(), ckpt_path)
        
        # Gestisce il limite di checkpoint da salvare
        if self._checkpoints is not None and not is_best and not final:
            self._checkpoints.append(output_path)
            if len(self._checkpoints) > self.args.save_total_limit:
                oldest_ckpt = self._checkpoints.pop(0)
                if os.path.exists(oldest_ckpt):
                    logger.info(f"Removing old checkpoint: {oldest_ckpt}")
                    shutil.rmtree(oldest_ckpt)
                else:
                    logger.warning(f"Old checkpoint not found at: {oldest_ckpt}, unable to remove.")
            
        if self.args.report_to == "wandb" and _WANDB_AVAILABLE and self.args.log_checkpoints_to_wandb:
            aliases = []
            if final: aliases.append("final")
            if is_best: aliases.append("best")
            self._log_artifact_to_wandb(name, ckpt_path, aliases)
    
    def _log_artifact_to_wandb(self, name: str, path: str, aliases: List[str]):
        """Log model artifact to W&B."""
        if _WANDB_AVAILABLE:
            artifact = wandb.Artifact(name=f"{self.args.run_name}", type="model", metadata={"step": name, "model_type": self.model_type})
            artifact.add_file(path)
            wandb.log_artifact(artifact, aliases=aliases)
            logger.info(f"Checkpoint '{name}' uploaded to W&B with aliases: {aliases}.")

    def _check_for_improvement(self, metrics: Dict[str, float]) -> bool:
        """Check if the monitored metric has improved."""
        metric_value = metrics.get(self.args.early_stopping_metric)
        if metric_value is None:
            return False

        improvement = (metric_value - self.best_metric) if self.metric_greater_is_better else (self.best_metric - metric_value)
        if improvement > self.args.early_stopping_threshold:
            self.best_metric = metric_value
            self.patience_counter = 0
            logger.info(f"New best metric found: {self.args.early_stopping_metric} = {metric_value:.4f}. Resetting patience counter.")
            return True
        else:
            self.patience_counter += 1
            logger.info(f"No improvement. Patience counter: {self.patience_counter}/{self.args.early_stopping_patience}.")
            return False

    def _check_early_stopping(self) -> bool:
        """Check if early stopping condition is met."""
        if self.args.early_stopping_patience <= 0:
            return False
        return self.patience_counter >= self.args.early_stopping_patience
