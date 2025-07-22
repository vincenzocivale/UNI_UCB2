import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from torch.cuda.amp import GradScaler, autocast

import os
import math
import logging
from dataclasses import dataclass, field
from tqdm.auto import tqdm
from typing import Optional, Dict, Tuple, List

# Set up logging
# ... (logging setup remains the same) ...
logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Try to import Weights & Biases
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

@dataclass
class TrainingArguments:
    """
    Configuration for the training process, inspired by Hugging Face's TrainingArguments.
    """
    output_dir: str = field(metadata={"help": "The output directory where model predictions and checkpoints will be written."})
    run_name: str = field(default="vit-ucb-run", metadata={"help": "A specific name for this run, used for logging."})
    
    # --- MODIFIED: Added num_train_epochs ---
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(default=-1, metadata={"help": "If > 0: overrides num_train_epochs. Total number of training steps to perform."})
    
    # Training strategy
    learning_rate: float = field(default=3e-2, metadata={"help": "The initial learning rate for SGD."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for the optimizer."})
    warmup_steps: int = field(default=500, metadata={"help": "Number of steps for the learning rate warmup."})
    train_batch_size: int = field(default=64, metadata={"help": "Batch size for training."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."})
    
    # ... (rest of TrainingArguments is the same) ...
    evaluation_strategy: str = field(default="steps", metadata={"help": "When to perform evaluation ('steps' or 'no')."})
    eval_steps: int = field(default=100, metadata={"help": "Run evaluation every N steps."})
    eval_batch_size: int = field(default=64, metadata={"help": "Batch size for evaluation."})
    logging_strategy: str = field(default="steps", metadata={"help": "When to log metrics ('steps')."})
    logging_steps: int = field(default=50, metadata={"help": "Log metrics every N steps."})
    save_strategy: str = field(default="steps", metadata={"help": "When to save checkpoints ('steps')."})
    save_steps: int = field(default=100, metadata={"help": "Save a checkpoint every N steps."})
    save_total_limit: Optional[int] = field(default=2, metadata={"help": "Limit the total number of checkpoints."})
    seed: int = field(default=42, metadata={"help": "Random seed for initialization."})
    fp16: bool = field(default=True, metadata={"help": "Whether to use 16-bit (mixed) precision training."})
    report_to: str = field(default="wandb", metadata={"help": "The integration to report results to ('wandb', 'tensorboard', or 'none')."})

    early_stopping_patience: int = field(default=0, metadata={"help": "Number of evaluations with no improvement after which training will be stopped. If 0, disabled."})
    early_stopping_metric: str = field(default="eval/loss", metadata={"help": "Metric to monitor for early stopping (e.g., 'eval/loss' or 'eval/accuracy')."})
    early_stopping_threshold: float = field(default=0.0001, metadata={"help": "Minimum change in the monitored metric to qualify as an improvement."})


class ModelTrainer:
    # --- MODIFIED: __init__ signature updated ---
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        test_dataloader: Optional[DataLoader] = None,
        class_names: Optional[List[str]] = None,
        optimizers: Tuple[Optional[Optimizer], Optional[_LRScheduler]] = (None, None)
    ):
        self.model = model
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        # --- NEW: Store test_dataloader and class_names ---
        self.test_dataloader = test_dataloader
        self.class_names = class_names
        
        # --- NEW: Logic to handle epochs vs. steps ---
        num_update_steps_per_epoch = len(self.train_dataloader) // self.args.gradient_accumulation_steps
        if self.args.max_steps <= 0:
            self.args.max_steps = int(self.args.num_train_epochs * num_update_steps_per_epoch)
        else:
            self.args.num_train_epochs = math.ceil(self.args.max_steps / num_update_steps_per_epoch)

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # --- MODIFIED: Handle external vs. internal optimizer/scheduler ---
        optimizer, scheduler = optimizers
        self.optimizer = optimizer if optimizer is not None else self.create_optimizer()
        self.scheduler = scheduler if scheduler is not None else self.create_scheduler()
        
        # Mixed Precision & Logging
        self.scaler = GradScaler(enabled=self.args.fp16)
        if self.args.report_to == "wandb" and _WANDB_AVAILABLE:
            wandb.init(project="vit-ucb-pruning", name=self.args.run_name, config=vars(self.args))
            wandb.watch(self.model, log_freq=self.args.logging_steps)

        self.best_metric = None
        self.patience_counter = 0
        # Determina se un valore più alto della metrica è migliore (es. accuratezza) o peggiore (es. loss)
        self.metric_greater_is_better = "loss" not in self.args.early_stopping_metric
        
        if self.args.early_stopping_patience > 0:
            self.best_metric = float('-inf') if self.metric_greater_is_better else float('inf')
            logger.info(f"Early stopping enabled: monitoring '{self.args.early_stopping_metric}' with patience {self.args.early_stopping_patience} and threshold {self.args.early_stopping_threshold}.")

    def create_optimizer(self):
        logger.info("No optimizer passed; creating default SGD optimizer.")
        return SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9, weight_decay=self.args.weight_decay)

    def create_scheduler(self):
        logger.info("No scheduler passed; creating default Warmup-Cosine scheduler.")
        def lr_lambda(current_step: int):
            if current_step < self.args.warmup_steps:
                return float(current_step) / float(max(1, self.args.warmup_steps))
            progress = float(current_step - self.args.warmup_steps) / float(max(1, self.args.max_steps - self.args.warmup_steps))
            return 0.5 * (1.0 + torch.cos(torch.tensor(math.pi * progress)))
        return LambdaLR(self.optimizer, lr_lambda)

    def train(self):
        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Total optimization steps = {self.args.max_steps}")
        
        global_step = 0
        train_loss = 0.0
        self.model.train()
    
        
        for epoch in range(int(math.ceil(self.args.num_train_epochs))):
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{int(self.args.num_train_epochs)}", leave=False)

            for step, batch in enumerate(progress_bar):
                # ... (inner loop logic for training, logging, eval, saving remains the same) ...
                loss = self._training_step(batch, counter=global_step)
                train_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"})
                    
                    if global_step % self.args.logging_steps == 0:
                        # Prepare standard metrics
                        avg_loss = train_loss / self.args.logging_steps
                        metrics_to_log = {
                            "train/loss": avg_loss,
                            "train/learning_rate": self.scheduler.get_last_lr()[0]
                        }

                      
                        if global_step > 500: # Only log UCB stats after pruning starts
                            with torch.no_grad():
                                # Access the UCB buffer from the model
                                ucb_scores = self.model.ucb_count_scores.detach()
                                
                                # Calculate sparsity (percentage of unused connections)
                                total_elements = ucb_scores.numel()
                                zero_counts = (ucb_scores == 0).sum().item()
                                sparsity = zero_counts / total_elements
                                
                                # Calculate other stats
                                mean_selection_count = ucb_scores.mean().item()
                                max_selection_count = ucb_scores.max().item()
                                
                                # Add UCB metrics to the log dictionary
                                ucb_metrics = {
                                    "ucb/sparsity": sparsity,
                                    "ucb/mean_selection_count": mean_selection_count,
                                    "ucb/max_selection_count": max_selection_count,
                                    # We also pass the raw tensor to log a histogram
                                    "ucb/selection_distribution": ucb_scores 
                                }
                                metrics_to_log.update(ucb_metrics)
                        
                        # Pass all metrics to the log function
                        self._log(metrics_to_log, step=global_step)
                        train_loss = 0.0

                    if self.args.evaluation_strategy == "steps" and global_step % self.args.eval_steps == 0:
                        metrics = self.evaluate(counter=global_step)
                        self._log(metrics, step=global_step)

                if global_step >= self.args.max_steps:
                    break

            if self.args.early_stopping_patience > 0:
                logger.info(f"--- Running End-of-Epoch Evaluation for Epoch {epoch + 1} ---")
                metrics = self.evaluate(counter=global_step)
                self._log(metrics, step=global_step)
                
                if self._check_early_stopping(metrics):
                    logger.info(f"Early stopping condition met after epoch {epoch + 1}. Terminating training.")
                    break  # Interrompe il ciclo principale delle epoche
            
            if global_step >= self.args.max_steps:
                break
            
            
        
        logger.info("Training finished. Final evaluation:")
        metrics = self.evaluate(counter=global_step)
        self._log(metrics, step=global_step)
        self._save_checkpoint(global_step, final=True)

    def _training_step(self, batch: tuple, counter: int):
        self.model.train()

        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        
        # Use autocast for mixed-precision
        with autocast(enabled=self.args.fp16):
            loss, _ = self.model(x=inputs, labels=labels, counter=counter, ucb_enabled=True)
        
        # Scale loss for gradient accumulation
        scaled_loss = self.scaler.scale(loss / self.args.gradient_accumulation_steps)
        scaled_loss.backward()
        
        return loss.detach()

    def evaluate(self, counter: int) -> Dict[str, float]:
        logger.info(f"***** Running Evaluation at Step {counter} *****")
        self.model.eval()
        
        total_loss, total_correct, total_samples = 0, 0, 0
        
        progress_bar = tqdm(self.eval_dataloader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            with torch.no_grad(), autocast(enabled=self.args.fp16):
                loss, logits = self.model(x=inputs, labels=labels, counter=counter, ucb_enabled=True)
            
            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += inputs.size(0)
            
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        metrics = {"eval/loss": avg_loss, "eval/accuracy": accuracy}
        logger.info(f"  Evaluation results: {metrics}")
        
        self.model.train() # Set back to training mode
        return metrics
    
    def predict(self):
        logger.info("***** Running Prediction on Test Set *****")
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.test_dataloader, desc="Predicting", leave=False)
        for batch in progress_bar:
            inputs, labels = batch
            inputs = inputs.to(self.device)
            
            with torch.no_grad(), autocast(enabled=self.args.fp16):
                logits = self.model(x=inputs, counter=99999, ucb_enabled=True) # Use a high counter
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        logger.info("Logging confusion matrix to Weights & Biases.")
        self._log({
            "test/confusion_matrix": wandb.plot.confusion_matrix(
                preds=all_preds,
                y_true=all_labels,
                class_names=self.class_names
            )
        }, step=self.args.max_steps) # Log at the final step

        self.model.train()

    def _log(self, metrics: Dict[str, float], step: int):
        if self.args.report_to == "wandb" and _WANDB_AVAILABLE:
            # Create a new dictionary to hold W&B-compatible objects
            wandb_metrics = {}
            for key, value in metrics.items():
                # If the value is a tensor, log it as a histogram
                if isinstance(value, torch.Tensor):
                    wandb_metrics[key] = wandb.Histogram(value.cpu())
                else:
                    wandb_metrics[key] = value
            
            wandb.log(wandb_metrics, step=step)
            
    def _save_checkpoint(self, step: int, final: bool = False):
        name = "final_checkpoint" if final else f"checkpoint-{step}"
        output_path = os.path.join(self.args.output_dir, name)
        os.makedirs(output_path, exist_ok=True)
        
        logger.info(f"Saving model checkpoint to {output_path}")
        torch.save(self.model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    def _check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """Restituisce True se l'addestramento deve essere interrotto, altrimenti False."""
        
        metric_value = metrics.get(self.args.early_stopping_metric)
        if metric_value is None:
            logger.warning(f"Early stopping metric '{self.args.early_stopping_metric}' not found in evaluation metrics. Skipping check.")
            return False

        improvement = 0.0
        if self.metric_greater_is_better:
            improvement = metric_value - self.best_metric
        else:
            improvement = self.best_metric - metric_value

        if improvement > self.args.early_stopping_threshold:
            self.best_metric = metric_value
            self.patience_counter = 0
            logger.info(f"New best metric found: {self.args.early_stopping_metric} = {metric_value:.4f}. Resetting patience.")
        else:
            self.patience_counter += 1
            logger.info(f"No improvement found. Patience counter: {self.patience_counter}/{self.args.early_stopping_patience}.")

        return self.patience_counter >= self.args.early_stopping_patience

