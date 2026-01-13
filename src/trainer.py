"""
Trainer for Vision Transformer with UCB-based Dynamic Pruning
Updated with dataset tagging for W&B
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler, autocast

import os
import math
import logging
import shutil
from dataclasses import dataclass
from tqdm.auto import tqdm
from typing import Optional, Dict, Tuple, List, Union
from sklearn.metrics import f1_score
import time
from fvcore.nn import FlopCountAnalysis

from src.phases import TrainingPhase

# Logging setup
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class TrainingArguments:
    """Training configuration"""
    # Paths
    output_dir: str
    run_name: str = "vit-pruning"
    dataset_name: str = "unknown"  # For W&B tagging (e.g., "BACH", "CRC")
    
    # Training schedule
    num_train_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    warmup_steps: int = 500
    
    # Batch settings
    train_batch_size: int = 64
    eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    
    # Logging & Evaluation
    evaluation_strategy: str = "epoch"  # "epoch" or "steps"
    logging_steps: int = 100
    eval_steps: int = 500
    
    # Checkpointing
    save_strategy: str = "epoch"  # "epoch" or "steps"
    save_steps: int = 500
    save_best_model: bool = True
    save_total_limit: int = 2
    
    # Early stopping
    early_stopping_patience: int = 0
    early_stopping_metric: str = "eval/loss"
    early_stopping_threshold: float = 0.0001
    
    # Model specific
    model_type: str = "ucb"  # "ucb", "random", or "baseline"
    input_aware_weight: float = 0.7
    
    # Hardware
    fp16: bool = True
    seed: int = 42
    
    # Tracking
    report_to: str = "wandb"
    log_checkpoints_to_wandb: bool = False


class ModelTrainer:
    """Trainer for ViT with dynamic pruning"""
    
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        test_dataloader: Optional[DataLoader] = None,
        class_names: Optional[List[str]] = None,
        optimizers: Tuple[Optional[Optimizer], Optional[_LRScheduler]] = (None, None),
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.class_names = class_names
        
        # Device setup
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Model type detection
        self.model_type = self._detect_model_type()
        logger.info(f"Model type: {self.model_type}")
        
        # Training steps calculation
        self._compute_training_steps()
        
        # Optimizer and scheduler
        optimizer, scheduler = optimizers
        self.optimizer = optimizer or self._create_default_optimizer()
        self.scheduler = scheduler or self._create_default_scheduler()
        self.scaler = GradScaler(enabled=self.args.fp16)
        
        # Checkpointing
        self._checkpoints = [] if self.args.save_total_limit > 0 else None
        
        # Early stopping
        self.patience_counter = 0
        self.metric_greater_is_better = "loss" not in self.args.early_stopping_metric
        self.best_metric = float('-inf') if self.metric_greater_is_better else float('inf')
        
        # Model configuration
        if hasattr(self.model, "input_aware_weight"):
            self.model.input_aware_weight = self.args.input_aware_weight
        
        # Initialize tracking
        self._init_wandb()
    
    def _detect_model_type(self) -> str:
        """Auto-detect model type from attributes"""
        if hasattr(self.model, 'ucb_count_scores'):
            return "ucb"
        elif hasattr(self.model, 'random_seed'):
            return "random"
        return "baseline"
    
    def _compute_training_steps(self):
        """Calculate total training steps"""
        steps_per_epoch = len(self.train_dataloader) // self.args.gradient_accumulation_steps
        self.total_steps = int(self.args.num_train_epochs * steps_per_epoch)
        logger.info(f"Total training steps: {self.total_steps}")
    
    def _create_default_optimizer(self) -> Optimizer:
        """Create AdamW optimizer"""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
    
    def _create_default_scheduler(self) -> _LRScheduler:
        """Create cosine warmup scheduler"""
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(step: int):
            if step < self.args.warmup_steps:
                return step / max(1, self.args.warmup_steps)
            progress = (step - self.args.warmup_steps) / max(1, self.total_steps - self.args.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking"""
        if self.args.report_to == "wandb" and WANDB_AVAILABLE:
            config = vars(self.args).copy()
            config['model_type'] = self.model_type
            
            # Create tags including dataset name
            tags = [self.model_type, self.args.dataset_name]
            
            wandb.init(
                project="vit-ucb-pruning",
                name=self.args.run_name,
                config=config,
                tags=tags
            )
            wandb.watch(self.model, log_freq=self.args.logging_steps)
            logger.info(f"W&B initialized with tags: {tags}")
    
    def train(self):
        """Main training loop"""
        logger.info("=" * 60)
        logger.info("Starting Training")
        logger.info(f"  Dataset: {self.args.dataset_name}")
        logger.info(f"  Epochs: {self.args.num_train_epochs}")
        logger.info(f"  Steps: {self.total_steps}")
        logger.info(f"  Model: {self.model_type}")
        logger.info("=" * 60)
        
        self.model.train()
        global_step = 0
        train_loss = 0.0
        best_model_path = None
        
        for epoch in range(int(self.args.num_train_epochs)):
            epoch_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")
            
            for step, batch in enumerate(epoch_bar):
                # Training step
                loss = self._training_step(batch, global_step)
                train_loss += loss.item()
                
                # Gradient accumulation
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # Update progress bar
                    epoch_bar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                    })
                    
                    # Periodic logging
                    if global_step % self.args.logging_steps == 0:
                        self._log_training_metrics(train_loss, global_step, epoch)
                        train_loss = 0.0
                    
                    # Mid-epoch evaluation
                    if self.args.evaluation_strategy == "steps" and global_step % self.args.eval_steps == 0:
                        metrics = self.evaluate(global_step)
                        self._log(metrics, global_step)
                        
                        if self.args.save_best_model and self._check_improvement(metrics):
                            self._save_checkpoint(global_step, is_best=True, epoch=epoch + 1)
                            best_model_path = self._get_best_model_path()
                        
                        if self._check_early_stopping():
                            self._finalize(global_step, best_model_path)
                            return
            
            # End of epoch
            logger.info(f"Epoch {epoch + 1} completed")
            
            # Epoch evaluation
            if self.args.evaluation_strategy == "epoch":
                metrics = self.evaluate(global_step)
                metrics["train/epoch"] = epoch + 1
                self._log(metrics, global_step)
                
                if self.args.save_best_model and self._check_improvement(metrics):
                    self._save_checkpoint(global_step, is_best=True, epoch=epoch + 1)
                    best_model_path = self._get_best_model_path()
                
                if self._check_early_stopping():
                    self._finalize(global_step, best_model_path)
                    return
            
            # Save checkpoint
            if self.args.save_strategy == "epoch":
                self._save_checkpoint(global_step, epoch=epoch + 1)
        
        self._finalize(global_step, best_model_path)
    
    def _training_step(self, batch: Tuple, counter: int) -> torch.Tensor:
        """Single training step"""
        inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
        
        with autocast(enabled=self.args.fp16):
            output = self._forward(inputs, labels, counter, TrainingPhase.UCB_ESTIMATION)
            loss = output[0] if isinstance(output, tuple) else output
        
        scaled_loss = self.scaler.scale(loss / self.args.gradient_accumulation_steps)
        scaled_loss.backward()
        return loss.detach()
    
    def _forward(self, inputs: torch.Tensor, labels: Optional[torch.Tensor], 
                 counter: int, phase: TrainingPhase):
        """Model forward pass"""
        if self.model_type == "ucb":
            if phase == TrainingPhase.UCB_ESTIMATION:
                return self.model(x=inputs, labels=labels, counter=counter, ucb_enabled=True)
            else:
                with torch.no_grad():
                    output = self.model(x=inputs, labels=None, ucb_enabled=False)
                    logits = output.logits if hasattr(output, 'logits') else output
                    
                    loss = None
                    if labels is not None:
                        loss = nn.CrossEntropyLoss()(logits, labels)
                    return (loss, logits) if loss is not None else (torch.tensor(0.0), logits)
        
        elif self.model_type == "random":
            random_enabled = (phase == TrainingPhase.UCB_ESTIMATION)
            return self.model(x=inputs, labels=labels, counter=counter, random_enabled=random_enabled)
        
        else:  # Baseline
            output = self.model(inputs)
            logits = output.logits if hasattr(output, 'logits') else output
            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits, labels)
            return (loss, logits) if loss is not None else (torch.tensor(0.0), logits)
    
    def evaluate(self, step: int) -> Dict[str, float]:
        """Run evaluation"""
        logger.info(f"Running evaluation at step {step}")
        self.model.eval()
        
        total_loss, total_correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        
        for batch in tqdm(self.eval_dataloader, desc="Evaluating", leave=False):
            inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
            
            with torch.no_grad(), autocast(enabled=self.args.fp16):
                output = self._forward(inputs, labels, step, TrainingPhase.PRUNING_INFERENCE)
                loss, logits = output if isinstance(output, tuple) else (torch.tensor(0.0), output)
            
            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == labels).sum().item()
            total += inputs.size(0)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
        
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        metrics = {
            "eval/loss": total_loss / total,
            "eval/accuracy": total_correct / total,
            "eval/f1": f1_score(all_labels, all_preds, average="macro", zero_division=0)
        }
        
        logger.info(f"Eval results: {metrics}")
        self.model.train()
        return metrics
    
    def predict(self, step: int):
        """Run prediction on test set"""
        if self.test_dataloader is None:
            return
        
        logger.info("Running test prediction")
        self.model.eval()
        
        all_preds, all_labels = [], []
        
        for batch in tqdm(self.test_dataloader, desc="Testing", leave=False):
            inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
            
            with torch.no_grad(), autocast(enabled=self.args.fp16):
                output = self._forward(inputs, None, step, TrainingPhase.PRUNING_INFERENCE)
                logits = output[1] if isinstance(output, tuple) else output
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
        
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        metrics = {
            "test/accuracy": (all_preds == all_labels).mean(),
            "test/f1": f1_score(all_labels, all_preds, average="macro", zero_division=0)
        }
        
        self._log(metrics, step)
        logger.info(f"Test results: {metrics}")
        self.model.train()
    
    def _log_training_metrics(self, train_loss: float, step: int, epoch: int):
        """Log training metrics"""
        metrics = {
            "train/loss": train_loss / self.args.logging_steps,
            "train/learning_rate": self.scheduler.get_last_lr()[0],
            "train/epoch": epoch + 1,
            "train/step": step
        }
        
        self._log(metrics, step)
    
    def _log(self, metrics: Dict, step: int):
        """Log metrics to W&B"""
        if self.args.report_to == "wandb" and WANDB_AVAILABLE:
            wandb.log(metrics, step=step)
    
    def _save_checkpoint(self, step: int, is_best: bool = False, 
                        final: bool = False, epoch: Optional[int] = None):
        """Save model checkpoint"""
        if is_best:
            name = "best_model"
        elif final:
            name = "final_checkpoint"
        elif epoch is not None:
            name = f"epoch-{epoch}"
        else:
            name = f"step-{step}"
        
        path = os.path.join(self.args.output_dir, name)
        os.makedirs(path, exist_ok=True)
        
        ckpt_path = os.path.join(path, f"{self.args.run_name}.bin")
        torch.save(self.model.state_dict(), ckpt_path)
        logger.info(f"Saved checkpoint: {name}")
        
        # Manage checkpoint limit
        if self._checkpoints is not None and not is_best and not final:
            self._checkpoints.append(path)
            if len(self._checkpoints) > self.args.save_total_limit:
                old_path = self._checkpoints.pop(0)
                if os.path.exists(old_path):
                    shutil.rmtree(old_path)
        
        # Log to W&B
        if self.args.log_checkpoints_to_wandb and WANDB_AVAILABLE:
            artifact = wandb.Artifact(
                name=self.args.run_name,
                type="model",
                metadata={"checkpoint": name}
            )
            artifact.add_file(ckpt_path)
            aliases = []
            if is_best: aliases.append("best")
            if final: aliases.append("final")
            wandb.log_artifact(artifact, aliases=aliases)
    
    def _get_best_model_path(self) -> str:
        """Get path to best model"""
        return os.path.join(self.args.output_dir, "best_model", f"{self.args.run_name}.bin")
    
    def _check_improvement(self, metrics: Dict) -> bool:
        """Check if metric improved"""
        value = metrics.get(self.args.early_stopping_metric)
        if value is None:
            return False
        
        delta = value - self.best_metric if self.metric_greater_is_better else self.best_metric - value
        
        if delta > self.args.early_stopping_threshold:
            self.best_metric = value
            self.patience_counter = 0
            logger.info(f"New best {self.args.early_stopping_metric}: {value:.4f}")
            return True
        
        self.patience_counter += 1
        return False
    
    def _check_early_stopping(self) -> bool:
        """Check if should stop early"""
        if self.args.early_stopping_patience <= 0:
            return False
        
        if self.patience_counter >= self.args.early_stopping_patience:
            logger.info(f"Early stopping triggered (patience={self.patience_counter})")
            return True
        return False
    
    def _finalize(self, step: int, best_model_path: Optional[str]):
        logger.info("Finalizing training")
        
        # Load best
        if best_model_path and os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        
        # Test set evaluation
        if self.test_dataloader:
            logger.info("Running final test evaluation")
            self.model.eval()
            
            start = time.time()
            all_preds, all_labels = [], []
            
            for batch in tqdm(self.test_dataloader, desc="Test", leave=False):
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                with torch.no_grad(), autocast(enabled=self.args.fp16):
                    output = self._forward(inputs, None, step, TrainingPhase.PRUNING_INFERENCE)
                    logits = output[1] if isinstance(output, tuple) else output
                all_preds.append(torch.argmax(logits, dim=-1).cpu())
                all_labels.append(labels.cpu())
            
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()
            inference_time = time.time() - start
            
            # Metrics
            metrics = {
                "test/f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
                "test/accuracy": (all_preds == all_labels).mean(),
                "test/inference_time": inference_time
            }
            
            # FLOPs
            try:
                sample = next(iter(self.test_dataloader))[0][:1].to(self.device)
                flops = FlopCountAnalysis(self.model, sample).total()
                metrics["test/gflops"] = flops / 1e9
            except:
                pass
            
            self._log(metrics, step)
            logger.info(f"Test F1: {metrics['test/f1']:.4f}, Time: {inference_time:.2f}s")
        
        self._save_checkpoint(step, final=True)
        if WANDB_AVAILABLE:
            wandb.finish()