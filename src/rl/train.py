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
from sklearn.metrics import f1_score

# Set up logging
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
    output_dir: str
    run_name: str = "vit-ucb-run"
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

        num_update_steps_per_epoch = len(self.train_dataloader) // self.args.gradient_accumulation_steps
        if self.args.max_steps <= 0:
            self.args.max_steps = int(self.args.num_train_epochs * num_update_steps_per_epoch)
        else:
            self.args.num_train_epochs = math.ceil(self.args.max_steps / num_update_steps_per_epoch)

        self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model.to(self.device)

        optimizer, scheduler = optimizers
        self.optimizer = optimizer if optimizer is not None else self.create_optimizer()
        self.scheduler = scheduler if scheduler is not None else self.create_scheduler()

        self.scaler = GradScaler(enabled=self.args.fp16)
        if self.args.report_to == "wandb" and _WANDB_AVAILABLE:
            wandb.init(project="vit-ucb-pruning3", name=self.args.run_name, config=vars(self.args))
            wandb.watch(self.model, log_freq=self.args.logging_steps)

        self.best_metric = None
        self.patience_counter = 0
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

    def train(self, ucb_logging: bool = True):
        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Total optimization steps = {self.args.max_steps}")

        global_step = 0
        train_loss = 0.0
        self.model.train()

        for epoch in range(int(math.ceil(self.args.num_train_epochs))):
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{int(self.args.num_train_epochs)}", leave=False)

            for step, batch in enumerate(progress_bar):
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
                        avg_loss = train_loss / self.args.logging_steps
                        metrics_to_log = {
                            "train/loss": avg_loss,
                            "train/learning_rate": self.scheduler.get_last_lr()[0]
                        }

                        if global_step > 500 and ucb_logging:
                            with torch.no_grad():
                                ucb_scores = self.model.ucb_count_scores.detach()
                                total_elements = ucb_scores.numel()
                                zero_counts = (ucb_scores == 0).sum().item()
                                sparsity = zero_counts / total_elements

                                metrics_to_log.update({
                                    "ucb/sparsity": sparsity,
                                    "ucb/mean_selection_count": ucb_scores.mean().item(),
                                    "ucb/max_selection_count": ucb_scores.max().item(),
                                    "ucb/selection_distribution": ucb_scores
                                })

                        self._log(metrics_to_log)
                        train_loss = 0.0

                    if self.args.evaluation_strategy == "steps" and global_step % self.args.eval_steps == 0:
                        metrics = self.evaluate(counter=global_step)
                        self._log(metrics)

                if global_step >= self.args.max_steps:
                    break

            if self.args.early_stopping_patience > 0:
                logger.info(f"--- Running End-of-Epoch Evaluation for Epoch {epoch + 1} ---")
                metrics = self.evaluate(counter=global_step)
                self._log(metrics)

                if self._check_early_stopping(metrics):
                    logger.info(f"Early stopping condition met after epoch {epoch + 1}. Terminating training.")
                    break

            if global_step >= self.args.max_steps:
                break

        logger.info("Training finished. Final evaluation (validation set):")
        metrics, val_preds, val_labels = self.evaluate(counter=global_step, return_preds=True)
        self._log(metrics)

        if self.args.report_to == "wandb" and _WANDB_AVAILABLE:
            logger.info("Logging confusion matrix for validation set to W&B.")
            wandb.log({
                "val/confusion_matrix": wandb.plot.confusion_matrix(
                    preds=val_preds,
                    y_true=val_labels,
                    class_names=self.class_names
                )
            })

        if self.test_dataloader is not None:
            logger.info("Final evaluation (test set):")
            self.predict()

        self._save_checkpoint(global_step, final=True)

    def _training_step(self, batch: tuple, counter: int):
        self.model.train()
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        with autocast(enabled=self.args.fp16):
            loss, _ = self.model(x=inputs, labels=labels, counter=counter, ucb_enabled=True)
        scaled_loss = self.scaler.scale(loss / self.args.gradient_accumulation_steps)
        scaled_loss.backward()
        return loss.detach()

    def evaluate(self, counter: int, return_preds: bool = False):
        logger.info(f"***** Running Evaluation at Step {counter} *****")
        self.model.eval()

        total_loss, total_correct, total_samples = 0, 0, 0
        all_preds, all_labels = [], []

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

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        f1 = f1_score(all_labels, all_preds, average="macro")

        metrics = {"eval/loss": avg_loss, "eval/accuracy": accuracy, "eval/f1": f1}
        logger.info(f"  Evaluation results: {metrics}")

        self.model.train()
        if return_preds:
            return metrics, all_preds, all_labels
        return metrics

    def predict(self):
        if self.test_dataloader is None:
            logger.warning("Test dataloader is not provided. Skipping test evaluation.")
            return

        logger.info("***** Running Prediction on Test Set *****")
        self.model.eval()

        all_preds = []
        all_labels = []

        progress_bar = tqdm(self.test_dataloader, desc="Predicting", leave=False)
        for batch in progress_bar:
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad(), autocast(enabled=self.args.fp16):
                logits = self.model(x=inputs, counter=99999, ucb_enabled=True)
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # Calcola metriche
        total_correct = (all_preds == all_labels).sum()
        total_samples = all_labels.shape[0]
        accuracy = total_correct / total_samples
        f1 = f1_score(all_labels, all_preds, average="macro")

        metrics_to_log = {
            "test/accuracy": accuracy,
            "test/f1": f1,
        }

        # self._log(metrics_to_log) # RIMUOVI QUESTA RIGA

        # Confusion matrix va loggata direttamente
        if self.args.report_to == "wandb" and _WANDB_AVAILABLE:
            wandb.log({
                "test/confusion_matrix": wandb.plot.confusion_matrix(
                    preds=all_preds,
                    y_true=all_labels,
                    class_names=self.class_names
                )
            }) 

        # Log su W&B
        # QUESTA CHIAMATA È CORRETTA E SUFFICIENTE
        self._log(metrics_to_log)

        self.model.train()


    def _log(self, metrics: Dict[str, float]):
        if self.args.report_to == "wandb" and _WANDB_AVAILABLE:
            wandb_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    # Log solo statistiche di base
                    wandb_metrics[key + "/mean"] = value.float().mean().item()
                    wandb_metrics[key + "/max"] = value.float().max().item()
                    wandb_metrics[key + "/std"] = value.float().std().item()
                else:
                    wandb_metrics[key] = value
            wandb.log(wandb_metrics)


    def _save_checkpoint(self, step: int, final: bool = False):
        name = "final_checkpoint" if final else f"checkpoint-{step}"
        output_path = os.path.join(self.args.output_dir, name)
        os.makedirs(output_path, exist_ok=True)

        ckpt_path = os.path.join(output_path, f"{self.args.run_name}.bin")
        logger.info(f"Saving model checkpoint to {output_path}")
        torch.save(self.model.state_dict(), ckpt_path)

        # Se W&B è abilitato, salva anche l'artifact
        if self.args.report_to == "wandb" and _WANDB_AVAILABLE:
            artifact = wandb.Artifact(
                name=f"{self.args.run_name}-{name}",
                type="model"
            )
            artifact.add_file(ckpt_path)
            wandb.log_artifact(artifact)
            logger.info(f"Uploaded checkpoint {name} to W&B.")

    def _check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        metric_value = metrics.get(self.args.early_stopping_metric)
        if metric_value is None:
            logger.warning(f"Early stopping metric '{self.args.early_stopping_metric}' not found in evaluation metrics. Skipping check.")
            return False

        improvement = (metric_value - self.best_metric) if self.metric_greater_is_better else (self.best_metric - metric_value)
        if improvement > self.args.early_stopping_threshold:
            self.best_metric = metric_value
            self.patience_counter = 0
            logger.info(f"New best metric found: {self.args.early_stopping_metric} = {metric_value:.4f}. Resetting patience.")
        else:
            self.patience_counter += 1
            logger.info(f"No improvement found. Patience counter: {self.patience_counter}/{self.args.early_stopping_patience}.")

        return self.patience_counter >= self.args.early_stopping_patience
