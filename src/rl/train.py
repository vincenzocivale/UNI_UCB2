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

# --- Configurazione del Logging ---
# Imposta un logger per un output pulito e informativo.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --- Integrazione con Weights & Biases (W&B) ---
# Tenta di importare wandb e imposta un flag per verificarne la disponibilità.
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

@dataclass
class TrainingArguments:
    """Argomenti che definiscono il processo di training."""
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
    save_best_model: bool = True # CORREZIONE: Aggiunto argomento per salvare il modello migliore

# class ModelTrainer:
#     def __init__(
#         self,
#         model: nn.Module,
#         args: TrainingArguments,
#         train_dataloader: DataLoader,
#         eval_dataloader: DataLoader,
#         test_dataloader: Optional[DataLoader] = None,
#         class_names: Optional[List[str]] = None,
#         optimizers: Tuple[Optional[Optimizer], Optional[_LRScheduler]] = (None, None),
#         device: Optional[torch.device] = None
#     ):
#         self.model = model
#         self.args = args
#         self.train_dataloader = train_dataloader
#         self.eval_dataloader = eval_dataloader
#         self.test_dataloader = test_dataloader
#         self.class_names = class_names

#         # Calcola il numero totale di passi di training
#         num_update_steps_per_epoch = len(self.train_dataloader) // self.args.gradient_accumulation_steps
#         if self.args.max_steps <= 0:
#             self.args.max_steps = int(self.args.num_train_epochs * num_update_steps_per_epoch)
#         else:
#             self.args.num_train_epochs = math.ceil(self.args.max_steps / num_update_steps_per_epoch)

#         self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
#         self.model.to(self.device)

#         # Inizializza optimizer e scheduler
#         optimizer, scheduler = optimizers
#         self.optimizer = optimizer if optimizer is not None else self.create_optimizer()
#         self.scheduler = scheduler if scheduler is not None else self.create_scheduler()

#         self.scaler = GradScaler(enabled=self.args.fp16)
        
#         # Inizializza W&B se richiesto
#         if self.args.report_to == "wandb" and _WANDB_AVAILABLE:
#             wandb.init(project="vit-ucb-pruning3", name=self.args.run_name, config=vars(self.args))
#             wandb.watch(self.model, log_freq=self.args.logging_steps)

#         # CORREZIONE: Migliorata la gestione dei checkpoint
#         self._checkpoints = deque(maxlen=self.args.save_total_limit) if self.args.save_total_limit is not None and self.args.save_total_limit > 0 else None

#         # Inizializzazione per l'early stopping
#         self.patience_counter = 0
#         self.metric_greater_is_better = "loss" not in self.args.early_stopping_metric
#         self.best_metric = None
#         if self.args.early_stopping_patience > 0 or self.args.save_best_model:
#             self.best_metric = float('-inf') if self.metric_greater_is_better else float('inf')
#             logger.info(f"Monitoraggio della metrica '{self.args.early_stopping_metric}' per early stopping e/o salvataggio del modello migliore.")


#     def create_optimizer(self) -> Optimizer:
#         logger.info("Nessun optimizer fornito; creazione di un optimizer SGD di default.")
#         return SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9, weight_decay=self.args.weight_decay)

#     def create_scheduler(self) -> _LRScheduler:
#         logger.info("Nessuno scheduler fornito; creazione di uno scheduler Warmup-Cosine di default.")
#         def lr_lambda(current_step: int):
#             if current_step < self.args.warmup_steps:
#                 return float(current_step) / float(max(1, self.args.warmup_steps))
#             progress = float(current_step - self.args.warmup_steps) / float(max(1, self.args.max_steps - self.args.warmup_steps))
#             # CORREZIONE: Utilizzo di math.cos per evitare la creazione non necessaria di un tensore
#             return 0.5 * (1.0 + math.cos(math.pi * progress))
#         return LambdaLR(self.optimizer, lr_lambda)

#     def train(self, ucb_logging: bool = True):
#         logger.info("***** Inizio del Training *****")
#         logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
#         logger.info(f"  Total optimization steps = {self.args.max_steps}")

#         global_step = 0
#         train_loss = 0.0
#         self.model.train()

#         # Ciclo di training principale
#         for epoch in range(int(math.ceil(self.args.num_train_epochs))):
#             progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{int(self.args.num_train_epochs)}", leave=False)
            
#             for step, batch in enumerate(progress_bar):
#                 loss = self._training_step(batch, counter=global_step)
#                 train_loss += loss.item()

#                 if (step + 1) % self.args.gradient_accumulation_steps == 0:
#                     self.scaler.unscale_(self.optimizer)
#                     torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
#                     self.scaler.step(self.optimizer)
#                     self.scaler.update()
#                     self.scheduler.step()
#                     self.optimizer.zero_grad()
#                     global_step += 1

#                     progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"})

#                     # Logging
#                     if global_step % self.args.logging_steps == 0:
#                         metrics_to_log = {
#                             "train/loss": train_loss / self.args.logging_steps,
#                             "train/learning_rate": self.scheduler.get_last_lr()[0]
#                         }
#                         if ucb_logging and hasattr(self.model, 'ucb_count_scores'):
#                              self._log_ucb_metrics(metrics_to_log)
#                         self._log(metrics_to_log, step=global_step)
#                         train_loss = 0.0

#                     # Valutazione
#                     if self.args.evaluation_strategy == "steps" and global_step % self.args.eval_steps == 0:
#                         metrics = self.evaluate(counter=global_step)
#                         if self._check_for_improvement(metrics):
#                              self._save_checkpoint(global_step, is_best=True)
#                         self._log(metrics, step=global_step)
#                         if self._check_early_stopping():
#                             logger.info("Early stopping attivato. Interruzione del training.")
#                             self.finalize_training(global_step)
#                             return

#                     # Salvataggio del checkpoint
#                     if self.args.save_strategy == "steps" and global_step % self.args.save_steps == 0:
#                         self._save_checkpoint(global_step)

#             if self.args.evaluation_strategy == "epoch":
#                 metrics = self.evaluate(counter=global_step)
    
#                 self._log(metrics, step=global_step)
#                 if self._check_early_stopping():
#                     logger.info("Early stopping attivato. Interruzione del training.")
#                     self.finalize_training(global_step)
#                     return
            
#             if global_step >= self.args.max_steps:
#                 break
        
#         self.finalize_training(global_step)

#     def finalize_training(self, global_step: int):
#         """Esegue i passaggi finali dopo il completamento del training."""
#         logger.info("Training terminato. Esecuzione della valutazione finale.")
#         metrics, val_preds, val_labels = self.evaluate(counter=global_step, return_preds=True)
#         self._log(metrics, step=global_step)

#         if self.args.report_to == "wandb" and _WANDB_AVAILABLE:
#             logger.info("Logging della confusion matrix del set di validazione su W&B.")
#             self._log_confusion_matrix("val", val_preds, val_labels, step=global_step)

#         if self.test_dataloader is not None:
#             self.predict(step=global_step)
        
#         self._save_checkpoint(global_step, final=True)
#         if self.args.report_to == "wandb" and _WANDB_AVAILABLE:
#             wandb.finish()

#     def _training_step(self, batch: tuple, counter: int) -> torch.Tensor:
#         self.model.train()
#         inputs, labels = batch
#         inputs, labels = inputs.to(self.device), labels.to(self.device)
        
#         with autocast(enabled=self.args.fp16):
#             # CORREZIONE: Gestisce l'output del modello in modo robusto, che sia un tensore o una tupla
#             output = self.model(x=inputs, labels=labels, counter=counter, ucb_enabled=True)
#             loss = output[0] if isinstance(output, tuple) else output

#         scaled_loss = self.scaler.scale(loss / self.args.gradient_accumulation_steps)
#         scaled_loss.backward()
#         return loss.detach()

#     def evaluate(self, counter: int, return_preds: bool = False) -> Union[Dict[str, float], Tuple[Dict[str, float], list, list]]:
#         logger.info(f"***** Esecuzione Valutazione allo Step {counter} *****")
#         self.model.eval()

#         total_loss, total_correct, total_samples = 0, 0, 0
#         all_preds, all_labels = [], []

#         for batch in tqdm(self.eval_dataloader, desc="Valutazione", leave=False):
#             inputs, labels = batch
#             inputs, labels = inputs.to(self.device), labels.to(self.device)

#             with torch.no_grad(), autocast(enabled=self.args.fp16):
#                 output = self.model(x=inputs, labels=labels, counter=counter, ucb_enabled=True)
#                 loss, logits = output if isinstance(output, tuple) else (torch.tensor(0.0), output)
            
#             total_loss += loss.item() * inputs.size(0)
#             preds = torch.argmax(logits, dim=-1)
#             total_correct += (preds == labels).sum().item()
#             total_samples += inputs.size(0)

#             all_preds.append(preds.cpu())
#             all_labels.append(labels.cpu())

#         all_preds = torch.cat(all_preds).numpy()
#         all_labels = torch.cat(all_labels).numpy()

#         avg_loss = total_loss / total_samples if total_samples > 0 else 0
#         accuracy = total_correct / total_samples if total_samples > 0 else 0
#         f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

#         metrics = {"eval/loss": avg_loss, "eval/accuracy": accuracy, "eval/f1": f1}
#         logger.info(f"  Risultati valutazione: {metrics}")
        
#         self.model.train()
#         return (metrics, all_preds, all_labels) if return_preds else metrics

#     def predict(self, step: Optional[int] = None):
#         if self.test_dataloader is None:
#             logger.warning("Test dataloader non fornito. Salto della predizione sul test set.")
#             return

#         logger.info("***** Esecuzione Predizione sul Test Set *****")
#         self.model.eval()

#         all_preds, all_labels = [], []
#         for batch in tqdm(self.test_dataloader, desc="Predizione", leave=False):
#             inputs, labels = batch
#             inputs, labels = inputs.to(self.device), labels.to(self.device)

#             with torch.no_grad(), autocast(enabled=self.args.fp16):
#                 # CORREZIONE: Gestisce l'output del modello in modo robusto
#                 output = self.model(x=inputs, counter=99999, ucb_enabled=True)
#                 logits = output[1] if isinstance(output, tuple) and len(output) > 1 else output

#             preds = torch.argmax(logits, dim=-1)
#             all_preds.append(preds.cpu())
#             all_labels.append(labels.cpu())

#         all_preds = torch.cat(all_preds).numpy()
#         all_labels = torch.cat(all_labels).numpy()

#         accuracy = (all_preds == all_labels).sum() / all_labels.shape[0]
#         f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

#         metrics = {"test/accuracy": accuracy, "test/f1": f1}
#         self._log(metrics, step=step)
#         logger.info(f"  Risultati test set: {metrics}")

#         if self.args.report_to == "wandb" and _WANDB_AVAILABLE:
#             self._log_confusion_matrix("test", all_preds, all_labels, step=step)
        
#         self.model.train()
    
#     def _log_ucb_metrics(self, metrics_dict: Dict):
#         """Logga le metriche specifiche UCB."""
#         with torch.no_grad():
#             ucb_scores = self.model.ucb_count_scores.detach().float()
#             metrics_dict.update({
#                 "ucb/sparsity": (ucb_scores == 0).sum().item() / ucb_scores.numel(),
#                 "ucb/mean_selection_count": ucb_scores.mean().item(),
#                 "ucb/max_selection_count": ucb_scores.max().item(),
#                 # CORREZIONE: Logga un istogramma invece del tensore grezzo
#                 "ucb/selection_distribution": wandb.Histogram(ucb_scores.cpu().numpy()) if _WANDB_AVAILABLE else ucb_scores.cpu().numpy().tolist()
#             })

#     def _log(self, metrics: Dict[str, float], step: Optional[int] = None):
#         """Logga le metriche su W&B."""
#         if self.args.report_to == "wandb" and _WANDB_AVAILABLE:
#             wandb.log(metrics, step=step)
    
#     def _log_confusion_matrix(self, prefix: str, preds: list, labels: list, step: Optional[int] = None):
#         """Logga una confusion matrix su W&B."""
#         wandb.log({
#             f"{prefix}/confusion_matrix": wandb.plot.confusion_matrix(
#                 preds=preds, y_true=labels, class_names=self.class_names
#             )
#         }, step=step)

#     def _save_checkpoint(self, step: int, final: bool = False, is_best: bool = False):
#         if is_best:
#             name = "best_model"
#         elif final:
#             name = "final_checkpoint"
#         else:
#             name = f"checkpoint-{step}"
        
#         output_path = os.path.join(self.args.output_dir, name)
#         os.makedirs(output_path, exist_ok=True)
#         ckpt_path = os.path.join(output_path, f"{self.args.run_name}.bin")
        
#         logger.info(f"Salvataggio del checkpoint del modello in {output_path}")
#         torch.save(self.model.state_dict(), ckpt_path)
        
#         # CORREZIONE: Gestisce il limite di checkpoint da salvare
#         if self._checkpoints is not None and not is_best and not final:
#             if len(self._checkpoints) == self._checkpoints.maxlen:
#                 oldest_ckpt = self._checkpoints.popleft()
#                 if os.path.exists(oldest_ckpt):
#                     logger.info(f"Rimozione del vecchio checkpoint: {oldest_ckpt}")
#                     shutil.rmtree(oldest_ckpt)
#             self._checkpoints.append(output_path)
            
#         if self.args.report_to == "wandb" and _WANDB_AVAILABLE:
#             aliases = []
#             if final: aliases.append("final")
#             if is_best: aliases.append("best")
#             self._log_artifact_to_wandb(name, ckpt_path, aliases)
    
#     def _log_artifact_to_wandb(self, name: str, path: str, aliases: List[str]):
#         """Logga un artifact del modello su W&B."""
#         artifact = wandb.Artifact(name=f"{self.args.run_name}", type="model", metadata={"step": name})
#         artifact.add_file(path)
#         wandb.log_artifact(artifact, aliases=aliases)
#         logger.info(f"Checkpoint '{name}' caricato su W&B con alias: {aliases}.")

#     def _check_for_improvement(self, metrics: Dict[str, float]) -> bool:
#         """Controlla se la metrica monitorata è migliorata."""
#         metric_value = metrics.get(self.args.early_stopping_metric)
#         if metric_value is None:
#             return False

#         improvement = (metric_value - self.best_metric) if self.metric_greater_is_better else (self.best_metric - metric_value)
#         if improvement > self.args.early_stopping_threshold:
#             self.best_metric = metric_value
#             self.patience_counter = 0
#             logger.info(f"Nuova metrica migliore trovata: {self.args.early_stopping_metric} = {metric_value:.4f}. Reset della pazienza.")
#             return True
#         else:
#             self.patience_counter += 1
#             logger.info(f"Nessun miglioramento. Contatore pazienza: {self.patience_counter}/{self.args.early_stopping_patience}.")
#             return False

#     def _check_early_stopping(self) -> bool:
#         """Verifica se la condizione di early stopping è soddisfatta."""
#         if self.args.early_stopping_patience <= 0:
#             return False
#         return self.patience_counter >= self.args.early_stopping_patience

from transformers import Trainer
import torch

class ViTUCBTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Inizializziamo un contatore di step di training
        self.training_step_counter = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Sovrascriviamo questo metodo per passare gli argomenti extra al modello.
        """
        # Estraiamo gli input standard dal dizionario del dataset
        labels = inputs.get("labels")
        pixel_values = inputs.get("pixel_values")

        # Determiniamo se siamo in fase di training o di valutazione
        is_training = model.training

        # Chiamiamo il forward del modello con gli argomenti custom
        # Durante il training, ucb è abilitato e il contatore viene usato
        # Durante la valutazione (model.eval()), potresti voler disabilitare ucb
        outputs = model(
            pixel_values=pixel_values,
            labels=labels,
            counter=self.training_step_counter,
            ucb_enabled=is_training  # Abilita UCB solo in training
        )

        # Se il modello è in training, incrementiamo il nostro contatore
        if is_training:
            self.training_step_counter += 1

        # Il tuo modello restituisce una tupla (loss, logits)
        # Il Trainer di HF si aspetta che la loss sia il primo elemento
        loss = outputs[0]
        
        # Restituiamo la loss e, se richiesto, gli output completi
        return (loss, outputs) if return_outputs else loss