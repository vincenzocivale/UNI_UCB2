import logging
import os
import random
import time
from datetime import timedelta, datetime # Importa datetime
from contextlib import suppress
import sys # Importa sys per StreamHandler

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm
import wandb
import math # Importa math per le schedule


logger = logging.getLogger(__name__)

class TrainingArguments:
    """
    Arguments for configuring the training process.
    """
    def __init__(self, **kwargs):
        # Modifiche qui
        self.project_name: str = kwargs.pop("project_name", "vision-transformer-training") # Nuovo parametro: nome del progetto W&B
        self.name: str = kwargs.pop("name", None) # Sarà generato se None
        self.output_dir: str = kwargs.pop("output_dir", "output")
        self.eval_every: int = kwargs.pop("eval_every", 100)
        self.learning_rate: float = kwargs.pop("learning_rate", 3e-2)
        self.weight_decay: float = kwargs.pop("weight_decay", 0.0)
        self.num_steps: int = kwargs.pop("num_steps", 10000)
        self.num_train_epochs: float = kwargs.pop("num_train_epochs", 3.0)
        self.decay_type: str = kwargs.pop("decay_type", "cosine")
        self.warmup_steps: int = kwargs.pop("warmup_steps", 500)
        self.max_grad_norm: float = kwargs.pop("max_grad_norm", 1.0)
        self.local_rank: int = kwargs.pop("local_rank", -1)
        self.seed: int = kwargs.pop("seed", 42)
        self.gradient_accumulation_steps: int = kwargs.pop("gradient_accumulation_steps", 1)
        self.fp16: bool = kwargs.pop("fp16", False)
        self.num_classes: int = kwargs.pop("num_classes", 21843)
        self.logging_steps: int = kwargs.pop("logging_steps", 50)

        self.img_size: int = kwargs.pop("img_size", 224)
        self.train_batch_size: int = kwargs.pop("train_batch_size", 64)

        self.n_gpu: int = 0
        self.device: torch.device = None

        self._setup_device()
        self._setup_logging()

        # Genera il nome della run solo dopo che il logger è stato configurato e il processo è il main
        if self.name is None:
            self.name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_run"
            logger.info(f"Run name automatically set to: {self.name}")


    def _setup_device(self):
        if self.local_rank == -1:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                self.n_gpu = 1
                logger.info(f"Using single GPU or DataParallel: {self.device}")
            else:
                self.device = torch.device("cpu")
                self.n_gpu = 0
                logger.info("CUDA is not available, using CPU.")
        else:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            dist.init_process_group(backend="nccl")
            self.n_gpu = 1
            logger.info(f"Using DistributedDataParallel on GPU: {self.device}")


        logger.warning(
            f"Process rank: {self.local_rank}, device: {self.device}, n_gpu: {self.n_gpu}, "
            f"distributed training: {bool(self.local_rank != -1)}, 16-bits training: {self.fp16}"
        )

    def _setup_logging(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if self.is_main_process() else logging.WARN,
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    def is_main_process(self) -> bool:
        return self.local_rank in [-1, 0]

    def __str__(self):
        s = f"{self.__class__.__name__}(\n"
        for k, v in self.__dict__.items():
            s += f"    {k} = {v},\n"
        s += ")"
        return s

# Definisci le classi WarmupCosineSchedule e WarmupLinearSchedule (come nel tuo codice originale)
class WarmupCosineSchedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (float(self.last_epoch) / self.warmup_steps) for base_lr in self.base_lrs]
        return [base_lr * 0.5 * (1.0 + math.cos(math.pi * (float(self.last_epoch) - self.warmup_steps) / (self.t_total - self.warmup_steps))) for base_lr in self.base_lrs]

class WarmupLinearSchedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (float(self.last_epoch) / self.warmup_steps) for base_lr in self.base_lrs]
        return [base_lr * max(0.0, float(self.t_total - self.last_epoch) / (self.t_total - self.warmup_steps)) for base_lr in self.base_lrs]

class Trainer:
    def __init__(
        self,
        args: TrainingArguments,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: torch.utils.data.DataLoader,
        loss_function: nn.Module,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None
    ):
        self.args = args
        self._set_seed(self.args)

        if self.args.is_main_process():
            os.makedirs(self.args.output_dir, exist_ok=True)
            # Usa self.args.project_name e self.args.name
            wandb.init(project=self.args.project_name, name=self.args.name, config=vars(self.args))
            self.writer = wandb
        else:
            self.writer = None

        self.model = model.to(self.args.device)
        if self.args.local_rank != -1:
            self.model = DDP(self.model, device_ids=[self.args.local_rank])
        elif self.args.n_gpu > 1:
            self.model = nn.DataParallel(self.model)
            logger.info(f"Using nn.DataParallel on {self.args.n_gpu} GPUs.")

        self.train_loader = train_dataloader
        self.test_loader = eval_dataloader
        self.loss_fct = loss_function

        self.num_training_steps_per_epoch = len(self.train_loader) // self.args.gradient_accumulation_steps
        self.args.num_steps = int(self.args.num_train_epochs * self.num_training_steps_per_epoch)
        logger.info(f"Calculated total training steps: {self.args.num_steps} over {self.args.num_train_epochs} epochs.")

        self.optimizer = optimizer if optimizer else self._setup_optimizer()
        self.scheduler = scheduler if scheduler else self._setup_scheduler(self.optimizer)

        self.scaler = GradScaler() if self.args.fp16 else None

        self.global_step = 0
        self.best_acc = 0.0
        self.step_counter_for_ucb = 0
        self.eval_losses_history = []
        self.train_losses_history = []

        logger.info("Training arguments: %s", self.args)
        logger.info("Total parameters: %2.1fM" % self._count_parameters(self.model))


    @staticmethod
    def _set_seed(args: TrainingArguments):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

    @staticmethod
    def _count_parameters(model: nn.Module) -> float:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return params / 1_000_000

    @staticmethod
    def _simple_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
        return (preds == labels).mean()

    def _setup_optimizer(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.args.learning_rate,
            momentum=0.9,
            weight_decay=self.args.weight_decay,
        )
        return optimizer

    def _setup_scheduler(self, optimizer):
        t_total = self.args.num_steps
        if self.args.decay_type == "cosine":
            scheduler = WarmupCosineSchedule(
                optimizer, warmup_steps=self.args.warmup_steps, t_total=t_total
            )
        else:
            scheduler = WarmupLinearSchedule(
                optimizer, warmup_steps=self.args.warmup_steps, t_total=t_total
            )
        return scheduler

    def _save_model(self):
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_checkpoint = os.path.join(self.args.output_dir, f"{self.args.name}_checkpoint.bin")
        torch.save(model_to_save.state_dict(), model_checkpoint)
        logger.info("Saved model checkpoint to [DIR: %s]", self.args.output_dir)

    def _cleanup_distributed(self):
        if self.args.local_rank != -1:
            dist.destroy_process_group()

    def validate(self, epoch: int):
        eval_losses = AverageMeter()

        logger.info(f"***** Running Validation for Epoch {epoch} *****")
        logger.info("  Num examples = %d", len(self.test_loader.dataset) if hasattr(self.test_loader, 'dataset') else len(self.test_loader) * self.test_loader.batch_size)
        logger.info("  Batch size = %d", self.test_loader.batch_size)

        self.model.eval()
        all_preds, all_label = [], []
        epoch_iterator = tqdm(
            self.test_loader,
            desc=f"Validating Epoch {epoch}... (loss=X.X)",
            bar_format="{l_bar}{r_bar}",
            dynamic_ncols=True,
            disable=not self.args.is_main_process(),
        )

        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(self.args.device) for t in batch)
            x, y = batch

            with torch.no_grad():
                current_batch_size = x.shape[0]
                block_size = (self.args.img_size * self.args.img_size) // (16 * 16) + 1

                num_heads = 16 # Esempio, adatta questo al tuo modello

                dummy_ucb_count_score = torch.ones(
                    current_batch_size,
                    num_heads,
                    block_size,
                    block_size,
                    requires_grad=False,
                ).to(self.args.device)

                with autocast() if self.args.fp16 else suppress():
                    logits = self.model(
                        x=x,
                        counter=self.step_counter_for_ucb,
                        ucb=False,
                    )[0]

                eval_loss = self.loss_fct(logits, y)
                eval_losses.update(eval_loss.item())

                preds = torch.argmax(logits, dim=-1)

            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
            epoch_iterator.set_description(f"Validating Epoch {epoch}... (loss={eval_losses.val:.5f})")

        all_preds_np = np.concatenate(all_preds, axis=0)
        all_label_np = np.concatenate(all_label, axis=0)
        accuracy = self._simple_accuracy(all_preds_np, all_label_np)

        if self.args.is_main_process():
            self.writer.log({
                "eval/loss": eval_losses.avg,
                "eval/accuracy": accuracy,
                "global_step": self.global_step,
                "epoch": epoch,
            })
            logger.info(f"Epoch {epoch} - Global Steps: {self.global_step} - Valid Loss: {eval_losses.avg:.5f} - Valid Accuracy: {accuracy:.5f}")

        self.eval_losses_history.append(eval_losses.avg)
        return accuracy, eval_losses.avg

    def train(self):
        """Train the model."""
        logger.info("***** Running training *****")
        logger.info("  Total training epochs = %d", self.args.num_train_epochs)
        logger.info("  Total optimization steps = %d", self.args.num_steps)
        logger.info("  Instantaneous batch size per GPU = %d", self.train_loader.batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            self.train_loader.batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1),
        )
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)

        self.model.zero_grad()
        train_losses = AverageMeter()

        start_time = time.time()
        completed_epochs = 0
        steps_trained_this_epoch = 0

        for epoch_idx in range(int(self.args.num_train_epochs)):
            if self.global_step >= self.args.num_steps:
                break

            self.model.train()
            epoch_iterator = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch_idx+1}/{int(self.args.num_train_epochs)} Training ({self.global_step} / {self.args.num_steps} Steps) (Loss: X.X)",
                bar_format="{l_bar}{r_bar}",
                dynamic_ncols=True,
                disable=not self.args.is_main_process(),
            )
            
            train_losses.reset()
            steps_trained_this_epoch = 0

            for step, batch in enumerate(epoch_iterator):
                if self.global_step >= self.args.num_steps:
                    break

                self.step_counter_for_ucb += 1
                batch = tuple(t.to(self.args.device) for t in batch)
                x, y = batch

                with autocast() if self.args.fp16 else suppress():
                    model_output = self.model(
                        x=x,
                        counter=self.step_counter_for_ucb,
                        ucb=True,
                    )
                    logits = model_output[0]
                    count_output = model_output[1] if len(model_output) > 1 else None

                    loss = self.loss_fct(logits, y)

                loss_item = loss.mean() if self.args.n_gpu > 1 and not isinstance(self.model, DDP) else loss
                loss_item = loss_item / self.args.gradient_accumulation_steps

                if self.args.fp16:
                    self.scaler.scale(loss_item).backward()
                else:
                    loss_item.backward()

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (step + 1) == len(self.train_loader):
                    if self.args.fp16:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.max_grad_norm
                        )
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    steps_trained_this_epoch += 1

                    train_losses.update(loss.item())

                    if self.global_step % self.args.logging_steps == 0 and self.args.is_main_process():
                        current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.optimizer.param_groups[0]['lr']
                        self.writer.log({
                            "train/loss": train_losses.avg,
                            "train/lr": current_lr,
                            "global_step": self.global_step,
                            "epoch": epoch_idx + 1,
                        })
                        logger.info(
                            f"Epoch {epoch_idx+1} Step {self.global_step}/{self.args.num_steps} - "
                            f"Loss: {train_losses.avg:.5f} - "
                            f"LR: {current_lr:.6f}"
                        )
                        train_losses.reset()

                    if not isinstance(count_output, int) and self.args.is_main_process():
    
                        self.writer.log({
                            "ucb/count_output_min": count_output.min().item(),
                            "ucb/count_output_max": count_output.max().item(),
                            "ucb/count_output_mean": count_output.mean().item(),
                            "global_step": self.global_step
                        })

                epoch_iterator.set_description(
                    f"Epoch {epoch_idx+1}/{int(self.args.num_train_epochs)} Training ({self.global_step} / {self.args.num_steps} Steps) (Loss: {train_losses.val:.5f})"
                )

                if self.global_step >= self.args.num_steps:
                    break

            if self.args.is_main_process():
                accuracy, eval_loss = self.validate(epoch_idx + 1)
                self.train_losses_history.append(train_losses.avg)
                if self.best_acc < accuracy:
                    self._save_model()
                    self.best_acc = accuracy
                self.model.train()

            completed_epochs += 1

        end_time = time.time()
        logger.info(f"Training finished in {timedelta(seconds=int(end_time - start_time))}")
        logger.info("Best Accuracy: \t%f", self.best_acc)
        logger.info("End Training!")

        if self.args.is_main_process():
            self.writer.finish()
            np.save(os.path.join(self.args.output_dir, "train_losses.npy"), np.array(self.train_losses_history))
            np.save(os.path.join(self.args.output_dir, "eval_losses.npy"), np.array(self.eval_losses_history))

        self._cleanup_distributed()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count