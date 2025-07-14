import logging
import os
import random
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast


from tqdm import tqdm
import wandb
from contextlib import suppress


logger = logging.getLogger(__name__)

class TrainingArguments:
    """
    Arguments for configuring the training process.
    """
    def __init__(self, **kwargs):
        self.name: str = kwargs.pop("name", "default_run")
        self.output_dir: str = kwargs.pop("output_dir", "output")
        self.eval_every: int = kwargs.pop("eval_every", 100)
        self.learning_rate: float = kwargs.pop("learning_rate", 3e-2)
        self.weight_decay: float = kwargs.pop("weight_decay", 0.0)
        self.num_steps: int = kwargs.pop("num_steps", 10000)
        self.decay_type: str = kwargs.pop("decay_type", "cosine")
        self.warmup_steps: int = kwargs.pop("warmup_steps", 500)
        self.max_grad_norm: float = kwargs.pop("max_grad_norm", 1.0)
        self.local_rank: int = kwargs.pop("local_rank", -1)
        self.seed: int = kwargs.pop("seed", 42)
        self.gradient_accumulation_steps: int = kwargs.pop("gradient_accumulation_steps", 1)
        self.fp16: bool = kwargs.pop("fp16", False)
        self.num_classes: int = kwargs.pop("num_classes", 21843)

        # Parametri per il logging o per le dimensioni se non specificati dal modello/dataloader
        self.img_size: int = kwargs.pop("img_size", 224) # Utile per UCB_Count_Score
        self.train_batch_size: int = kwargs.pop("train_batch_size", 64) # Necessario per UCB_Count_Score

        # Derived properties (verranno impostate in _setup_device)
        self.n_gpu: int = 0
        self.device: torch.device = None

        self._setup_device()
        self._setup_logging()

    def _setup_device(self):
        # Questa logica ora gestirà solo il caso non-DDP (local_rank == -1)
        # Se vuoi specificare una GPU precisa (es. cuda:0) anche se ce ne sono più disponibili
        if torch.cuda.is_available():
    
            self.device = torch.device("cuda:0") # Usa esplicitamente la GPU con indice 0
            self.n_gpu = 1 # Stiamo usando una singola GPU
            logger.info(f"Using single GPU: {self.device}")
        else:
            self.device = torch.device("cpu")
            self.n_gpu = 0
            logger.info("CUDA is not available, using CPU.")

        logger.warning(
            f"Process rank: {self.local_rank}, device: {self.device}, n_gpu: {self.n_gpu}, "
            f"distributed training: {bool(self.local_rank != -1)}, 16-bits training: {self.fp16}"
        )

    def _setup_logging(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if self.is_main_process() else logging.WARN,
        )

    def is_main_process(self) -> bool:
        return self.local_rank in [-1, 0]

    def __str__(self):
        s = f"{self.__class__.__name__}(\n"
        for k, v in self.__dict__.items():
            s += f"    {k} = {v},\n"
        s += ")"
        return s


class Trainer:
    def __init__(
        self,
        args: TrainingArguments,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: torch.utils.data.DataLoader,
        loss_function: nn.Module,
        optimizer: torch.optim.Optimizer = None, # Possibilità di passare un optimizer predefinito
        scheduler: torch.optim.lr_scheduler._LRScheduler = None # Possibilità di passare uno scheduler predefinito
    ):
        self.args = args
        self._set_seed(self.args)

        if self.args.is_main_process():
            os.makedirs(self.args.output_dir, exist_ok=True)
            wandb.init(project="vision-transformer-training", name=self.args.name, config=vars(self.args))
            self.writer = wandb
        else:
            self.writer = None

        self.model = model.to(self.args.device) # Sposta il modello sul device
        if self.args.local_rank != -1:
            self.model = DDP(self.model, device_ids=[self.args.local_rank])
        elif self.args.n_gpu > 1:
            self.model = nn.DataParallel(self.model)
            logger.info(f"Using nn.DataParallel on {self.args.n_gpu} GPUs.")

        self.train_loader = train_dataloader
        self.test_loader = eval_dataloader
        self.loss_fct = loss_function

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
        # logger.info("Model configuration: %s", CONFIGS[self.args.model_type]) # Non più necessario se il modello è esterno


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
        # Questo metodo viene usato solo se l'optimizer non è stato passato esternamente
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.args.learning_rate,
            momentum=0.9,
            weight_decay=self.args.weight_decay,
        )
        return optimizer

    def _setup_scheduler(self, optimizer):
        # Questo metodo viene usato solo se lo scheduler non è stato passato esternamente
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

        logger.info("***** Running Validation *****")
        logger.info("  Num steps = %d", len(self.test_loader))
        logger.info("  Batch size = %d", self.test_loader.batch_size) # Usa batch_size dal dataloader

        self.model.eval()
        all_preds, all_label = [], []
        epoch_iterator = tqdm(
            self.test_loader,
            desc="Validating... (loss=X.X)",
            bar_format="{l_bar}{r_bar}",
            dynamic_ncols=True,
            disable=not self.args.is_main_process(),
        )

        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(self.args.device) for t in batch)
            x, y = batch
            with torch.no_grad():
                # UCB_Count_Score: è importante che la dimensione del batch sia corretta
                # Potrebbe essere necessario passare il batch_size effettivo del dataloader di test
                # o usare x.shape[0] per la dimensione del batch corrente.
                # Assumiamo che il modello possa gestire UCB_Count_Score per batch variabili
                # o che non sia usato in validazione con lo stesso meccanismo.
                # Qui usiamo un placeholder basato sul batch corrente.
                current_batch_size = x.shape[0]
                block_size = (self.args.img_size * self.args.img_size) // (16 * 16) + 1
                dummy_ucb_count_score = torch.ones(
                    current_batch_size,
                    16, # Numero di attention heads (dipende dal tuo modello)
                    block_size,
                    block_size,
                    requires_grad=False,
                ).to(self.args.device)

                with autocast() if self.args.fp16 else torch.no_grad():
                    logits = self.model(
                        x=x,
                        counter=self.step_counter_for_ucb,
                        UCB_Count_Score=dummy_ucb_count_score,
                        ucb=True # o False, a seconda di come gestisci UCB in validazione
                    )[0]
                eval_loss = self.loss_fct(logits, y) # Usa la loss function passata
                eval_losses.update(eval_loss.item())

                preds = torch.argmax(logits, dim=-1)

            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
            epoch_iterator.set_description(f"Validating... (loss={eval_losses.val:.5f})")

        all_preds_np = np.concatenate(all_preds, axis=0)
        all_label_np = np.concatenate(all_label, axis=0)
        accuracy = self._simple_accuracy(all_preds_np, all_label_np)

        logger.info("\n")
        logger.info("***** Validation Results *****")
        logger.info("  Global Steps: %d", self.global_step)
        logger.info("  Valid Loss: %2.5f", eval_losses.avg)
        logger.info("  Valid Accuracy: %2.5f", accuracy)

        if self.args.is_main_process():
            self.writer.log({
                "eval/loss": eval_losses.avg,
                "eval/accuracy": accuracy,
                "global_step": self.global_step
            })
        self.eval_losses_history.append(eval_losses.avg)

        return accuracy, eval_losses.avg

    def train(self):
        """Train the model."""
        logger.info("***** Running training *****")
        logger.info("  Total optimization steps = %d", self.args.num_steps)
        # Il batch size istantaneo per GPU dovrebbe essere dedotto dal dataloader,
        # ma se è fisso e uguale per tutti i processi DDP, args.train_batch_size è ok.
        # Altrimenti, self.train_loader.batch_size.
        logger.info("  Instantaneous batch size per GPU = %d", self.train_loader.batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            self.train_loader.batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1),
        )
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)

        self.model.zero_grad()
        losses = AverageMeter()

        # Inizializza Count_Score, usando il batch_size effettivo del dataloader di training
        # Assumendo che sia costante per il training
        block_size = (self.args.img_size * self.args.img_size) // (16 * 16) + 1
        Count_Score = torch.ones(
            self.train_loader.batch_size, # Usa il batch_size del dataloader di training
            16, # Numero di attention heads (dipende dal tuo modello)
            block_size,
            block_size,
            requires_grad=False,
        ).to(self.args.device)


        start_time = time.time()
        # Il loop esterno può essere basato su epoche o su `num_steps`.
        # Dato che `num_steps` è il totale, useremo un loop while per i global_step.
        epoch_idx = 0
        while self.global_step < self.args.num_steps:
            self.model.train()
            epoch_iterator = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch_idx+1} Training ({self.global_step} / {self.args.num_steps} Steps) (loss=X.X)",
                bar_format="{l_bar}{r_bar}",
                dynamic_ncols=True,
                disable=not self.args.is_main_process(),
            )
            for step, batch in enumerate(epoch_iterator):
                if self.global_step >= self.args.num_steps:
                    break # Break inner loop if total steps reached

                self.step_counter_for_ucb += 1
                batch = tuple(t.to(self.args.device) for t in batch)
                x, y = batch

                with autocast() if self.args.fp16 else suppress(): # 'suppress' è un placeholder, o semplicemente non usare il contesto
                    logits, count = self.model(
                        x=x,
                        counter=self.step_counter_for_ucb,
                        ucb=True,
                        UCB_Count_Score=Count_Score,
                    )
                    loss = self.loss_fct(logits, y)

                if not isinstance(count, int):
                    # Clona e detach per evitare che i gradienti scorrano indietro in Count_Score
                    Count_Score = count.clone().detach()

                loss_item = loss.mean() if self.args.n_gpu > 1 and not isinstance(self.model, DDP) else loss
                # La divisione per gradient_accumulation_steps deve essere fatta sulla loss *prima* di backward()
                # se stai accumulando i gradienti per media (come qui)
                loss_item = loss_item / self.args.gradient_accumulation_steps # Assicurati che loss_item sia un float32 qui

                if self.args.fp16:
                    self.scaler.scale(loss_item).backward()
                else:
                    # Ora loss_item dovrebbe avere requires_grad=True
                    loss_item.backward()

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
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
                    losses.update(loss_item.item() * self.args.gradient_accumulation_steps)

                    epoch_iterator.set_description(
                        f"Epoch {epoch_idx+1} Training ({self.global_step} / {self.args.num_steps} Steps) (loss={losses.val:.5f})"
                    )

                    if self.args.is_main_process():
                        self.writer.log({
                            "train/loss": losses.val,
                            "train/lr": self.scheduler.get_lr()[0],
                            "global_step": self.global_step
                        })

                    if self.global_step % self.args.eval_every == 0 and self.args.is_main_process():
                        accuracy, eval_loss = self.validate(self.step_counter_for_ucb)
                        self.train_losses_history.append(losses.avg)

                        if self.best_acc < accuracy:
                            self._save_model()
                            self.best_acc = accuracy
                        self.model.train()

            epoch_idx += 1
            losses.reset() # Reset epoch losses

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