
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from transformers import get_cosine_schedule_with_warmup
import numpy as np
import pandas as pd
import argparse
import os
import logging

# --- Importa i componenti necessari dal tuo progetto ---
from src.data.dataset import PatchFromH5Dataset, plot_class_distributions
from src.rl.train import ModelTrainer, TrainingArguments
from src.rl.modelling import ViT_UCB_Pruning

# Setup del logging di base
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FreezeFineTuneTrainer(ModelTrainer):
    """
    Una classe Trainer personalizzata per il fine-tuning con un set di patch fisso (pruning congelato).
    Eredita da ModelTrainer e ne sovrascrive il metodo di training (_training_step)
    per usare la logica di `forward_pruned`.
    """
    def __init__(self, *args, top_k_indices: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        
        if top_k_indices is None:
            raise ValueError("FreezeFineTuneTrainer richiede l'argomento 'top_k_indices'.")
        
        # Assicura che gli indici siano sul dispositivo corretto
        self.top_k_indices = top_k_indices.to(self.device)
        
        # Congela il buffer dei punteggi UCB per sicurezza, non deve più essere aggiornato.
        if hasattr(self.model, 'ucb_count_scores'):
            self.model.ucb_count_scores.requires_grad_(False)
        
        logger.info(f"FreezeFineTuneTrainer inizializzato. Il training avverrà su {len(self.top_k_indices)} patch fisse.")

    def _training_step(self, batch: tuple, counter: int) -> torch.Tensor:
        """
        Sovrascrive il training step per usare la logica di `forward_pruned`.
        Questo garantisce che il modello si addestri solo sul set di patch pre-calcolato,
        allineando il training alla validazione/inferenza.
        """
        self.model.train()
        inputs, labels = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        
        with torch.cuda.amp.autocast(enabled=self.args.fp16):
            # --- MODIFICA CHIAVE: si usa `forward_pruned` per il training ---
            logits = self.model.forward_pruned(inputs, self.top_k_indices)
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        # Logica di backpropagation standard
        scaled_loss = self.scaler.scale(loss / self.args.gradient_accumulation_steps)
        scaled_loss.backward()
        
        return loss.detach()


def main():
    parser = argparse.ArgumentParser(description="Fase 2: Fine-tuning di ViT-UCB con Pruning Fisso.")
    
    # --- Argomenti specifici per la Fase 2 ---
    parser.add_argument("--stage1_checkpoint_path", type=str, required=True,
                        help="Percorso al file .bin del checkpoint della Fase 1 (addestrato con keep_ratio alto).")
    parser.add_argument("--frozen_keep_ratio", type=float, required=True,
                        help="Il keep_ratio finale da usare per il pruning fisso (es. 0.3).")

    # --- Argomenti di Data-loading e Training (da training.py) ---
    parser.add_argument("--h5_dir", type=str, default="/data/patches/", help="Directory con i file H5 del dataset.")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--output_dir", type=str, default="./results_stage2_freeze", help="Directory dove salvare i risultati della Fase 2.")
    parser.add_argument("--run_name", type=str, default="ViT-Freeze-Finetune", help="Nome base per il run su W&B.")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Numero di epoche per il fine-tuning.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate per il fine-tuning (consigliato basso).")
    parser.add_argument("--train_batch_size", type=int, default=12)
    parser.add_argument("--eval_batch_size", type=int, default=12)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", type=str, default="wandb", choices=["wandb", "none"])
    parser.add_argument("--fp16", action='store_true', help="Abilita mixed precision (FP16).")

    args = parser.parse_args()

    DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Dispositivo in uso: {DEVICE}")

    # --- 1. Caricamento Dati (Logica esatta da training.py) ---
    logger.info("Fase 1: Caricamento e preparazione del dataset.")
    dataset = PatchFromH5Dataset(
        h5_dir=args.h5_dir,
        transform=transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    )
    labels = dataset.labels
    df = pd.DataFrame({"index": np.arange(len(labels)), "label": labels})
    min_count = df["label"].value_counts().min()
    undersampled_df = df.groupby("label", group_keys=False).apply(lambda x: x.sample(n=min_count, random_state=42)).reset_index(drop=True)
    undersampled_indices = undersampled_df["index"].sample(frac=1, random_state=42).tolist()
    undersampled_labels = [labels[i] for i in undersampled_indices]
    
    trainval_idx, test_idx = train_test_split(undersampled_indices, test_size=0.3, stratify=undersampled_labels, random_state=42)
    trainval_labels = [labels[i] for i in trainval_idx]
    train_idx, val_idx = train_test_split(trainval_idx, test_size=0.3, stratify=trainval_labels, random_state=42)

    train_dataset, val_dataset, test_dataset = Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)
    
    logger.info(f"Dataset splittato: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test.")

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=8, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=8, drop_last=True)

    # --- 2. Preparazione del Modello ---
    logger.info("Fase 2: Preparazione del modello.")
    labels_num = len(np.unique(dataset.labels))

    model = ViT_UCB_Pruning(
        model_name="hf-hub:MahmoodLab/uni",
        pretrained=False,
        n_classes=labels_num,
        keep_ratio=args.frozen_keep_ratio, # Cruciale: imposta il ratio che verrà usato in validazione
        exclude_cls=True
    )

    if not os.path.exists(args.stage1_checkpoint_path):
        raise FileNotFoundError(f"Checkpoint della Fase 1 non trovato a: {args.stage1_checkpoint_path}")
    
    logger.info(f"Caricamento pesi della Fase 1 da: {args.stage1_checkpoint_path}")
    model.load_state_dict(torch.load(args.stage1_checkpoint_path, map_location='cpu'))

    # --- 3. Calcolo Indici di Pruning Fissi ---
    logger.info(f"Fase 3: Calcolo indici Top-K per keep_ratio={args.frozen_keep_ratio}")
    with torch.no_grad():
        top_k_indices = model.get_top_k_patch_indices(keep_ratio=args.frozen_keep_ratio)
    logger.info(f"Indici calcolati. Verranno mantenute {len(top_k_indices)} patch (incluso il token CLS).")

    # --- 4. Configurazione del Trainer ---
    logger.info("Fase 4: Configurazione del trainer per il fine-tuning.")
    output_dir = os.path.join(args.output_dir, f"ratio_{args.frozen_keep_ratio}")
    run_name = f"{args.run_name}_ratio_{args.frozen_keep_ratio}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        evaluation_strategy="epoch",
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        seed=args.seed,
        report_to=args.report_to,
        fp16=args.fp16,
        save_best_model=True,
        early_stopping_metric="eval/f1"
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=training_args.learning_rate, weight_decay=0.0)
    num_steps = training_args.num_train_epochs * (len(train_loader) // training_args.gradient_accumulation_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=num_steps)

    trainer = FreezeFineTuneTrainer(
        model=model,
        args=training_args,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        test_dataloader=test_loader,
        class_names=dataset.class_names,
        optimizers=(optimizer, scheduler),
        device=DEVICE,
        top_k_indices=top_k_indices, # <-- Passaggio degli indici fissi
    )
    
    # --- 5. Avvio del Fine-tuning ---
    logger.info("Fase 5: Avvio del fine-tuning.")
    trainer.train(pruning_logging=False)
    logger.info("Fine-tuning completato.")


if __name__ == "__main__":
    main()
