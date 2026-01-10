# train_stage1_option_a.py
# Stage 1: Train with all patches (keep_ratio=1.0) to learn robust features

import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from transformers import get_cosine_schedule_with_warmup
import numpy as np
import pandas as pd
import argparse
import logging
import os

from src.data.dataset import PatchFromH5Dataset
from src.rl.train import ModelTrainer, TrainingArguments
from src.rl.modelling import ViT_UCB_Pruning

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




def main():
    parser = argparse.ArgumentParser("STAGE 1 - Train without pruning (keep_ratio=1.0)")

    # Dataset
    parser.add_argument("--h5_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=224)

    # Training
    parser.add_argument("--output_dir", type=str, default="./results_stage1")
    parser.add_argument("--run_name", type=str, default="vit_stage1_no_pruning")
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=12)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    # Model (Stage 1: no pruning)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--input_aware_weight", type=float, default=0.7)

    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--log_checkpoints_to_wandb", type=bool, default=False)

    parser.add_argument("--organ", type=str, default=None, help="Train only on patches from a specific organ (e.g. lung, liver). Default: all organs")


    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Aggiorna il nome della run per includere i parametri
    args.run_name = f"{args.run_name}-iaw_{args.input_aware_weight}"
    if args.organ:
        args.run_name += f"-{args.organ}"


    # --------------------------------------------------
    # DATASET + WEIGHTED RANDOM SAMPLER (NO UNDERSAMPLING)
    # --------------------------------------------------
    logger.info("Loading dataset (WeightedRandomSampler, no undersampling)")

    dataset = PatchFromH5Dataset(
        h5_dir=args.h5_dir,
        transform=transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ]),
        organ_filter=args.organ
    )

    labels = np.array(dataset.labels)
    indices = np.arange(len(labels))

    # Train / Val split (stratified, full dataset)
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.3,
        stratify=labels,
        random_state=args.seed
    )

    logger.info(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")

    # --------------------------------------------------
    # WEIGHTED RANDOM SAMPLER (TRAIN ONLY)
    # --------------------------------------------------
    train_labels = labels[train_idx]

    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts

    sample_weights = class_weights[train_labels]
    sample_weights = torch.from_numpy(sample_weights).double()

    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # --------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------
    num_workers = min(8, 4 * torch.cuda.device_count()) if torch.cuda.is_available() else 4

    logger.info(f"Using {num_workers} workers per DataLoader")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,              # Velocizza trasferimento CPU→GPU
        persistent_workers=True,      # Mantiene workers attivi tra epoche
        prefetch_factor=2,            # Pre-carica 2 batch per worker
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # --------------------------------------------------
    # MODEL (STAGE 1: NO PRUNING)
    # --------------------------------------------------
    logger.info("Creating model for Stage 1 (keep_ratio=1.0, no pruning)")
    
       
    model = ViT_UCB_Pruning(
        model_name="hf-hub:MahmoodLab/uni",
        pretrained=True,
        n_classes=len(np.unique(labels)),
        keep_ratio=1.0,
        beta=args.beta,
        input_aware_weight=args.input_aware_weight,
        exclude_cls=True
    )

    # --------------------------------------------------
    # TRAINING
    # --------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        report_to=args.report_to,
        fp16=False,
        save_best_model=True,
        early_stopping_metric="eval/f1",
        save_total_limit=2,
        log_checkpoints_to_wandb=args.log_checkpoints_to_wandb,
        model_type="ucb",
        input_aware_weight=args.input_aware_weight
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    num_steps = args.num_train_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_steps
    )

    trainer = ModelTrainer(
        model=model,
        args=training_args,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        optimizers=(optimizer, scheduler),
        device=device,
        class_names=dataset.class_names,
        organ=args.organ
    )
    
    trainer.train(starting_step=0)
    
    logger.info("STAGE 1 completed")


if __name__ == "__main__":
    main()
