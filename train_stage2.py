# train_stage2_option_a.py
# Stage 2: Train with pruning (keep_ratio<1.0), starting from Stage 1 weights

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
    parser = argparse.ArgumentParser("STAGE 2 - Train with pruning from Stage 1 weights")

    # Dataset
    parser.add_argument("--h5_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=224)

    # Training
    parser.add_argument("--output_dir", type=str, default="./results_stage2")
    parser.add_argument("--run_name", type=str, default="vit_stage2_pruned")
    parser.add_argument("--num_train_epochs", type=int, default=30)  # Fewer epochs for fine-tuning
    parser.add_argument("--learning_rate", type=float, default=1e-3)  # Lower LR
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    # Model (Stage 2: with pruning)
    parser.add_argument("--keep_ratio", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--input_aware_weight", type=float, default=0.7)

    # Stage 1 checkpoint
    parser.add_argument("--stage1_checkpoint", type=str, required=True,
                        help="Path to Stage 1 checkpoint (e.g., ./results_stage1/best_model/vit_stage1_no_pruning.bin)")

    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--log_checkpoints_to_wandb", type=bool, default=False)
    parser.add_argument("--organ", type=str, default=None, help="Train only on patches from a specific organ (e.g. lung, liver). Default: all organs")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # DATASET + UNDERSAMPLING
    # --------------------------------------------------
    logger.info("Loading dataset")

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

    num_workers = min(8, 4 * torch.cuda.device_count()) if torch.cuda.is_available() else 4
    logger.info(f"Using {num_workers} workers per DataLoader")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
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
    # MODEL (STAGE 2: WITH PRUNING)
    # --------------------------------------------------
    logger.info(f"Creating model for Stage 2 (keep_ratio={args.keep_ratio}, with pruning)")
    
    model = ViT_UCB_Pruning(
        model_name="hf-hub:MahmoodLab/uni",
        pretrained=False,  # Don't load pretrained, we'll load Stage 1
        n_classes=len(np.unique(labels)),
        keep_ratio=args.keep_ratio,  # NOW with pruning
        beta=args.beta,
        input_aware_weight=args.input_aware_weight,
        exclude_cls=True
    )

    # Load Stage 1 weights
    if not os.path.exists(args.stage1_checkpoint):
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {args.stage1_checkpoint}")
    
    logger.info(f"Loading Stage 1 weights from: {args.stage1_checkpoint}")
    state_dict = torch.load(args.stage1_checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    logger.info("Stage 1 weights loaded successfully")

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

    trainer.train()
    logger.info("STAGE 2 completed - Model fine-tuned with pruning")


if __name__ == "__main__":
    main()
