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
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=12)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    # Model (Stage 1: no pruning)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--input_aware_weight", type=float, default=0.7)

    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--log_checkpoints_to_wandb", type=bool, default=False)

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
        ])
    )

    labels = dataset.labels
    df = pd.DataFrame({
        "index": np.arange(len(labels)),
        "label": labels
    })

    # Undersampling
    min_count = df["label"].value_counts().min()
    logger.info(f"Undersampling to {min_count} samples per class")

    undersampled_df = (
        df.groupby("label", group_keys=False)
          .apply(lambda x: x.sample(n=min_count, random_state=args.seed))
          .reset_index(drop=True)
    )

    undersampled_indices = undersampled_df["index"].values
    undersampled_labels = undersampled_df["label"].values

    # Global shuffle
    perm = np.random.RandomState(args.seed).permutation(len(undersampled_indices))
    undersampled_indices = undersampled_indices[perm]
    undersampled_labels = undersampled_labels[perm]

    # Train/val split
    train_idx, val_idx = train_test_split(
        undersampled_indices,
        test_size=0.3,
        stratify=undersampled_labels,
        random_state=args.seed
    )

    logger.info(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False
    )

    # --------------------------------------------------
    # MODEL (STAGE 1: NO PRUNING)
    # --------------------------------------------------
    logger.info("Creating model for Stage 1 (keep_ratio=1.0, no pruning)")
    
    model = ViT_UCB_Pruning(
        model_name="hf-hub:MahmoodLab/uni",
        pretrained=True,
        n_classes=len(np.unique(labels)),
        keep_ratio=1.0,  # NO PRUNING in Stage 1
        beta=args.beta,
        input_aware_weight=args.input_aware_weight,  # Doesn't affect training
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
        class_names=dataset.class_names
    )

    trainer.train()
    logger.info("STAGE 1 completed - Model trained without pruning")


if __name__ == "__main__":
    main()
