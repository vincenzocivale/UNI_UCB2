import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
from transformers import TrainingArguments, get_cosine_schedule_with_warmup, EarlyStoppingCallback
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.h5_dataset import H5PatchDataset
from src.models.pruning_model import VisionTransformerUCB
from src.trainer.ucb_trainer import UcbTrainer, compute_metrics

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Dataset Preparation ---
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = H5PatchDataset(h5_dir=args.data_dir, transform=transform)

    # --- Undersampling and Splitting ---
    df = pd.DataFrame({"index": np.arange(len(dataset.labels)), "label": dataset.labels})
    min_count = df["label"].value_counts().min()
    
    undersampled_df = (
        df.groupby("label", group_keys=False)
          .apply(lambda x: x.sample(n=min_count, random_state=args.seed))
          .reset_index(drop=True)
    )
    undersampled_indices = undersampled_df["index"].sample(frac=1, random_state=args.seed).tolist()
    undersampled_labels = [dataset.labels[i] for i in undersampled_indices]

    trainval_idx, test_idx = train_test_split(
        undersampled_indices, test_size=0.3, stratify=undersampled_labels, random_state=args.seed
    )
    trainval_labels = [dataset.labels[i] for i in trainval_idx]
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=0.3, stratify=trainval_labels, random_state=args.seed
    )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")

    # --- Model ---
    num_classes = len(np.unique(dataset.labels))
    model = VisionTransformerUCB(
        model_name="hf-hub:MahmoodLab/uni",
        pretrained=True,
        n_classes=num_classes,
        keep_ratio=args.dynamic_keep_ratio,
        exclude_cls=True,
        selection_mode='ucb'
    )

    # --- Trainer ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        warmup_steps=500,
        report_to="wandb" if args.use_wandb else "none",
        fp16=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    
    num_training_steps = args.num_epochs * (len(train_dataset) // args.batch_size)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=500, 
        num_training_steps=num_training_steps
    )

    trainer = UcbTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()

    if test_dataset:
        test_results = trainer.predict(test_dataset)
        trainer.log_metrics("test", test_results.metrics)
        trainer.save_metrics("test", test_results.metrics)

    print("--- Dynamic Pruning Training Finished ---")
    print(f"Best model checkpoint saved to {trainer.state.best_model_checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training with Dynamic UCB-based Patch Selection.")
    parser.add_argument("--data_dir", type=str, default="/data/patches/", help="Directory with H5 patch files.")
    parser.add_argument("--stage1_checkpoint_path", type=str, default=None, help="Path to the Phase 1 checkpoint directory from Hugging Face Trainer (optional).")
    parser.add_argument("--output_dir", type=str, default="./results/dynamic_pruning", help="Directory to save checkpoints and results.")
    parser.add_argument("--run_name", type=str, default="vit-dynamic-pruning", help="Name for the WandB run.")
    parser.add_argument("--img_size", type=int, default=224, help="Image size.")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for training and evaluation.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--dynamic_keep_ratio", type=float, default=0.3, help="Keep ratio for dynamic pruning during training.")
    parser.add_argument("--model_name", type=str, default="hf-hub:MahmoodLab/uni", help="Name of the pretrained model from timm.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--use_wandb", action='store_true', help="Flag to enable WandB logging.")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Patience for early stopping.")

    args = parser.parse_args()
    main(args)
