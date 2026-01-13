"""
Two-stage training pipeline for UCB ViT - Simplified version
Uses pre-split datasets from data/DATASET_NAME/{train,val,test}
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import get_cosine_schedule_with_warmup
import numpy as np
import argparse
from pathlib import Path

from src.dataset import HistologicalImageDataset
from src.trainer import ModelTrainer, TrainingArguments
from src.modelling import ViT_UCB_Pruning


def create_balanced_sampler(dataset):
    """Create weighted sampler for class balancing"""
    class_counts = np.bincount(dataset.labels)
    class_weights = 1.0 / class_counts
    sample_weights = torch.from_numpy(class_weights[dataset.labels]).double()
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def load_datasets(base_dir, transform):
    """Load pre-split train/val/test datasets"""
    base_path = Path(base_dir)
    
    train_dataset = HistologicalImageDataset(
        data_dir=str(base_path / "train"),
        transform=transform
    )
    
    val_dataset = HistologicalImageDataset(
        data_dir=str(base_path / "val"),
        transform=transform
    )
    
    test_dataset = HistologicalImageDataset(
        data_dir=str(base_path / "test"),
        transform=transform
    )
    
    return train_dataset, val_dataset, test_dataset


def stage1_train(args):
    """Stage 1: Train classifier with UCB statistics (no pruning)"""
    print("\n" + "="*60)
    print("STAGE 1: Classification + UCB Statistics")
    print("  keep_ratio=1.0 (no pruning)")
    print("="*60 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    # Load datasets
    train_dataset, val_dataset, test_dataset = load_datasets(args.data_dir, transform)
    
    # DataLoaders
    train_sampler = create_balanced_sampler(train_dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Model - Stage 1: keep_ratio=1.0, input_aware_weight=0.0
    model = ViT_UCB_Pruning(
        model_name=args.model_name,
        pretrained=True,
        n_classes=len(train_dataset.class_names),
        keep_ratio=1.0,
        beta=args.beta,
        exclude_cls=True,
        input_aware_weight=0.0
    )
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.stage1_lr,
        weight_decay=args.weight_decay
    )
    
    num_steps = args.stage1_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_steps
    )
    
    # Training args
    dataset_name = args.dataset_name or Path(args.data_dir).name
    training_args = TrainingArguments(
        output_dir=args.output_dir / "stage1",
        run_name=f"{args.run_name}-stage1",
        dataset_name=dataset_name,
        num_train_epochs=args.stage1_epochs,
        learning_rate=args.stage1_lr,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        fp16=args.fp16,
        save_best_model=True,
        save_total_limit=2,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_metric="eval/f1",
        report_to=args.report_to,
        model_type="ucb",
        input_aware_weight=0.0
    )
    
    # Trainer
    trainer = ModelTrainer(
        model=model,
        args=training_args,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        test_dataloader=test_loader,
        class_names=train_dataset.class_names,
        optimizers=(optimizer, scheduler),
        device=device
    )
    
    # Train
    trainer.train()
    
    # Return best checkpoint
    best_ckpt = args.output_dir / "stage1" / "best_model" / f"{args.run_name}-stage1.bin"
    return best_ckpt, len(train_dataset.class_names)


def stage2_train(args, stage1_checkpoint, n_classes):
    """Stage 2: Fine-tune with pruning and input-aware"""
    print("\n" + "="*60)
    print("STAGE 2: Pruned Inference with Input-Aware")
    print(f"  keep_ratio={args.keep_ratio}")
    print(f"  input_aware_weight={args.input_aware_weight}")
    print("="*60 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    # Load datasets
    train_dataset, val_dataset, test_dataset = load_datasets(args.data_dir, transform)
    
    # DataLoaders
    train_sampler = create_balanced_sampler(train_dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Model - Stage 2: keep_ratio<1.0, input_aware_weight>0
    model = ViT_UCB_Pruning(
        model_name=args.model_name,
        pretrained=True,
        n_classes=n_classes,
        keep_ratio=args.keep_ratio,
        beta=args.beta,
        exclude_cls=True,
        input_aware_weight=args.input_aware_weight
    )
    
    # Load stage 1 checkpoint
    print(f"Loading checkpoint: {stage1_checkpoint}")
    state_dict = torch.load(stage1_checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print("Checkpoint loaded\n")
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.stage2_lr,
        weight_decay=args.weight_decay
    )
    
    num_steps = args.stage2_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps // 2,
        num_training_steps=num_steps
    )
    
    # Training args
    dataset_name = args.dataset_name or Path(args.data_dir).name
    training_args = TrainingArguments(
        output_dir=args.output_dir / f"stage2_kr{args.keep_ratio}",
        run_name=f"{args.run_name}-kr{args.keep_ratio}-iaw{args.input_aware_weight}",
        dataset_name=dataset_name,
        num_train_epochs=args.stage2_epochs,
        learning_rate=args.stage2_lr,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps // 2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        fp16=args.fp16,
        save_best_model=True,
        save_total_limit=2,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_metric="eval/f1",
        report_to=args.report_to,
        model_type="ucb",
        input_aware_weight=args.input_aware_weight
    )
    
    # Trainer
    trainer = ModelTrainer(
        model=model,
        args=training_args,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        test_dataloader=test_loader,
        class_names=train_dataset.class_names,
        optimizers=(optimizer, scheduler),
        device=device
    )
    
    # Train
    trainer.train()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)


def main():
    parser = argparse.ArgumentParser("Two-stage UCB ViT Training")
    
    # Data - now just base directory
    parser.add_argument("--data_dir", type=str, required=True, 
                       help="Base dir with train/val/test subdirs (e.g., data/BACH)")
    parser.add_argument("--dataset_name", type=str, default=None,
                       help="Dataset name for W&B tag (default: extracted from data_dir)")
    parser.add_argument("--img_size", type=int, default=224)
    
    # Model
    parser.add_argument("--model_name", type=str, default="hf-hub:MahmoodLab/uni")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--keep_ratio", type=float, default=0.7)
    parser.add_argument("--input_aware_weight", type=float, default=0.2)
    
    # Training - Stage 1
    parser.add_argument("--stage1_epochs", type=int, default=10)
    parser.add_argument("--stage1_lr", type=float, default=1e-3)
    
    # Training - Stage 2
    parser.add_argument("--stage2_epochs", type=int, default=5)
    parser.add_argument("--stage2_lr", type=float, default=3e-4)
    
    # Common training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    
    # Logging
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--run_name", type=str, default="ucb-vit")
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    
    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_stage1", action="store_true")
    parser.add_argument("--stage1_checkpoint", type=str)
    
    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Stage 1
    if not args.skip_stage1:
        stage1_ckpt, n_classes = stage1_train(args)
    else:
        assert args.stage1_checkpoint, "Must provide --stage1_checkpoint"
        stage1_ckpt = Path(args.stage1_checkpoint)
        # Infer n_classes
        temp_ds = HistologicalImageDataset(data_dir=f"{args.data_dir}/train")
        n_classes = len(temp_ds.class_names)
    
    # Stage 2
    stage2_train(args, stage1_ckpt, n_classes)


if __name__ == "__main__":
    main()