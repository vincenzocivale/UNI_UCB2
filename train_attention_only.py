"""
Attention-only pruning training (stage 2 only)
Uses UCB stage1 checkpoint, prunes via first block attention
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
from src.attention_pruning import ViT_AttentionPruning


def create_balanced_sampler(dataset):
    class_counts = np.bincount(dataset.labels)
    class_weights = 1.0 / class_counts
    sample_weights = torch.from_numpy(class_weights[dataset.labels]).double()
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def stage2_train(args, stage1_checkpoint, n_classes):
    print("\n" + "="*60)
    print(f"STAGE 2: Attention-Only Pruning (keep_ratio={args.keep_ratio})")
    print(f"Loading from: {stage1_checkpoint}")
    print("="*60 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    train_dataset = HistologicalImageDataset(
        data_dir=args.data_dir, transform=transform, split='train',
        test_size=args.test_size, val_size=args.val_size, seed=args.seed
    )
    val_dataset = HistologicalImageDataset(
        data_dir=args.data_dir, transform=transform, split='val',
        test_size=args.test_size, val_size=args.val_size, seed=args.seed
    )
    test_dataset = HistologicalImageDataset(
        data_dir=args.data_dir, transform=transform, split='test',
        test_size=args.test_size, val_size=args.val_size, seed=args.seed
    )
    
    train_sampler = create_balanced_sampler(train_dataset)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    model = ViT_AttentionPruning(
        model_name=args.model_name,
        pretrained=True,
        n_classes=n_classes,
        keep_ratio=args.keep_ratio,
        beta=1.0,
        exclude_cls=True,
        input_aware_weight=0.0
    )
    
    print(f"Loading checkpoint: {stage1_checkpoint}")
    state_dict = torch.load(stage1_checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_steps = args.epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_steps)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir / f"stage2_kr{args.keep_ratio}",
        run_name=f"{args.run_name}-attn-kr{int(args.keep_ratio*100)}",
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
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
        model_type="attention"
    )
    
    trainer = ModelTrainer(
        model=model, args=training_args,
        train_dataloader=train_loader, eval_dataloader=val_loader, test_dataloader=test_loader,
        class_names=train_dataset.class_names, optimizers=(optimizer, scheduler), device=device
    )
    
    trainer.train()


def main():
    parser = argparse.ArgumentParser("Attention-Only Pruning Training")
    
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--stage1_checkpoint", type=str, required=True, help="UCB stage1 checkpoint")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    
    parser.add_argument("--model_name", type=str, default="hf-hub:MahmoodLab/uni")
    parser.add_argument("--keep_ratio", type=float, default=0.7)
    
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=250)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    
    parser.add_argument("--output_dir", type=str, default="./results_attention")
    parser.add_argument("--run_name", type=str, default="attn")
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Infer n_classes
    temp_dataset = HistologicalImageDataset(data_dir=args.data_dir, split='train', test_size=args.test_size, val_size=args.val_size)
    n_classes = len(temp_dataset.class_names)
    
    stage2_train(args, Path(args.stage1_checkpoint), n_classes)


if __name__ == "__main__":
    main()
