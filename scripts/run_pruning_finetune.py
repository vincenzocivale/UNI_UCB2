
import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset
from transformers import TrainingArguments, get_cosine_schedule_with_warmup
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

from src.data.h5_dataset import H5PatchDataset
from src.models.pruning_model import VisionTransformerUCB
from src.trainer.ucb_trainer import UcbTrainer

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Dataset Preparation (same as Phase 1) ---
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = H5PatchDataset(h5_dir=args.data_dir, transform=transform)

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

    # --- Model Preparation ---
    num_classes = len(np.unique(dataset.labels))
    model = VisionTransformerUCB(
        model_name=args.model_name,
        pretrained=False, # We load our own weights
        n_classes=num_classes,
        keep_ratio=args.frozen_keep_ratio,
        exclude_cls=True
    )

    if not os.path.exists(args.stage1_checkpoint_path):
        raise FileNotFoundError(f"Phase 1 checkpoint not found at: {args.stage1_checkpoint_path}")
    
    print(f"Loading Phase 1 weights from: {args.stage1_checkpoint_path}")
    # The checkpoint from HF Trainer is a directory, we need to load the pytorch_model.bin file.
    model_path = os.path.join(args.stage1_checkpoint_path, "pytorch_model.bin")
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # --- Calculate Top-K Indices ---
    print(f"Calculating Top-K indices for keep_ratio={args.frozen_keep_ratio}")
    with torch.no_grad():
        top_k_indices = model.get_top_k_patch_indices(keep_ratio=args.frozen_keep_ratio)
    print(f"Indices calculated. {len(top_k_indices)} patches will be kept (including CLS token).")


    # --- Trainer ---
    output_dir = os.path.join(args.output_dir, f"ratio_{args.frozen_keep_ratio}")
    run_name = f"{args.run_name}_ratio_{args.frozen_keep_ratio}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        warmup_steps=200,
        report_to="wandb" if args.use_wandb else "none",
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    
    num_training_steps = args.num_epochs * (len(train_dataset) // args.batch_size)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=200, 
        num_training_steps=num_training_steps
    )

    trainer = UcbTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, scheduler),
        top_k_indices=top_k_indices, # Pass the frozen indices
    )

    trainer.train()

    print("--- Phase 2 Fine-tuning Finished ---")
    print(f"Best model checkpoint saved to {trainer.state.best_model_checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Fine-tuning with Fixed Pruning")
    parser.add_argument("--stage1_checkpoint_path", type=str, required=True, help="Path to the Phase 1 checkpoint directory from Hugging Face Trainer.")
    parser.add_argument("--frozen_keep_ratio", type=float, required=True, help="The final keep_ratio for fixed pruning.")
    
    parser.add_argument("--data_dir", type=str, default="/data/patches/", help="Directory with H5 patch files.")
    parser.add_argument("--output_dir", type=str, default="./results/ucb_phase2", help="Directory to save checkpoints and results.")
    parser.add_argument("--run_name", type=str, default="vit-ucb-phase2-finetune", help="Name for the WandB run.")
    parser.add_argument("--img_size", type=int, default=224, help="Image size.")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for training and evaluation.")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of fine-tuning epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for fine-tuning.")
    parser.add_argument("--model_name", type=str, default="hf-hub:MahmoodLab/uni", help="Name of the pretrained model from timm.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--use_wandb", action='store_true', help="Flag to enable WandB logging.")

    args = parser.parse_args()
    main(args)
