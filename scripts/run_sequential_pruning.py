import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset
from transformers import TrainingArguments, get_cosine_schedule_with_warmup, EarlyStoppingCallback
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import wandb

from src.data.h5_dataset import H5PatchDataset
from src.models.pruning_model import VisionTransformerUCB
from src.trainer.ucb_trainer import UcbTrainer, compute_metrics
from src.utils.performance import calculate_performance_metrics

def main(args):
    """
    Main orchestration script for the sequential pruning pipeline.
    Combines the dataset/trainer setup from run_pruning_finetune.py with sequential logic.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Dataset Preparation ---
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
    num_classes = len(np.unique(dataset.labels))

    # --- Sequential Training Loop ---
    previous_stage_checkpoint = None

    for i, keep_ratio in enumerate(args.pruning_stages):
        stage_name = f"pruning_{int(keep_ratio*100)}"
        output_dir = os.path.join(args.output_dir, stage_name)
        
        print("\n" + "="*50)
        print(f"STARTING STAGE {i+1}/{len(args.pruning_stages)}: keep_ratio = {keep_ratio}")
        print("="*50)

        # --- Initialize W&B ---
        if args.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=f"{args.run_name}_{stage_name}",
                reinit=True,
                config={{**vars(args), "keep_ratio": keep_ratio, "stage": i+1}}
            )

        # --- Model Setup ---
        print("Initializing model...")
        model = VisionTransformerUCB(
            model_name=args.model_name,
            pretrained=True if i == 0 else False,
            n_classes=num_classes,
            keep_ratio=keep_ratio,
            selection_mode='ucb'
        )

        if previous_stage_checkpoint:
            print(f"Loading weights from previous stage: {previous_stage_checkpoint}")
            model.load_state_dict(torch.load(previous_stage_checkpoint), strict=False)
        
        # --- Trainer Setup ---
        training_args = TrainingArguments(
            output_dir=output_dir,
            run_name=f"{args.run_name}_{stage_name}",
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            eval_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            learning_rate=args.learning_rate,
            warmup_steps=200,
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
            num_warmup_steps=200, 
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
            pruning_enabled=True,
            top_k_indices=None,
            ucb_update_enabled=(i == 0) # CRITICAL: Only update UCB scores in the first stage
        )

        print("Starting training for this stage...")
        trainer.train()

        # --- Evaluation & Performance Logging ---
        print("Evaluating on the test set...")
        test_results = trainer.predict(test_dataset)
        trainer.log_metrics("test", test_results.metrics)
        trainer.save_metrics("test", test_results.metrics)
        
        print("Calculating and logging performance metrics...")
        # For performance calculation, we use the best model from this stage
        best_model_path = trainer.state.best_model_checkpoint
        # The trainer saves the full state, we need to load the model from it
        perf_model = VisionTransformerUCB(
            model_name=args.model_name, pretrained=False, n_classes=num_classes, keep_ratio=keep_ratio
        )
        model_weights_path = os.path.join(best_model_path, "pytorch_model.bin")
        perf_model.load_state_dict(torch.load(model_weights_path))

        performance_metrics = calculate_performance_metrics(
            perf_model, 
            input_size=(3, args.img_size, args.img_size),
            device=device,
            keep_ratio=keep_ratio
        )
        if args.use_wandb:
            wandb.log({
                "pruning_gflops": performance_metrics['gflops'],
                "pruning_inference_ms": performance_metrics['inference_ms']
            })

        # --- Set Checkpoint for Next Stage ---
        previous_stage_checkpoint = model_weights_path
        
        if args.use_wandb:
            wandb.finish()

    print("\nSequential pruning pipeline finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential Pruning with Token Merging")
    
    # --- Arguments from run_pruning_finetune.py ---
    parser.add_argument("--data_dir", type=str, default="/data/patches/", help="Directory with H5 patch files.")
    parser.add_argument("--output_dir", type=str, default="./results/sequential_pruning_merge", help="Base directory to save checkpoints and results.")
    parser.add_argument("--run_name", type=str, default="vit-pruning-merging", help="Base name for the WandB run.")
    parser.add_argument("--img_size", type=int, default=224, help="Image size.")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for training and evaluation.")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs for each stage.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--model_name", type=str, default="hf-hub:MahmoodLab/uni", help="Name of the pretrained model from timm.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--use_wandb", action='store_true', help="Flag to enable WandB logging.")
    parser.add_argument("--wandb_project", type=str, default="vit_pruning", help="WandB project name.")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Patience for early stopping.")

    # --- New arguments for sequential pruning ---
    parser.add_argument("--pruning_stages", type=float, nargs='+', default=[1.0, 0.7, 0.5, 0.3, 0.1], help="A list of keep_ratio values for each pruning stage.")

    args = parser.parse_args()
    
    if args.use_wandb:
        print("WandB logging is enabled.")
    
    main(args)