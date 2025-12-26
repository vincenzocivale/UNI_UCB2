
import os
import torch
import wandb
from transformers import TrainingArguments, ViTImageProcessor

from src.models.pruning_model import VisionTransformerUCB
from src.trainer.ucb_trainer import UcbTrainer, compute_metrics
from src.utils.performance import calculate_performance_metrics
from src.data.h5_dataset import H5Dataset

def run_sequential_pruning():
    """
    Main orchestration script for the sequential pruning pipeline.
    """
    # =================================================================================
    # --- Configuration ---
    # Please adjust these settings for your environment.
    # =================================================================================
    
    # --- Pipeline Stages ---
    # Define the sequence of `keep_ratio` values for pruning.
    PRUNING_STAGES = [1.0, 0.7, 0.5, 0.3, 0.1]
    
    # --- Model & Checkpoints ---
    # The timm model name to use for the base architecture.
    # E.g., "vit_base_patch16_224" or your specific model like "hf-hub:MahmoodLab/uni"
    BASE_MODEL_NAME = "hf-hub:MahmoodLab/uni"
    # Directory to save and load checkpoints from.
    CHECKPOINT_ROOT_DIR = "checkpoints/sequential_pruning"
    
    # --- Dataset ---
    # !! IMPORTANT !!: Update these paths to your HDF5 dataset files.
    TRAIN_H5_PATH = "/home/oem/vcivale/UNI_UCB2/results/dynamic_pruning/data/train_patches.h5"
    EVAL_H5_PATH = "/home/oem/vcivale/UNI_UCB2/results/dynamic_pruning/data/val_patches.h5"
    
    # --- Training ---
    # Adjust training hyperparameters as needed.
    NUM_TRAIN_EPOCHS = 30
    PER_DEVICE_TRAIN_BATCH_SIZE = 16
    PER_DEVICE_EVAL_BATCH_SIZE = 16
    
    # --- W&B Logging ---
    WANDB_PROJECT_NAME = "vit_pruning"

    # =================================================================================

    print("Starting sequential pruning pipeline...")
    
    # --- Datasets ---
    print("Loading datasets...")
    try:
        train_dataset = H5Dataset(h5_path=TRAIN_H5_PATH)
        eval_dataset = H5Dataset(h5_path=EVAL_H5_PATH)
        # Infer number of classes from the dataset
        n_classes = len(train_dataset.dataset['labels'].unique())
        print(f"Loaded {len(train_dataset)} training samples and {len(eval_dataset)} evaluation samples.")
        print(f"Inferred {n_classes} classes from the dataset.")
    except Exception as e:
        print(f"Failed to load datasets. Please check the paths in the script configuration.")
        print(f"Error: {e}")
        return

    # --- Sequential Training Loop ---
    previous_stage_checkpoint = None

    for i, keep_ratio in enumerate(PRUNING_STAGES):
        stage_name = f"pruning_{int(keep_ratio*100)}"
        stage_checkpoint_dir = os.path.join(CHECKPOINT_ROOT_DIR, stage_name)
        os.makedirs(stage_checkpoint_dir, exist_ok=True)
        
        print("\n" + "="*50)
        print(f"STARTING STAGE {i+1}/{len(PRUNING_STAGES)}: keep_ratio = {keep_ratio}")
        print("="*50)

        # --- Initialize W&B ---
        run = wandb.init(
            project=WANDB_PROJECT_NAME,
            name=f"{stage_name}",
            reinit=True, # Allows re-initializing in the same script
            config={
                "keep_ratio": keep_ratio,
                "stage": i+1,
                "base_model": BASE_MODEL_NAME,
                "epochs": NUM_TRAIN_EPOCHS,
            }
        )

        # --- Model Setup ---
        print("Initializing model...")
        model = VisionTransformerUCB(
            model_name=BASE_MODEL_NAME,
            # Load official pretrained weights only for the first stage.
            # For subsequent stages, we load our own fine-tuned checkpoints.
            pretrained=True if i == 0 else False,
            n_classes=n_classes,
            keep_ratio=keep_ratio,
            selection_mode='ucb'
        )

        if previous_stage_checkpoint:
            print(f"Loading weights from previous stage: {previous_stage_checkpoint}")
            try:
                model.load_state_dict(torch.load(previous_stage_checkpoint), strict=False)
            except Exception as e:
                print(f"Error loading state dict: {e}")
                print("This can sometimes happen if the model architecture changed (e.g., number of patches).")
                print("Continuing with the randomly initialized final layer.")

        # --- Training ---
        training_args = TrainingArguments(
            output_dir=stage_checkpoint_dir,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            report_to="wandb",
            logging_dir=f"{stage_checkpoint_dir}/logs",
            remove_unused_columns=False, # Important for custom models
        )

        trainer = UcbTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            pruning_enabled=True, # We use the dynamic merging logic in all stages
            top_k_indices=None,
            ucb_update_enabled=(i == 0) # Only update UCB scores in the first stage
        )

        print("Starting training for this stage...")
        trainer.train()

        # --- Evaluation & Performance Logging ---
        print("Evaluating on the test set...")
        eval_metrics = trainer.evaluate()
        wandb.log(eval_metrics)

        print("Calculating and logging performance metrics (GFLOPS, Inference Time)...")
        # For performance calculation, we load the best weights from this stage
        best_model_path = os.path.join(stage_checkpoint_dir, "pytorch_model.bin")
        perf_model = VisionTransformerUCB(
            model_name=BASE_MODEL_NAME, pretrained=False, n_classes=n_classes, keep_ratio=keep_ratio
        )
        perf_model.load_state_dict(torch.load(best_model_path))

        performance_metrics = calculate_performance_metrics(
            perf_model, 
            device='cuda' if torch.cuda.is_available() else 'cpu',
            keep_ratio=keep_ratio
        )
        wandb.log(performance_metrics)

        # --- Save Final Model for Next Stage ---
        # We save the model state from the *trainer* which holds the best model weights
        final_checkpoint_path = os.path.join(stage_checkpoint_dir, "final_model.pt")
        print(f"Saving final model for this stage to: {final_checkpoint_path}")
        trainer.save_model(stage_checkpoint_dir) # Saves to pytorch_model.bin
        # For our sequential loading, we'll point to the standard file saved by trainer
        previous_stage_checkpoint = os.path.join(stage_checkpoint_dir, "pytorch_model.bin")
        
        run.finish()

    print("\nSequential pruning pipeline finished!")

if __name__ == "__main__":
    run_sequential_pruning()
