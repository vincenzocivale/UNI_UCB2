"""
Train/val/test split with checkpoint chaining
"""
import subprocess
import sys
from pathlib import Path

# Config
DATA_DIR = "CRC-VAL-HE-7K"
OUTPUT_DIR = "./results_multi_ratio2"
RUN_NAME = "ucb-vit-nct"
BATCH_SIZE = 16
NUM_WORKERS = 8

# Stage 1
STAGE1_EPOCHS = 20
STAGE1_LR = 1e-3

# Stage 2
STAGE2_EPOCHS = 10
STAGE2_LR = 5e-4
KEEP_RATIOS = [0.4, 0.3, 0.2, 0.1]
INPUT_AWARE_WEIGHT = 0.5

def run(cmd):
    print(f"\n>>> {' '.join(cmd[:5])}")
    subprocess.run(cmd, check=True)

# # Stage 1
# run([
#     "python", "train_two_stage.py",
#     "--data_dir", DATA_DIR,
#     "--output_dir", OUTPUT_DIR,
#     "--run_name", RUN_NAME,
#     "--stage1_epochs", str(STAGE1_EPOCHS),
#     "--stage1_lr", str(STAGE1_LR),
#     "--stage2_epochs", str(STAGE2_EPOCHS),
#     "--batch_size", str(BATCH_SIZE),
#     "--num_workers", str(NUM_WORKERS),
#     "--test_size", "0.15",  # 15% test, 15% val
#     "--val_size", "0.15",
#     "--warmup_steps", "500",
#     "--weight_decay", "0.01",
#     "--gradient_checkpointing",
#     "--report_to", "wandb"
# ])

stage1_ckpt = Path(OUTPUT_DIR) / "stage1" / "best_model" / f"{RUN_NAME}-stage1.bin"

# Stage 2 - Chained checkpoints
prev_ckpt = stage1_ckpt

for keep_ratio in KEEP_RATIOS:
    print(f"\n{'='*60}\nStage 2: keep_ratio={keep_ratio}\n{'='*60}")
    
    run([
        "python", "train_two_stage.py",
        "--data_dir", DATA_DIR,
        "--output_dir", OUTPUT_DIR,
        "--run_name", f"{RUN_NAME}-kr{keep_ratio}",
        "--skip_stage1",
        "--stage1_checkpoint", str(prev_ckpt),
        "--stage2_epochs", str(STAGE2_EPOCHS),
        "--stage2_lr", str(STAGE2_LR),
        "--keep_ratio", str(keep_ratio),
        "--input_aware_weight", str(INPUT_AWARE_WEIGHT),
        "--batch_size", str(BATCH_SIZE),
        "--num_workers", str(NUM_WORKERS),
        "--test_size", "0.15",
        "--val_size", "0.15",
        "--warmup_steps", "250",
        "--gradient_checkpointing",
        "--report_to", "wandb"
    ])
    
    # Update checkpoint for next iteration
    # Update checkpoint for next iteration
prev_ckpt = Path(OUTPUT_DIR) / f"stage2_kr{keep_ratio}" / "best_model" / f"{RUN_NAME}-kr{keep_ratio}-iaw{INPUT_AWARE_WEIGHT}-stage2.bin"
print("\n=== COMPLETED ===")