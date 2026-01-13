"""
Attention-only pruning pipeline
Reuses UCB stage1 checkpoint
"""
import subprocess
from pathlib import Path

DATA_DIR = "CRC-VAL-HE-7K"
OUTPUT_DIR = "./results_attention"
RUN_NAME = "attn"

# UCB stage1 checkpoint
UCB_STAGE1_CKPT = "./results_multi_ratio2/stage1/best_model/ucb-vit-nct-stage1.bin"

EPOCHS = 10
LR = 5e-4
BATCH_SIZE = 16
KEEP_RATIOS = [0.5, 0.4, 0.3, 0.2, 0.1]

def run(cmd):
    print(f"\n>>> Attention pruning: kr={cmd[-9]}")
    subprocess.run(cmd, check=True)

prev_ckpt = UCB_STAGE1_CKPT

for keep_ratio in KEEP_RATIOS:
    print(f"\n{'='*60}\nAttention Pruning: keep_ratio={keep_ratio}\n{'='*60}")
    
    run([
        "python", "train_attention_only.py",
        "--data_dir", DATA_DIR,
        "--output_dir", OUTPUT_DIR,
        "--run_name", RUN_NAME,
        "--stage1_checkpoint", str(prev_ckpt),
        "--epochs", str(EPOCHS),
        "--lr", str(LR),
        "--keep_ratio", str(keep_ratio),
        "--batch_size", str(BATCH_SIZE),
        "--num_workers", "8",
        "--test_size", "0.15",
        "--val_size", "0.15",
        "--warmup_steps", "250",
        "--weight_decay", "0.01",
        "--gradient_checkpointing",
        "--report_to", "wandb"
    ])
    
    prev_ckpt = Path(OUTPUT_DIR) / f"stage2_kr{keep_ratio}" / "best_model" / f"{RUN_NAME}-attn-kr{int(keep_ratio*100)}-stage1.bin"

print("\n=== COMPLETED ===")
