import argparse
import subprocess
import os
import logging

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command):
    """Runs a command and logs its output."""
    logger.info(f"🚀 Running command: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    for line in iter(process.stdout.readline, ''):
        logger.info(line.strip())
        
    process.stdout.close()
    return_code = process.wait()
    
    if return_code != 0:
        logger.error(f"❌ Command failed with exit code {return_code}")
        raise subprocess.CalledProcessError(return_code, command)
    logger.info("✅ Command completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Automated training pipeline for Stage 1 and Stage 2.")
    
    # --- Shared arguments ---
    parser.add_argument("--h5_dir", type=str, required=True, help="Directory containing H5 files.")
    parser.add_argument("--img_size", type=int, default=224, help="Image size for training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--organ", type=str, default=None, help="Organ to filter for training (e.g., lung, liver).")
    parser.add_argument("--report_to", type=str, default="wandb", help="Reporting tool (e.g., wandb, none).")
    parser.add_argument("--log_checkpoints_to_wandb", type=bool, default=False, help="Log checkpoints to W&B.")

    # --- Stage 1 arguments ---
    s1_group = parser.add_argument_group("Stage 1 Arguments")
    s1_group.add_argument("--s1_output_dir", type=str, default="./results_stage1", help="Output directory for Stage 1.")
    s1_group.add_argument("--s1_run_name", type=str, default="vit_stage1", help="Run name for Stage 1.")
    s1_group.add_argument("--s1_num_train_epochs", type=int, default=30, help="Number of training epochs for Stage 1.")
    s1_group.add_argument("--s1_learning_rate", type=float, default=1e-3, help="Learning rate for Stage 1.")
    s1_group.add_argument("--s1_train_batch_size", type=int, default=8, help="Training batch size for Stage 1.")
    s1_group.add_argument("--s1_eval_batch_size", type=int, default=12, help="Evaluation batch size for Stage 1.")
    s1_group.add_argument("--s1_warmup_steps", type=int, default=500, help="Warmup steps for Stage 1.")
    s1_group.add_argument("--s1_logging_steps", type=int, default=1000, help="Logging steps for Stage 1.")
    s1_group.add_argument("--s1_beta", type=float, default=1.0, help="Beta value for UCB in Stage 1.")
    s1_group.add_argument("--s1_input_aware_weight", type=float, default=0.7, help="Input aware weight for Stage 1.")

    # --- Stage 2 arguments ---
    s2_group = parser.add_argument_group("Stage 2 Arguments")
    s2_group.add_argument("--s2_output_dir", type=str, default="./results_stage2", help="Output directory for Stage 2.")
    s2_group.add_argument("--s2_run_name", type=str, default="vit_stage2", help="Run name for Stage 2.")
    s2_group.add_argument("--s2_num_train_epochs", type=int, default=30, help="Number of training epochs for Stage 2.")
    s2_group.add_argument("--s2_learning_rate", type=float, default=1e-3, help="Learning rate for Stage 2.")
    s2_group.add_argument("--s2_train_batch_size", type=int, default=8, help="Training batch size for Stage 2.")
    s2_group.add_argument("--s2_eval_batch_size", type=int, default=8, help="Evaluation batch size for Stage 2.")
    s2_group.add_argument("--s2_warmup_steps", type=int, default=100, help="Warmup steps for Stage 2.")
    s2_group.add_argument("--s2_logging_steps", type=int, default=100, help="Logging steps for Stage 2.")
    s2_group.add_argument("--s2_keep_ratio", type=float, default=0.5, help="Keep ratio for Stage 2 pruning.")
    s2_group.add_argument("--s2_beta", type=float, default=1.0, help="Beta value for UCB in Stage 2.")
    s2_group.add_argument("--s2_input_aware_weight", type=float, default=0.7, help="Input aware weight for Stage 2.")
    
    args = parser.parse_args()

    # Aggiorna i nomi delle run per includere i parametri
    args.s1_run_name = f"{args.s1_run_name}-iaw_{args.s1_input_aware_weight}"
    if args.organ:
        args.s1_run_name += f"-{args.organ}"

    args.s2_run_name = f"{args.s2_run_name}-kr_{args.s2_keep_ratio}-iaw_{args.s2_input_aware_weight}"
    if args.organ:
        args.s2_run_name += f"-{args.organ}"

    # === STAGE 1: Train without pruning ===
    logger.info("="*80)
    logger.info("                          STARTING STAGE 1                          ")
    logger.info("="*80)
    
    stage1_cmd = [
        "python", "train_stage1.py",
        "--h5_dir", args.h5_dir,
        "--img_size", str(args.img_size),
        "--output_dir", args.s1_output_dir,
        "--run_name", args.s1_run_name,
        "--num_train_epochs", str(args.s1_num_train_epochs),
        "--learning_rate", str(args.s1_learning_rate),
        "--train_batch_size", str(args.s1_train_batch_size),
        "--eval_batch_size", str(args.s1_eval_batch_size),
        "--warmup_steps", str(args.s1_warmup_steps),
        "--logging_steps", str(args.s1_logging_steps),
        "--beta", str(args.s1_beta),
        "--input_aware_weight", str(args.s1_input_aware_weight),
        "--seed", str(args.seed),
        "--report_to", args.report_to,
        "--log_checkpoints_to_wandb", str(args.log_checkpoints_to_wandb),
    ]
    if args.organ:
        stage1_cmd.extend(["--organ", args.organ])
        
    try:
        run_command(stage1_cmd)
    except subprocess.CalledProcessError:
        logger.error("Stage 1 failed. Aborting pipeline.")
        return

    # === STAGE 2: Train with pruning from Stage 1 weights ===
    logger.info("="*80)
    logger.info("                          STARTING STAGE 2                          ")
    logger.info("="*80)
    
    stage1_checkpoint_path = os.path.join(args.s1_output_dir, "best_model", f"{args.s1_run_name}.bin")
    
    if not os.path.exists(stage1_checkpoint_path):
        logger.error(f"Stage 1 checkpoint not found at: {stage1_checkpoint_path}")
        logger.error("Cannot proceed to Stage 2 without Stage 1 checkpoint.")
        return
        
    logger.info(f"Found Stage 1 checkpoint: {stage1_checkpoint_path}")

    stage2_cmd = [
        "python", "train_stage2.py",
        "--h5_dir", args.h5_dir,
        "--img_size", str(args.img_size),
        "--stage1_checkpoint", stage1_checkpoint_path,
        "--output_dir", args.s2_output_dir,
        "--run_name", args.s2_run_name,
        "--num_train_epochs", str(args.s2_num_train_epochs),
        "--learning_rate", str(args.s2_learning_rate),
        "--train_batch_size", str(args.s2_train_batch_size),
        "--eval_batch_size", str(args.s2_eval_batch_size),
        "--warmup_steps", str(args.s2_warmup_steps),
        "--logging_steps", str(args.s2_logging_steps),
        "--keep_ratio", str(args.s2_keep_ratio),
        "--beta", str(args.s2_beta),
        "--input_aware_weight", str(args.s2_input_aware_weight),
        "--seed", str(args.seed),
        "--report_to", args.report_to,
        "--log_checkpoints_to_wandb", str(args.log_checkpoints_to_wandb),
    ]
    if args.organ:
        stage2_cmd.extend(["--organ", args.organ])

    try:
        run_command(stage2_cmd)
    except subprocess.CalledProcessError:
        logger.error("Stage 2 failed.")
        return
        
    logger.info("🎉 Training pipeline completed successfully! 🎉")

if __name__ == "__main__":
    main()
