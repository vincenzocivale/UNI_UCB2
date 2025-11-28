
import logging
import os
import sys
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from src.data.dataset import PatchFromH5Dataset
from src.rl.modelling import ViT_UCB_Pruning
from src.rl.hf_trainer import CustomTrainer, PruningMetricsCallback, compute_metrics

# --- Setup di Logging ---
logger = logging.getLogger(__name__)

# --- Definizione degli Argomenti ---

@dataclass
class ModelArguments:
    """
    Argomenti relativi al modello.
    """
    model_name: str = field(
        default="hf-hub:MahmoodLab/uni",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    keep_ratio: float = field(
        default=0.5,
        metadata={"help": "The ratio of patches to keep during UCB training and deterministic evaluation."}
    )
    beta: float = field(
        default=1.0,
        metadata={"help": "Beta parameter for the UCB exploration term."}
    )
    model_type: str = field(
        default="ucb",
        metadata={"help": "Type of model to train: 'ucb', 'random', or 'baseline'."}
    )


@dataclass
class DataArguments:
    """
    Argomenti relativi ai dati.
    """
    data_dir: str = field(
        default="/data/patches/",
        metadata={"help": "The directory where the HDF5 patch data is stored."}
    )
    undersample: bool = field(
        default=True,
        metadata={"help": "Whether to undersample the dataset to balance classes."}
    )
    test_split_size: float = field(
        default=0.2,
        metadata={"help": "The proportion of the dataset to include in the test split."}
    )
    val_split_size: float = field(
        default=0.2,
        metadata={"help": "The proportion of the train+val dataset to include in the validation split."}
    )
    img_size: int = field(
        default=224,
        metadata={"help": "The size to resize images to."}
    )


def main():
    # --- Parsing degli Argomenti ---
    parser = HfArgumentParser((TrainingArguments, ModelArguments, DataArguments))
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()

    # --- Setup del Logging ---
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARNING)

    # Log a separator
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # --- Set Seed ---
    set_seed(training_args.seed)

    # --- Caricamento Dati ---
    logger.info("Loading dataset...")
    full_dataset = PatchFromH5Dataset(
        h5_dir=data_args.data_dir,
        transform=transforms.Compose([
            transforms.Resize((data_args.img_size, data_args.img_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    )
    
    indices = list(range(len(full_dataset)))
    labels = full_dataset.labels

    if data_args.undersample:
        logger.info("Performing undersampling to balance classes...")
        df = pd.DataFrame({"index": indices, "label": labels})
        min_count = df["label"].value_counts().min()
        undersampled_df = (
            df.groupby("label", group_keys=False)
              .apply(lambda x: x.sample(n=min_count, random_state=training_args.seed))
        )
        indices = undersampled_df["index"].tolist()
        labels = undersampled_df["label"].tolist()

    # Stratified Split
    logger.info("Performing stratified train/val/test split...")
    trainval_idx, test_idx = train_test_split(
        indices,
        test_size=data_args.test_split_size,
        stratify=labels,
        random_state=training_args.seed
    )
    
    trainval_labels = [full_dataset.labels[i] for i in trainval_idx]
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=data_args.val_split_size,
        stratify=trainval_labels,
        random_state=training_args.seed
    )

    train_dataset = Subset(full_dataset, train_idx)
    eval_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(eval_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    num_labels = len(np.unique(full_dataset.labels))

    # --- Caricamento Modello ---
    logger.info(f"Loading model: {model_args.model_name}")
    model = ViT_UCB_Pruning(
        model_name=model_args.model_name,
        pretrained=True,
        n_classes=num_labels,
        keep_ratio=model_args.keep_ratio,
        beta=model_args.beta,
    )

    # --- Inizializzazione Trainer ---
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        callbacks=[PruningMetricsCallback()],
    )

    # --- Esecuzione Training ---
    if training_args.do_train:
        logger.info("*** Starting Training ***")
        train_result = trainer.train()
        trainer.save_model()  # Salva il modello finale
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # --- Esecuzione Valutazione ---
    if training_args.do_eval:
        logger.info("*** Starting Evaluation ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # --- Esecuzione Predizione sul Test Set ---
    if training_args.do_predict:
        logger.info("*** Starting Prediction on Test Set ***")
        predictions = trainer.predict(test_dataset, metric_key_prefix="predict")
        trainer.log_metrics("predict", predictions.metrics)
        trainer.save_metrics("predict", predictions.metrics)

if __name__ == "__main__":
    main()
