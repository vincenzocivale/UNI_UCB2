# %%

import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.dataset import PatchFromH5Dataset, stratified_split, plot_class_distributions
from src.rl.train import ModelTrainer, TrainingArguments
from src.rl.modelling import ViT_UCB_Pruning


# %%
IMG_SIZE = 224
TRAIN_BATCH_SIZE = 8
NUM_EPOCHS = 30

PRUNING_RATIO = 0.1

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# %%
dataset = PatchFromH5Dataset(
    h5_dir='/equilibrium/datasets/TCGA-histological-data/hest/patches/patches/',
    transform=transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])
)

# %%
labels = dataset.labels

# %%
# Crea un DataFrame con indici e label
df = pd.DataFrame({
    "index": np.arange(len(labels)),
    "label": labels
})

# Trova il numero di elementi della classe minoritaria
min_count = df["label"].value_counts().min()

# Per ogni classe, seleziona min_count elementi a caso
undersampled_df = (
    df.groupby("label", group_keys=False)
      .apply(lambda x: x.sample(n=min_count, random_state=42)).reset_index(drop=True)
)

# Mischia gli indici
undersampled_indices = undersampled_df["index"].sample(frac=1, random_state=42).tolist()


# %%
undersampled_labels = [labels[i] for i in undersampled_indices]

trainval_idx, test_idx = train_test_split(
    undersampled_indices,
    test_size=0.3,
    stratify=undersampled_labels,
    random_state=42
)

# Ottieni i label corrispondenti per il secondo split
trainval_labels = [labels[i] for i in trainval_idx]

# Split: train vs val
train_idx, val_idx = train_test_split(
    trainval_idx,
    test_size=0.3,
    stratify=trainval_labels,
    random_state=42
)

# Crea i subset
train_dataset = Subset(dataset, train_idx)
val_dataset   = Subset(dataset, val_idx)
test_dataset  = Subset(dataset, test_idx)

# %%
plot_class_distributions(train_dataset, val_dataset, test_dataset, full_dataset=dataset)

# %%
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=16, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=16, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=16, drop_last=True)

# %%
labels_num = len(np.unique(dataset.labels))

print(f"Number of classes: {labels_num}")
model = ViT_UCB_Pruning(model_name="hf-hub:MahmoodLab/uni", 
    pretrained=True, 
    n_classes=labels_num, 
    keep_ratio=PRUNING_RATIO,        
    exclude_cls=False
)

# %%
args = TrainingArguments(
        output_dir="./results",
        run_name=f"ViT-L-UCB-{PRUNING_RATIO}",
        num_train_epochs=NUM_EPOCHS,
        evaluation_strategy="epoch",
        learning_rate=0.1,
        train_batch_size=8,
        eval_batch_size=8,
        max_steps=-1,
        warmup_steps=500,
        eval_steps=5000,
        save_steps=10000,
        logging_steps=300,
        fp16=False,
        report_to="wandb", 
        early_stopping_patience=7, 
        early_stopping_metric="eval/loss", 
    )


# %%
optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
# The scheduler needs max_steps, so we calculate it first
num_steps = args.num_train_epochs * (len(train_loader) // args.gradient_accumulation_steps)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_steps)

# %%
trainer = ModelTrainer(
        model=model,
        args=args,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        test_dataloader=test_loader,
        class_names=dataset.class_names,           # Pass the class names
        optimizers=(optimizer, scheduler),
        device= DEVICE
    )

# %%
trainer.train()


