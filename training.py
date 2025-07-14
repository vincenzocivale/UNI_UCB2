# %%

import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from src.utils.hf_utils import download_weights
from src.utils.vit_config import inizialize_model
from src.data.dataset import PatchFromH5Dataset
from src.rl.train import Trainer, TrainingArguments


# %%
NUM_STEPS = 100000 
LEARNING_RATE = 5e-5 
WEIGHT_DECAY = 0.01 
DECAY_TYPE = "cosine"
WARMUP_STEPS = 500
IMG_SIZE = 224
TRAIN_BATCH_SIZE = 8 
VAL_BATCH_SIZE = 8 
NUM_CLASSES = 6
EVAL_EVERY = 500 
GRADIENT_ACCUMULATION_STEPS = 4

# %%
HF_WEIGHTS_PATH = "/equilibrium/datasets/TCGA-histological-data/vit_weights_cache"
weights_path = download_weights(HF_WEIGHTS_PATH)

timm_pretrained_state_dict = torch.load(weights_path, map_location="cpu")

# %%
model = inizialize_model(timm_pretrained_state_dict, num_classes=NUM_CLASSES)

# %%
dataset = PatchFromH5Dataset(
    h5_dir='/equilibrium/datasets/TCGA-histological-data/hest_dataset/datasets--MahmoodLab--hest/snapshots/cf37675c2006e6dfcdaa084ddeca863d21a8ddbb/patches',
    transform=transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])
)

# %%
labels = [dataset.label_to_idx[dataset.sample_to_label[file.replace('.h5','')]]
          for (file, _) in dataset.data_index]

indices = list(range(len(dataset)))

# Split stratificato
train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# Crea i Subset
train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=16)

# %%
# Plot distribution of classes in train and val sets
import matplotlib.pyplot as plt
import numpy as np

def plot_class_distribution(labels, title):
    unique, counts = np.unique(labels, return_counts=True)
    plt.bar(unique, counts)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks(unique)
    plt.show()
plot_class_distribution([labels[i] for i in train_idx], "Train Set Class Distribution")
plot_class_distribution([labels[i] for i in val_idx], "Validation Set Class Distribution")

# %%
loss_function = torch.nn.CrossEntropyLoss()

# %%
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9,
    weight_decay=WEIGHT_DECAY,
)

# %%
if DECAY_TYPE == "cosine":
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=NUM_STEPS
    )
else: # DECAY_TYPE == "linear"
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=NUM_STEPS
    )

# %%
training_params = {
    "name": "vit_training_notebook_run",
    "output_dir": "./output_notebook",
    "eval_every": EVAL_EVERY, 
    "num_steps": NUM_STEPS,
    "learning_rate": LEARNING_RATE, # Passa l'LR se il trainer lo calcola internamente
    "weight_decay": WEIGHT_DECAY,
    "decay_type": DECAY_TYPE,
    "warmup_steps": WARMUP_STEPS,
    "max_grad_norm": 1.0,
    "local_rank": -1, # Usa -1 per non distribuito in un singolo notebook
    "seed": 42,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "fp16": False, # Abilita o disabilita AMP
    "img_size": IMG_SIZE, # Necessario per UCB_Count_Score
    "train_batch_size": TRAIN_BATCH_SIZE, # Necessario per UCB_Count_Score
     "num_classes": NUM_CLASSES,
}

args = TrainingArguments(**training_params)

trainer = Trainer(
    args=args,
    model=model,
    train_dataloader=train_loader,
    eval_dataloader=val_loader,
    loss_function=loss_function,
    optimizer=optimizer, 
    scheduler=scheduler 
)

# %%
trainer.train()


