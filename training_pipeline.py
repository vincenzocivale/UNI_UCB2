# %%

import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

# from src.utils.hf_utils import download_weights
# from src.utils.vit_config import inizialize_model
from src.data.dataset import PatchFromH5Dataset, stratified_split, plot_class_distributions
from src.rl.train import ModelTrainer, TrainingArguments

from src.rl.modelling import ViT_UCB_Pruning


# %%
IMG_SIZE = 224
TRAIN_BATCH_SIZE = 8
NUM_EPOCHS = 70

# %%
dataset = PatchFromH5Dataset(
    h5_dir='/equilibrium/datasets/TCGA-histological-data/hest/patches/patches/',
    transform=transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])
)

# %%
train_idx, val_idx, test_idx = stratified_split(dataset, test_size=0.3, val_size=0.3, random_state=42)

# %%
plot_class_distributions(train_idx, val_idx, test_idx, full_dataset=dataset)

# %%
train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)
test_dataset = Subset(dataset, test_idx)

# %%
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=16, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=16, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=16, drop_last=True)

# %%
labels_num = len(dataset.labels)

print(f"Number of classes: {labels_num}")
model = ViT_UCB_Pruning(model_name="hf-hub:MahmoodLab/uni", pretrained=True, n_classes=labels_num)

# %%
args = TrainingArguments(
        output_dir="./results",
        run_name="ViT-L-UCB-Pruning-run1",
        num_train_epochs=NUM_EPOCHS,
        learning_rate=0.1,
        train_batch_size=8,
        eval_batch_size=8,
        max_steps=20000,
        warmup_steps=500,
        eval_steps=92406,
        save_steps=92406,
        logging_steps=2500,
        fp16=False,
        report_to="wandb", 
        early_stopping_patience=3, # Interrompi dopo 3 valutazioni senza miglioramento
        early_stopping_metric="eval/loss", # Oppure monitora la loss (un valore più basso è meglio)
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
        optimizers=(optimizer, scheduler) 
    )

# %%
trainer.train()


