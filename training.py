# %%
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import wandb

from src.data.dataset import PatchFromH5Dataset, stratified_split, plot_class_distributions
from src.rl.train import ModelTrainer, TrainingArguments
from src.rl.modelling import ViT_UCB_Pruning

# %%
IMG_SIZE = 224
TRAIN_BATCH_SIZE = 8
NUM_EPOCHS = 30

# Questo è il rapporto di pruning usato durante il TRAINING
TRAINING_KEEP_RATIO = 0.3 # Aumentato da 0.01 per un training più stabile

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# %%
dataset = PatchFromH5Dataset(
    h5_dir='/data/patches/',
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
    keep_ratio=TRAINING_KEEP_RATIO, # Usato durante il training UCB
    exclude_cls=True # Escludiamo sempre il CLS token dal pruning
)

# %%
args = TrainingArguments(
        output_dir="./results",
        run_name=f"ViT-UCB-Training-keep_ratio-{TRAINING_KEEP_RATIO}",
        num_train_epochs=NUM_EPOCHS,
        evaluation_strategy="epoch",
        learning_rate=0.01,
        train_batch_size=8,
        eval_batch_size=8,
        max_steps=-1,
        warmup_steps=500,
        eval_steps=5000,
        save_steps=10000,
        logging_steps=300,
        fp16=False, # Impostato a False, GradScaler non verrà usato
        report_to="wandb", 
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
# Avvia il training. Il benchmark di velocità verrà eseguito e loggato automaticamente alla fine.
trainer.train()

# %% [markdown]
# ## Benchmark di Velocità Post-Training
# 
# Dopo aver completato l'addestramento, la cella seguente eseguirà un benchmark per misurare la velocità di inferenza del modello potato. Questo test:
# 1. Usa il modello appena addestrato.
# 2. Calcola quali token tenere (`top_k_indices`) basandosi sui punteggi UCB appresi.
# 3. Misura il tempo medio di inferenza usando il metodo `forward_pruned`.
# 4. Logga questo risultato in un nuovo progetto su W&B chiamato `vit-ucb-pruning-final` per un'analisi chiara e separata.

# %%
# --- Benchmark di Velocità di Inferenza Post-Training ---
print("***** Inizio Benchmark di Velocità Post-Training *****")

# Parametri per il benchmark
INFERENCE_KEEP_RATIO = 0.5  # Rapporto di token da conservare per l'inferenza
WARMUP_RUNS = 10
BENCHMARK_RUNS = 50

trained_model = trainer.model
trained_model.eval()

# 1. Ottieni gli indici dei token più importanti dal modello addestrato
try:
    top_k_indices = trained_model.get_top_k_patch_indices(keep_ratio=INFERENCE_KEEP_RATIO)
    print(f"Indici per il pruning calcolati con successo con keep_ratio={INFERENCE_KEEP_RATIO}")
    num_total_tokens = trained_model.pos_embed.shape[1]
    num_pruned_tokens = len(top_k_indices)
    print(f"Token conservati per l'inferenza: {num_pruned_tokens}/{num_total_tokens}")
except Exception as e:
    print(f"Errore nel calcolo degli indici di pruning: {e}")
    top_k_indices = None

if top_k_indices is not None:
    # 2. Crea dati fittizi per il benchmark
    batch_size = args.eval_batch_size
    dummy_input = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    top_k_indices = top_k_indices.to(DEVICE)

    # 3. Esegui il benchmark (con warmup)
    print(f"Esecuzione di {WARMUP_RUNS} warmup runs...")
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = trained_model.forward_pruned(dummy_input, top_k_indices)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()

    print(f"Esecuzione di {BENCHMARK_RUNS} benchmark runs...")
    total_time = 0
    with torch.no_grad():
        for _ in range(BENCHMARK_RUNS):
            start_time = time.perf_counter()
            _ = trained_model.forward_pruned(dummy_input, top_k_indices)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            total_time += (end_time - start_time)
    
    avg_time_ms = (total_time / BENCHMARK_RUNS) * 1000


    print(f"Tempo medio di inferenza potata: {avg_time_ms:.3f} ms")
    print("---------------------------")
    
    # 4. Logga il risultato su un nuovo progetto/run W&B
    print("Logging della metrica di velocità su Weights & Biases...")
    try:
        wandb.init(
            project="vit-ucb-pruning-final", 
            name=f"{args.run_name}-inference-benchmark",
            config={
                "inference_keep_ratio": INFERENCE_KEEP_RATIO,
                "benchmark_runs": BENCHMARK_RUNS,
                "training_run_name": args.run_name
            }
        )
        wandb.log({"pruned_inference_avg_time_ms": avg_time_ms})
        wandb.finish()
        print("Metrica loggata con successo!")
    except Exception as e:
        print(f"Errore durante il logging su W&B: {e}")


