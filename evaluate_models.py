# evaluate_models.py
import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import time
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Importa le classi necessarie dai tuoi moduli src
# Assicurati che il PYTHONPATH sia impostato correttamente o esegui dalla root del progetto
from src.data.dataset import PatchFromH5Dataset
from src.rl.modelling import ViT_UCB_Pruning

# --- CONFIGURAZIONE ---
IMG_SIZE = 224
BATCH_SIZE = 8 # Puoi aumentarlo se hai abbastanza VRAM
MODELS_BASE_DIR = './results'
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

print(f"Using device: {DEVICE}")

# --- 1. CARICAMENTO DATASET (stessa logica di training.py) ---
print("Loading dataset...")
dataset = PatchFromH5Dataset(
    h5_dir='/data/patches/',
    transform=transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
)

labels = dataset.labels
df = pd.DataFrame({"index": np.arange(len(labels)), "label": labels})
min_count = df["label"].value_counts().min()
undersampled_df = (
    df.groupby("label", group_keys=False)
      .apply(lambda x: x.sample(n=min_count, random_state=42)).reset_index(drop=True)
)
undersampled_indices = undersampled_df["index"].sample(frac=1, random_state=42).tolist()

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


test_dataset  = Subset(dataset, test_idx)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, drop_last=False)
print(f"Test set loaded with {len(test_dataset)} samples.")

# --- 2. RICERCA DEI MODELLI ---
print("Finding trained models...")
# Cerca i file .bin direttamente nelle sottocartelle di primo livello (es. results/50, results/70, etc.)
model_paths = glob.glob(os.path.join(MODELS_BASE_DIR, '*', '*.bin'))

# Filtra via i modelli che sono dentro le cartelle dei checkpoint
model_paths = [path for path in model_paths if 'checkpoint-' not in path]

if not model_paths:
    print(f"ERRORE: Nessun file 'pytorch_model.bin' trovato in {MODELS_BASE_DIR}. Controlla i percorsi.")
    exit()

print(f"Found {len(model_paths)} models.")

# --- 3. VALUTAZIONE ---
results = []
labels_num = len(np.unique(dataset.labels))

for model_path in sorted(model_paths):
    try:
        # Estrai il keep_ratio dal percorso
        path_parts = model_path.split(os.sep)
        # es: './results/70/pytorch_model.bin' -> '70'
        # es: './results/baseline/pytorch_model.bin' -> 'baseline'
        print(path_parts)
        ratio_str = path_parts[-2]
        if ratio_str == 'baseline':
            keep_ratio = 1.0
        else:
            keep_ratio = float(ratio_str) / 100.0
    except (ValueError, IndexError):
        print(f"ATTENZIONE: Impossibile determinare il keep_ratio per {model_path}. Salto il modello.")
        continue

    print(f"\n--- Evaluating model for keep_ratio: {keep_ratio} ---")
    print(f"Model path: {model_path}")
    print(keep_ratio)
    print(f"Evaluating model with labels num: {labels_num}")

    # Carica il modello
    model = ViT_UCB_Pruning(
        model_name="hf-hub:MahmoodLab/uni",
        pretrained=False,  # Non serve riscaricare i pesi pre-addestrati
        n_classes=labels_num,
        keep_ratio=keep_ratio, # Questo serve per l'architettura
        exclude_cls=True
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []
    total_inference_time = 0
    
    top_k_indices = None
    if keep_ratio < 1.0:
        try:
            top_k_indices = model.get_top_k_patch_indices(keep_ratio=keep_ratio)
            top_k_indices = top_k_indices.to(DEVICE)
            print(f"Pruning indices calculated. Keeping {len(top_k_indices)} tokens.")
        except Exception as e:
            print(f"Errore nel calcolo degli indici di pruning: {e}")
            continue

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Inferencing r={keep_ratio}"):
            images, labels = batch
            images = images.to(DEVICE)
            
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            if keep_ratio == 1.0:
                # Forward pass standard per il baseline
                outputs = model(images)
            else:
                # Forward pass con pruning
                outputs = model.forward_pruned(images, top_k_indices)
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            total_inference_time += (end_time - start_time)

            if hasattr(outputs, 'logits'):
                preds = torch.argmax(outputs.logits, dim=1)
            else:
                # Se outputs è un Tensor diretto (come sembra essere il caso per forward_pruned)
                preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calcola metriche
    avg_inference_time_ms = (total_inference_time / len(test_loader)) * 1000
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"F1-Score (Weighted): {f1:.4f}")
    print(f"Average Inference Time: {avg_inference_time_ms:.4f} ms/sample")

    results.append({
        'keep_ratio': keep_ratio,
        'f1_score': f1,
        'inference_time_ms': avg_inference_time_ms
    })


# --- 4. PLOTTING ---
if results:
    print("\n--- Plotting Results ---")
    df_results = pd.DataFrame(results).sort_values(by='keep_ratio')
    
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Grafico F1 Score
    sns.lineplot(data=df_results, x='keep_ratio', y='f1_score', marker='o', ax=ax1, color='b')
    ax1.set_title('Performance vs. Pruning Ratio', fontsize=16)
    ax1.set_xlabel('Keep Ratio (proportion of patches kept)')
    ax1.set_ylabel('F1-Score (Weighted)')
    ax1.invert_xaxis() # Mostra i ratio più aggressivi a destra

    # Grafico Tempo di Inferenza
    sns.lineplot(data=df_results, x='keep_ratio', y='inference_time_ms', marker='o', ax=ax2, color='r')
    ax2.set_title('Inference Speed vs. Pruning Ratio', fontsize=16)
    ax2.set_xlabel('Keep Ratio (proportion of patches kept)')
    ax2.set_ylabel('Average Inference Time (ms/sample)')
    ax2.invert_xaxis()

    for ax in [ax1, ax2]:
        for i, row in df_results.iterrows():
            ax.text(row['keep_ratio'], row[ax.get_ylabel().lower().replace(' ', '_')], f"{row[ax.get_ylabel().lower().replace(' ', '_')]:.2f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    print("Plot saved to 'evaluation_results.png'")
    # plt.show() # Decommenta per visualizzare il plot direttamente
else:
    print("Nessun risultato da plottare.")
