
import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import time
import json
import os

from src.data.dataset import PatchFromH5Dataset
from src.rl.modelling import ViT_UCB_Pruning

# --- CONFIGURAZIONE ---
# Associa il percorso di ogni checkpoint al keep_ratio usato per la valutazione.
# MODIFICA QUESTO DIZIONARIO con i tuoi dati.
CHECKPOINTS_TO_BENCHMARK = {
    "/data/ucb_checkpoints/results2_freezed_0.3_lr/ratio_0.3/final_checkpoint/finetune-frozen-0.3_ratio_0.3.bin": 0.3,
    "/data/ucb_checkpoints/results2_freezed_0.5/ratio_0.5/final_checkpoint/finetune-frozen-0.5_ratio_0.5.bin": 0.5,
    "/data/ucb_checkpoints/results2_freezed_0.8_lr/ratio_0.8/final_checkpoint/finetune-frozen-0.8_ratio_0.8.bin": 0.8,
    "/data/ucb_checkpoints/results2_freezed_0.7_lr/ratio_0.7/final_checkpoint/finetune-frozen-0.7_ratio_0.7.bin": 0.7,
    "/data/ucb_checkpoints/results2_freezed_0.6_lr/ratio_0.6/final_checkpoint/finetune-frozen-0.6_ratio_0.6.bin": 0.6
}

# Parametri replicati da training.py per coerenza
IMG_SIZE = 224
BATCH_SIZE = 12 # Usiamo un batch size unico per il test
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
H5_DIR = '/data/patches/' # Assicurati che questo percorso sia corretto

# Parametri per il benchmark di velocità
WARMUP_RUNS = 10
BENCHMARK_RUNS = 50

# Directory per i risultati
RESULTS_DIR = "benchmark_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
# --- FINE CONFIGURAZIONE ---


def get_test_dataloader():
    """
    Replica la creazione del dataset e del dataloader di test da training.py
    per garantire che la valutazione sia fatta sugli stessi dati.
    """
    print("***** Creazione del Test Dataloader *****")
    dataset = PatchFromH5Dataset(
        h5_dir=H5_DIR,
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

    test_dataset = Subset(dataset, test_idx)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, drop_last=False) 
    
    print(f"Test dataset creato con {len(test_dataset)} campioni.")
    print("-------------------------------------------")
    return test_loader, dataset


def benchmark_speed(model, keep_ratio):
    """Misura la velocità di inferenza del modello potato."""
    print(f"--- Inizio Benchmark Velocità (keep_ratio: {keep_ratio}) ---")
    model.eval()
    
    try:
        top_k_indices = model.get_top_k_patch_indices(keep_ratio=keep_ratio)
    except Exception as e:
        print(f"Errore nel calcolo degli indici di pruning: {e}")
        return -1

    dummy_input = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    top_k_indices = top_k_indices.to(DEVICE)

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model.forward_pruned(dummy_input, top_k_indices)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    total_time = 0
    with torch.no_grad():
        for _ in range(BENCHMARK_RUNS):
            start_time = time.perf_counter()
            _ = model.forward_pruned(dummy_input, top_k_indices)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            total_time += (end_time - start_time)
    
    avg_time_ms = (total_time / BENCHMARK_RUNS) * 1000
    print(f"Tempo medio di inferenza: {avg_time_ms:.3f} ms")
    print("-------------------------------------------")
    return avg_time_ms

def evaluate_f1_score(model, dataloader, keep_ratio):
    """Valuta l'F1-score del modello potato sul test set."""
    print(f"--- Inizio Valutazione F1-Score (keep_ratio: {keep_ratio}) ---")
    model.eval()
    
    all_preds = []
    all_labels = []

    try:
        top_k_indices = model.get_top_k_patch_indices(keep_ratio=keep_ratio).to(DEVICE)
    except Exception as e:
        print(f"Errore nel calcolo degli indici di pruning: {e}")
        return -1

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(DEVICE)
            labels = batch[1].to(DEVICE)
            
            outputs = model.forward_pruned(inputs, top_k_indices)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    score = f1_score(all_labels, all_preds, average='weighted')
    print(f"F1-Score (weighted): {score:.4f}")
    print("-------------------------------------------")
    return score

def main():
    """
    Ciclo principale che carica i checkpoint, esegue benchmark e valutazione,
    e salva i risultati.
    """
    test_loader, dataset = get_test_dataloader()
    labels_num = len(np.unique(dataset.labels))
    
    results = []

    for checkpoint_path, keep_ratio in CHECKPOINTS_TO_BENCHMARK.items():
        print(f"***** Valutazione Checkpoint: {checkpoint_path} *****")
        print(f"***** Keep Ratio per Valutazione: {keep_ratio} *****")

        if not os.path.exists(checkpoint_path):
            print(f"ATTENZIONE: Il file {checkpoint_path} non esiste. Salto questo checkpoint.")
            continue

        # Inizializza il modello
        # Il `keep_ratio` qui è un placeholder, verrà sovrascritto dalla valutazione
        model = ViT_UCB_Pruning(
            model_name="hf-hub:MahmoodLab/uni",
            pretrained=False, # Non serve precaricare da HF, carichiamo il nostro stato
            n_classes=labels_num,
            keep_ratio=keep_ratio, 
            exclude_cls=True
        )

        # Carica i pesi dal checkpoint
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            model.to(DEVICE)
            print("Modello caricato con successo.")
        except Exception as e:
            print(f"Errore nel caricamento del modello: {e}")
            continue

        # 1. Benchmark di velocità
        avg_inference_time = benchmark_speed(model, keep_ratio)
        
        # 2. Valutazione F1-score
        f1 = evaluate_f1_score(model, test_loader, keep_ratio)

        results.append({
            "checkpoint_path": checkpoint_path,
            "keep_ratio": keep_ratio,
            "f1_score": f1,
            "avg_inference_time_ms": avg_inference_time
        })

    if not results:
        print("Nessun risultato da salvare. Controlla i percorsi dei checkpoint.")
        return

    # Crea un DataFrame e salvalo
    results_df = pd.DataFrame(results)
    
    csv_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    json_path = os.path.join(RESULTS_DIR, "benchmark_results.json")

    print("\n***** Risultati Finali *****")
    print(results_df)
    
    results_df.to_csv(csv_path, index=False)
    print(f"\nRisultati salvati in: {csv_path}")

    results_df.to_json(json_path, orient='records', indent=4)
    print(f"Risultati salvati in: {json_path}")


if __name__ == "__main__":
    main()
