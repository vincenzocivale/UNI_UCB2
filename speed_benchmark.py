import torch
import time
from src.rl.modelling import ViT_UCB_Pruning

# --- Parametri del Benchmark ---
BATCH_SIZE = 64
IMG_SIZE = 224
NUM_CHANNELS = 3
WARMUP_RUNS = 20
BENCHMARK_RUNS = 100
KEEP_RATIO = 0.5  # Conserviamo il 50% delle patch per il test

def benchmark(model, input_tensor, method_name, benchmark_runs, top_k_indices=None):
    """
    Funzione generica per misurare il tempo di inferenza di un dato metodo.
    """
    # Sposta il modello e i dati sul dispositivo corretto
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)
    if top_k_indices is not None:
        top_k_indices = top_k_indices.to(device)

    model.eval()
    
    method_to_call = getattr(model, method_name)

    # Crea gli argomenti per la chiamata al metodo
    args = (input_tensor,)
    if top_k_indices is not None:
        args = (input_tensor, top_k_indices)

    # 1. WARMUP: Esegui il metodo alcune volte per stabilizzare le performance
    #    e caricare i kernel CUDA in memoria.
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = method_to_call(*args)

    # Sincronizza per assicurarsi che il warmup sia completo prima di iniziare il timing
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 2. BENCHMARK: Misura il tempo di esecuzione
    total_time = 0
    with torch.no_grad():
        for _ in range(benchmark_runs):
            start_time = time.perf_counter()
            _ = method_to_call(*args)
            if device.type == 'cuda':
                torch.cuda.synchronize() # Forza l'attesa della fine dell'esecuzione sulla GPU
            end_time = time.perf_counter()
            total_time += (end_time - start_time)

    avg_time_ms = (total_time / benchmark_runs) * 1000
    return avg_time_ms

def main():
    """
    Esegue il confronto di velocità tra l'inferenza standard e quella potata.
    """
    print("***** Inizio del Benchmark di Velocità di Inferenza *****\n")
    
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo utilizzato: {device_name.upper()}")

    # 1. Inizializza il modello. Non usiamo pesi pre-addestrati.
    #    'ucb_count_scores' verrà inizializzato a 'ones', che è sufficiente
    #    per simulare una selezione di patch per il test di velocità.
    print("Caricamento del modello ViT_UCB_Pruning...")
    model = ViT_UCB_Pruning(pretrained=False, n_classes=10)
    
    # 2. Crea un tensore di input fittizio
    dummy_input = torch.randn(BATCH_SIZE, NUM_CHANNELS, IMG_SIZE, IMG_SIZE)
    
    # --- Benchmark del Modello Baseline (Completo) ---
    print(f"\nEsecuzione benchmark su modello completo ({BENCHMARK_RUNS} runs)...")
    baseline_avg_time = benchmark(
        model=model,
        input_tensor=dummy_input,
        method_name="forward",
        benchmark_runs=BENCHMARK_RUNS
    )
    print(f"Tempo medio inferenza (Baseline): {baseline_avg_time:.3f} ms")

    # --- Benchmark del Modello Potato ---
    # Estrai gli indici delle patch da conservare basandosi sui punteggi UCB
    print(f"\nCalcolo degli indici per il pruning con keep_ratio = {KEEP_RATIO}")
    top_k_indices = model.get_top_k_patch_indices(keep_ratio=KEEP_RATIO)
    
    num_total_tokens = model.pos_embed.shape[1]
    num_pruned_tokens = len(top_k_indices)
    print(f"Numero di token: {num_pruned_tokens}/{num_total_tokens} (CLS + {num_pruned_tokens - 1} patches)")

    print(f"Esecuzione benchmark su modello potato ({BENCHMARK_RUNS} runs)...")
    pruned_avg_time = benchmark(
        model=model,
        input_tensor=dummy_input,
        method_name="forward_pruned",
        benchmark_runs=BENCHMARK_RUNS,
        top_k_indices=top_k_indices
    )
    print(f"Tempo medio inferenza (Pruned): {pruned_avg_time:.3f} ms")

    # --- Risultati Finali ---
    print("\n\n--- Risultati del Benchmark ---")
    print(f"Tempo medio Baseline: {baseline_avg_time:.3f} ms")
    print(f"Tempo medio Pruned ({KEEP_RATIO*100}% dei token): {pruned_avg_time:.3f} ms")
    
    if pruned_avg_time > 0:
        speedup_factor = baseline_avg_time / pruned_avg_time
        speedup_percentage = (speedup_factor - 1) * 100
        print(f"\nGuadagno di velocità: {speedup_factor:.2f}x")
        print(f"L'inferenza potata è ~{speedup_percentage:.1f}% più veloce.")
    print("---------------------------------\n")


if __name__ == "__main__":
    main()
