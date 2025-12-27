# Vision Transformer Pruning with Upper Confidence Bound (UCB)

Questo progetto implementa una strategia di pruning dinamico per modelli Vision Transformer (ViT). L'obiettivo è identificare e utilizzare solo il sottoinsieme più saliente di patch di un'immagine durante il training, al fine di creare modelli finali più efficienti in fase di inferenza.

## Metodologia

Il nucleo del progetto si basa su un meccanismo di attenzione modificato e una strategia di addestramento a due fasi.

### 1. Calcolo dello Score UCB per il Pruning Dinamico

Durante il training, i blocchi di attenzione standard del ViT sono sostituiti da una versione personalizzata, `UCBAttention`, che applica un pruning dinamico. Per ogni forward pass, la decisione su quali patch (token) scartare si basa su uno score UCB calcolato per ciascuna patch.

Lo score è definito come:

`UCB Score = Score di Importanza + Bonus di Esplorazione`

-   **Score di Importanza**: Deriva direttamente dalle probabilità di attenzione del modello. Corrisponde a quanta attenzione una patch riceve, in media, da tutte le altre. Questo termine rappresenta lo sfruttamento (exploitation) della conoscenza attuale del modello.

-   **Bonus di Esplorazione**: È un termine che incentiva la selezione di patch visitate meno frequentemente. Viene calcolato con la formula UCB1:
    ```
    Bonus = β * sqrt(log(t) / N(p))
    ```
    -   `β` è un iperparametro che bilancia l'esplorazione.
    -   `t` è lo step di training corrente.
    -   `N(p)` è il numero di volte che la patch `p` è stata selezionata fino a quel momento.

Ad ogni blocco, vengono mantenute solo le `k` patch con il punteggio UCB più alto, dove `k` è determinato dal `keep_ratio`. Un buffer (`ucb_count_scores`) viene aggiornato per tenere traccia delle selezioni e informare il calcolo del bonus futuro.

### 2. Strategia di Addestramento a Due Fasi

Un pruning aggressivo e dinamico fin dall'inizio può rendere l'addestramento instabile. Per risolvere questo problema, il training è diviso in due fasi sequenziali:

#### Fase 1: Esplorazione e Mappatura dell'Importanza
L'obiettivo di questa fase è costruire una "mappa" affidabile dell'importanza di ogni patch, senza la pressione di un pruning aggressivo.

-   **Script**: `training.py`
-   **Metodo**: Il modello viene addestrato con la logica UCB attiva ma con un `keep_ratio` molto alto (es. `1.0` o `0.9`). In questo modo, il modello esplora e apprende quali patch sono importanti (aggiornando il buffer `ucb_count_scores`) senza subire un calo di performance dovuto allo scarto di troppe informazioni.
-   **Risultato**: Un modello i cui `ucb_count_scores` rappresentano una stima stabile dell'importanza di ogni patch per il task specifico.

#### Fase 2: Fine-tuning con Pruning Fisso
L'obiettivo di questa fase è specializzare il modello a operare con un budget di patch ridotto e fisso, simulando la condizione di inferenza finale.

-   **Script**: `training_stage2_freeze.py`
-   **Metodo**:
    1.  Si carica il modello salvato dalla Fase 1.
    2.  Si utilizzano i suoi `ucb_count_scores` per determinare un **singolo e fisso** insieme delle `k` patch più importanti, in base al `frozen_keep_ratio` desiderato (es. `0.3`).
    3.  Il modello viene ulteriormente addestrato (fine-tuning), ma in ogni step di training vengono processate **esclusivamente** le patch di questo insieme fisso. La logica UCB dinamica è disattivata.
-   **Risultato**: Un modello performante e specializzato per operare con un basso numero di patch, pronto per un'inferenza efficiente.

## Come Eseguire l'Addestramento

Il processo completo richiede l'esecuzione sequenziale dei due script di training.

### Prerequisiti
Assicurarsi che le dipendenze siano installate e che il dataset sia accessibile al percorso specificato negli script (`--h5_dir`).

### Fase 1: Eseguire `training.py`
Addestrare il modello con un `keep_ratio` alto per apprendere l'importanza delle patch.

```bash
python training.py \
  --output_dir ./results_stage1 \
  --run_name "ViT_Stage1_Explore_ratio_1.0" \
  --keep_ratio 1.0 \
  --num_train_epochs 50 \
  --learning_rate 0.01
```
Al termine, verrà salvato un checkpoint (es. in `results_stage1/best_model/ViT-UCB-Training-keep_ratio-1.0.bin`).

### Fase 2: Eseguire `training_stage2_freeze.py`
Eseguire il fine-tuning partendo dal checkpoint della Fase 1 e applicando un pruning fisso e aggressivo.

```bash
python training_stage2_freeze.py \
  --stage1_checkpoint_path ./results_stage1/best_model/ViT-UCB-Training-keep_ratio-1.0.bin \
  --frozen_keep_ratio 0.3 \
  --output_dir ./results_stage2_freeze \
  --run_name "ViT_Stage2_Finetune_Freeze" \
  --num_train_epochs 25 \
  --learning_rate 5e-5
```
Questo produrrà il modello finale, ottimizzato per un `keep_ratio` di 0.3.
