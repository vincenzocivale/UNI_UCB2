# Report Dettagliato: Pruning di Patch basato su UCB in Vision Transformer (ViT)

Questo report descrive l'implementazione di un Vision Transformer (ViT) modificato che esegue il pruning dinamico delle patch basato sull'Upper Confidence Bound (UCB). L'obiettivo è ridurre la complessità computazionale del modello durante l'inferenza mantenendo l'accuratezza, sfruttando un backbone ViT pre-addestrato.

## 1. Architettura del ViT Modificato: `ViT_UCB_Pruning`

Il modello `ViT_UCB_Pruning` è una versione estesa di un Vision Transformer standard, costruita a partire da un backbone pre-addestrato (tipicamente caricato tramite la libreria `timm`).

### Componenti principali:

-   **Backbone Pre-addestrato**: Il modello inizia caricando un'architettura ViT pre-addestrata (es. `hf-hub:MahmoodLab/uni`). Vengono mantenuti i moduli di `patch_embed`, `cls_token`, `pos_embed`, `pos_drop` e `norm` originali.
-   **`UCBBlock` (Blocchi Trasformatore Modificati)**: I blocchi del trasformatore originali del backbone vengono sostituiti con `UCBBlock` personalizzati. Ogni `UCBBlock` è un wrapper che mantiene la struttura di un blocco trasformatore standard ma sostituisce il modulo di attenzione (`Attention`) con un `UCBAttention` personalizzato. I pesi dei moduli di attenzione originali vengono copiati nel corrispondente `UCBAttention` per preservare le capacità del modello pre-addestrato.
-   **`UCBAttention` (Modulo di Attenzione con Pruning UCB)**: Questo è il cuore della logica di pruning. Invece di calcolare l'attenzione su tutte le patch in ogni layer, `UCBAttention` seleziona dinamicamente un sottoinsieme di patch più "importanti" utilizzando la strategia UCB.
-   **`ucb_count_scores` (Buffer di Conteggio UCB)**: Un tensore registrato nel modello (`self.register_buffer("ucb_count_scores", ...)`) che memorizza la frequenza con cui ogni patch (per ogni layer e per ogni head di attenzione) è stata selezionata durante il training. Questo buffer viene aggiornato continuamente e serve come base per il pruning deterministico in inferenza.

## 2. Processo di Pruning delle Patch

Il processo di pruning si articola in due fasi principali: una fase di *selezione dinamica* durante il training (usando UCB per "esplorare" e "sfruttare" l'importanza delle patch) e una fase di *pruning fisico* durante l'inferenza (basato sui conteggi appresi).

### 2.1. Fase di Training (Selezione Dinamica con UCB)

Durante il training, il modello `ViT_UCB_Pruning` viene chiamato con `ucb_enabled=True` (dopo un periodo di "warm-up" di 50 passi per stabilizzare il training iniziale).

1.  **Calcolo delle Probabilità di Attenzione**: Per ogni `UCBAttention` all'interno di un `UCBBlock`, vengono calcolate le probabilità di attenzione (`attn_probs`) come in un trasformatore standard.
2.  **Calcolo dei Punteggi di Importanza delle Patch**: Viene calcolato un punteggio di importanza per ogni patch, basato sull'attenzione media che riceve (escludendo il CLS token se `exclude_cls` è attivo).
3.  **Applicazione della Formula UCB**: Viene applicata la strategia UCB (Upper Confidence Bound) per bilanciare l'**esplorazione** (scoprire nuove patch importanti) e lo **sfruttamento** (selezionare patch già note come importanti).
    -   La formula UCB combina il punteggio di importanza della patch con un termine di esplorazione:
        `UCB_Score = Punteggio_Importanza_Patch + beta * sqrt(log(iterazione + 1) / (conteggio_selezioni + epsilon))`
    -   `iterazione` è il passo globale di training (`counter`).
    -   `conteggio_selezioni` è preso dal buffer `ucb_count_scores` per la specifica patch, layer e head.
    -   `beta` è un iperparametro che controlla il bilanciamento tra esplorazione e sfruttamento.
4.  **Selezione delle Patch**: Vengono selezionate le `k` patch (o una percentuale `keep_ratio` delle patch) con i `UCB_Score` più alti. Questa selezione può essere globale (stesse patch per tutte le head di attenzione all'interno di un blocco) o per head, a seconda dell'implementazione. Il CLS token viene sempre mantenuto.
5.  **Creazione della Maschera di Attenzione**: Viene creata una maschera che zero-out le probabilità di attenzione per le patch che non sono state selezionate. Le interazioni del CLS token sono sempre preservate (può attendere a tutte le patch e tutte le patch possono attendere al CLS).
6.  **Rinormalizzazione e Propagazione**: Le probabilità di attenzione mascherate vengono rinormalizzate e usate per calcolare il `context` (output del modulo di attenzione), che poi procede attraverso il blocco del trasformatore.
7.  **Aggiornamento del Buffer `ucb_count_scores`**: Il buffer `ucb_count_scores` viene aggiornato, incrementando il conteggio per le patch che sono state selezionate in questo passo di training. Questo accumulo fornisce una stima dell'importanza a lungo termine delle patch.

### 2.2. Fase di Inferenza/Valutazione (Pruning Fisico)

Durante la valutazione o l'inferenza, il modello esegue un pruning *fisico* dei token di input prima di passarli attraverso i blocchi del trasformatore.

1.  **Determinazione delle Patch più Importanti**: Utilizzando i `ucb_count_scores` accumulati durante il training, il metodo `get_top_k_patch_indices` calcola le patch globalmente più importanti. Tipicamente, questo avviene mediando i conteggi di selezione di ogni patch attraverso tutti i layer e tutte le head di attenzione (escluso il CLS token).
2.  **Selezione degli Indici**: Vengono selezionati gli indici delle `k` patch (o la percentuale `keep_ratio` di patch) con il conteggio di selezione più alto. Il CLS token (indice 0) viene sempre incluso.
3.  **Pruning Fisico dell'Input**: Prima di entrare nei blocchi del trasformatore, il tensore di input `x` viene ridotto (`torch.index_select`) per includere **solo** il CLS token e le patch selezionate.
4.  **Forward Pass**: Il set ridotto di token viene quindi passato attraverso i blocchi del trasformatore. In questa fase, i moduli `UCBAttention` operano in modalità standard (senza applicare la logica UCB di mascheramento, in quanto il pruning è già avvenuto "a monte").

## 3. Utilizzo di un Backbone Pre-addestrato

Il processo inizia caricando un modello ViT pre-addestrato da `timm` (es. "hf-hub:MahmoodLab/uni"). Questo è cruciale perché fornisce al modello una forte capacità di rappresentazione fin dall'inizio. L'algoritmo UCB non "impara da zero" l'importanza delle patch, ma piuttosto affina e adatta questa conoscenza derivata dal pre-training, concentrando le risorse computazionali sulle aree più salienti per il task di downstream, senza sacrificare eccessivamente le performance.

## 4. Punti Chiave per un Ingegnere (Modifica/Miglioramento della Logica di Pruning)

Per un ingegnere che desidera migliorare o modificare la logica di pruning, ecco i punti chiave e le aree di intervento:

### Aree di Codice Rilevanti:

-   **`src/rl/modelling.py`**: Contiene le definizioni di `UCBAttention`, `UCBBlock` e `ViT_UCB_Pruning`. Questa è la fonte primaria per modificare la logica di selezione delle patch.
    -   Metodo `ucb_score_pruning`: Implementazione diretta della logica UCB.
    -   Metodo `get_top_k_patch_indices`: Logica per determinare le patch finali da mantenere per l'inferenza.
-   **`src/rl/train.py`**: Contiene la classe `ModelTrainer` che gestisce il ciclo di training, inclusa l'attivazione del pruning e la gestione dei conteggi UCB.
    -   Metodo `_forward_pass`: Controlla come il modello `ViT_UCB_Pruning` viene chiamato durante training e valutazione, inclusa l'abilitazione UCB e il pruning fisico.
    -   `_log_pruning_metrics`: Utile per aggiungere o modificare metriche di logging relative al pruning.

### Parametri e Iperparametri da Esplorare/Modificare:

1.  **`k` / `keep_ratio`**:
    -   **Descrizione**: `k` è il numero fisso di patch da mantenere; `keep_ratio` è la percentuale di patch da mantenere.
    -   **Impatto**: Determina direttamente il livello di pruning. Un valore più basso di `keep_ratio` (o `k`) porta a un maggiore pruning e potenziali risparmi computazionali, ma può anche ridurre l'accuratezza.
    -   **Modifica**: Sperimentare con diversi valori per trovare il punto di equilibrio tra performance e efficienza. Potrebbe essere interessante rendere `k` o `keep_ratio` dinamici (es. per layer o in base alla complessità dell'immagine).
2.  **`beta` (Parametro di Esplorazione UCB)**:
    -   **Descrizione**: Controlla l'importanza del termine di esplorazione nella formula UCB. Un `beta` più alto incoraggia il modello a esplorare più patch meno selezionate.
    -   **Impatto**: Influenza il bilanciamento tra sfruttare le patch note come importanti e esplorare nuove patch che potrebbero rivelarsi rilevanti. Un `beta` troppo basso può portare a una selezione locale; troppo alto può rallentare la convergenza.
    -   **Modifica**: Tuning di `beta` per ottimizzare l'equilibrio tra esplorazione e sfruttamento per il task specifico.
3.  **`exclude_cls`**:
    -   **Descrizione**: Booleano che indica se il CLS token debba essere escluso dal calcolo dei punteggi di importanza delle patch (ma viene sempre mantenuto nel forward pass).
    -   **Impatto**: Modifica come i punteggi di importanza delle patch vengono calcolati.
    -   **Modifica**: Testare l'effetto sulla selezione delle patch e sulle performance.
4.  **Periodo di Warm-up (`counter > 50`)**:
    -   **Descrizione**: Il numero di passi di training prima che la logica UCB sia attivata.
    -   **Impatto**: Permette al modello di stabilizzare le sue rappresentazioni iniziali prima di iniziare il pruning.
    -   **Modifica**: Aumentare o diminuire il warm-up in base alla stabilità del training.
5.  **Strategia di Aggiornamento `ucb_count_scores`**:
    -   **Descrizione**: Attualmente, per ogni selezione, il conteggio della patch viene incrementato di `1.0 / B` (dove B è la batch size) per normalizzare l'aggiornamento.
    -   **Impatto**: Influenza il modo in cui i punteggi di importanza si accumulano nel tempo.
    -   **Modifica**: Potrebbe essere esplorato un aggiornamento pesato (es. in base all'attenzione effettiva ricevuta dalla patch, o un decadimento esponenziale dei vecchi conteggi).
6.  **Metodo di Aggregazione dei Punteggi per `get_top_k_patch_indices`**:
    -   **Descrizione**: Attualmente, i punteggi UCB vengono mediati su tutti i layer e le head per ottenere un punteggio globale di importanza delle patch.
    -   **Impatto**: Determina quali patch vengono considerate globalmente più importanti per il pruning fisico.
    -   **Modifica**: Esplorare altre strategie di aggregazione (es. somma, massimo, media pesata per layer, ecc.) o metodi più sofisticati per combinare i punteggi tra i layer.
7.  **Strategia di Reinormalizzazione dell'Attenzione**:
    -   **Descrizione**: `pruned_attn = pruned_attn / (pruned_attn.sum(dim=-1, keepdim=True) + 1e-8)` garantisce che le probabilità di attenzione rimangano valide dopo il mascheramento.
    -   **Impatto**: Cruciale per la stabilità numerica.
    -   **Modifica**: In casi specifici, la strategia di rinormalizzazione potrebbe essere affinata.
8.  **Pruning Per-Layer o Dinamico**:
    -   **Descrizione**: Attualmente, `k` o `keep_ratio` sono gli stessi per tutti i layer.
    -   **Modifica**: Si potrebbe implementare una strategia in cui i layer iniziali o finali potano in modo diverso, o in cui la decisione di potare dipenda dinamicamente dalle caratteristiche dell'input.
9.  **Fattore di Congelamento del Backbone (`freeze_backbone`)**:
    -   **Descrizione**: L'argomento `freeze_backbone` in `TrainingArguments` suggerisce la possibilità di congelare i pesi del backbone durante il training.
    -   **Impatto**: Congelare il backbone può velocizzare il training e prevenire l'overfitting, focalizzando l'apprendimento sulla logica di pruning e sul head di classificazione.
    -   **Modifica**: Sperimentare con il congelamento parziale (es. solo i primi N layer) o totale del backbone.

