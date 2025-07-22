from torch.utils.data import Dataset
import os
import h5py
import pandas as pd
import numpy as np
import torch

import h5py
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset 
from huggingface_hub import snapshot_download
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class PatchFromH5Dataset(Dataset):
    def __init__(self, h5_dir, transform=None):
        self.h5_dir = h5_dir
        self.transform = transform

        metadata = pd.read_csv("hf://datasets/MahmoodLab/hest/HEST_v1_1_0.csv")
        metadata['oncotree_code'] = metadata['oncotree_code'].fillna('Healthy')
        metadata['oncotree_code'] = metadata['oncotree_code'].astype(str)

        # # Il numero di classi è 6, come previsto.
        # self.labels = ['SKCM', 'PRAD', 'COAD', 'SOC', 'CESC', 'BRCA']
        # # Filtra i metadati solo per le tue 6 classi
        # metadata = metadata[metadata['oncotree_code'].isin(self.labels)] 

        # self.sample_to_label = dict(zip(metadata['id'], metadata['oncotree_code']))
        # self.label_to_idx = {label: i for i, label in enumerate(self.labels)}

        # self.labels = ['Healthy', 'Cancer']
        # # Filtra i metadati solo per le classi di interesse
        # metadata = metadata[metadata['disease_state'].isin(self.labels)]

        self.labels = metadata['oncotree_code'].unique().tolist()

        self.sample_to_label = dict(zip(metadata['id'], metadata['oncotree_code']))
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}

        self.class_names = [None] * len(self.label_to_idx)
        for label, idx in self.label_to_idx.items():
            self.class_names[idx] = label

        # Debug: stampa le label effettive
        print("Unique labels in metadata:", set(metadata['oncotree_code']))
        print("Labels used:", self.labels)
        print("Sample to label mapping example:", list(self.sample_to_label.items())[:10])

        self.data_index = [] 

        for file in os.listdir(h5_dir):
            if file.endswith(".h5"):
                sample_id = file.replace('.h5', '')
                if sample_id in self.sample_to_label:
                    h5_path = os.path.join(h5_dir, file)
                    with h5py.File(h5_path, 'r') as f:
                        n_patches = len(f['img'])
                        for i in range(n_patches):
                            self.data_index.append((file, i))
                else:
                    print(f"Skipping {file}: label not in selected classes.")
    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        file_name, patch_idx = self.data_index[idx]
        sample_id = file_name.replace('.h5', '')
        label_str = self.sample_to_label[sample_id]
        label_idx = self.label_to_idx[label_str] # Ottieni l'indice intero della classe

        h5_path = os.path.join(self.h5_dir, file_name)
        with h5py.File(h5_path, 'r') as f:
            patch = f['img'][patch_idx] 

        patch = patch.astype(np.float32)
        patch = torch.tensor(patch)

        # Debug: stampa la shape del patch
        if patch.ndim == 2:
            patch = patch.unsqueeze(0)
        elif patch.shape[-1] == 3:
            patch = patch.permute(2, 0, 1)
        else:
            print(f"Unexpected patch shape: {patch.shape}")

        if self.transform:
            patch = self.transform(patch)

        # Restituisci l'indice di classe come un tensore di tipo long (necessario per CrossEntropyLoss)
        return patch, torch.tensor(label_idx, dtype=torch.long)
    

def download_dataset(repo_id = "MahmoodLab/hest", dest_folder = "/equilibrium/datasets/TCGA-histological-data/hest/patches", subfolder = "patches"):

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=dest_folder,
        allow_patterns=[f"{subfolder}/*"],
        local_dir_use_symlinks=False  # imposta True se vuoi risparmiare spazio
    )



def stratified_split(dataset, test_size=0.1, val_size=0.1, random_state=42):
    """
    Suddivide il dataset in train, val e test in modo stratificato.
    """
    labels = [dataset.label_to_idx[dataset.sample_to_label[file.replace('.h5', '')]]
              for (file, _) in dataset.data_index]
    indices = list(range(len(dataset)))

    # Primo split: train+val vs test
    trainval_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    # Labels per il secondo split
    trainval_labels = [labels[i] for i in trainval_idx]

    # Secondo split: train vs val
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=val_size / (1 - test_size),
        stratify=trainval_labels,
        random_state=random_state
    )

    return train_idx, val_idx, test_idx

def get_labels_from_indices(indices, full_dataset):
    return [
        full_dataset.label_to_idx[full_dataset.sample_to_label[full_dataset.data_index[i][0].replace('.h5', '')]]
        for i in indices
    ]

def plot_class_distributions(train_dataset, val_dataset, test_dataset, full_dataset, class_names=None):
    """
    Plotta la distribuzione delle classi nei set di train, val e test.
    """
    # Estrai i label
    train_labels = get_labels_from_indices(train_dataset, full_dataset)
    val_labels = get_labels_from_indices(val_dataset, full_dataset)
    test_labels = get_labels_from_indices(test_dataset, full_dataset)

    # Conta le occorrenze
    def count_labels(labels):
        return dict(Counter(labels))

    train_counts = count_labels(train_labels)
    val_counts = count_labels(val_labels)
    test_counts = count_labels(test_labels)

    # Classi ordinate
    all_classes = sorted(set(train_counts) | set(val_counts) | set(test_counts))
    if class_names is None:
        class_names = {i: str(i) for i in all_classes}

    # Crea DataFrame per il plot
    import pandas as pd
    data = []
    for split_name, counts in zip(['Train', 'Validation', 'Test'], [train_counts, val_counts, test_counts]):
        for cls in all_classes:
            data.append({
                'Split': split_name,
                'Class': class_names.get(cls, str(cls)),
                'Count': counts.get(cls, 0)
            })
    df = pd.DataFrame(data)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Class', y='Count', hue='Split')
    plt.title("Distribuzione delle classi nei set di Train/Val/Test")
    plt.tight_layout()
    plt.show()


