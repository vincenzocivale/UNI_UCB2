from torch.utils.data import Dataset
import os
import h5py
import pandas as pd
import numpy as np
import torch

from torch.utils.data import Subset
from huggingface_hub import snapshot_download
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class PatchFromH5Dataset(Dataset):
    def __init__(self, h5_dir, transform=None, organ_filter=None):
        self.h5_dir = h5_dir
        self.transform = transform
        self.organ_filter = organ_filter

        metadata = pd.read_csv("hf://datasets/MahmoodLab/hest/HEST_v1_1_0.csv")
        metadata["oncotree_code"] = metadata["oncotree_code"].fillna("Healthy").astype(str)
        metadata["organ"] = metadata["organ"].astype(str)

        # Map sample_id -> label_str e organ_str
        self.sample_to_label_str = dict(zip(metadata["id"], metadata["oncotree_code"]))
        self.sample_to_organ_str = dict(zip(metadata["id"], metadata["organ"]))

        # Filtra sample_id in base all'organo
        filtered_sample_ids = [
            sample_id
            for sample_id, organ in self.sample_to_organ_str.items()
            if organ_filter is None or organ == organ_filter
        ]

        # Trova le classi presenti nel dataset filtrato
        labels_in_filtered = set(self.sample_to_label_str[sid] for sid in filtered_sample_ids)
        self.labels_str = sorted(labels_in_filtered)

        # Ricostruisci mappa stringa->indice classi solo per quelle presenti
        self.label_str_to_idx = {label: i for i, label in enumerate(self.labels_str)}

        # Organ names e mappatura
        organs_in_filtered = set(self.sample_to_organ_str[sid] for sid in filtered_sample_ids)
        self.organ_names = sorted(organs_in_filtered)
        self.organ_str_to_idx = {o: i for i, o in enumerate(self.organ_names)}

        self.class_names = self.labels_str

        self.data_index = []
        self.labels = []
        self.organs = []

        for file in os.listdir(h5_dir):
            if not file.endswith(".h5"):
                continue

            sample_id = file.replace(".h5", "")
            if sample_id not in filtered_sample_ids:
                continue

            sample_organ = self.sample_to_organ_str[sample_id]
            h5_path = os.path.join(h5_dir, file)
            with h5py.File(h5_path, "r") as f:
                n_patches = len(f["img"])

            label_idx = self.label_str_to_idx[self.sample_to_label_str[sample_id]]
            organ_idx = self.organ_str_to_idx[sample_organ]

            for i in range(n_patches):
                self.data_index.append((file, i))
                self.labels.append(label_idx)
                self.organs.append(organ_idx)

        self.labels = np.array(self.labels, dtype=np.int64)
        self.organs = np.array(self.organs, dtype=np.int64)

        if organ_filter is not None:
            print(
                f"Dataset filtered on organ='{organ_filter}': "
                f"{len(self.labels)} patches, {len(self.labels_str)} classes"
            )



    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        file_name, patch_idx = self.data_index[idx]

        h5_path = os.path.join(self.h5_dir, file_name)
        with h5py.File(h5_path, "r") as f:
            patch = f["img"][patch_idx]

        patch = torch.tensor(patch, dtype=torch.float32)

        if patch.ndim == 2:
            patch = patch.unsqueeze(0)
        elif patch.shape[-1] == 3:
            patch = patch.permute(2, 0, 1)

        if self.transform:
            patch = self.transform(patch)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        organ = torch.tensor(self.organs[idx], dtype=torch.long)

        return patch, label, organ

    

class EagleEmbeddingDataset(Dataset):
    """
    Una classe PyTorch Dataset per la lettura di embeddings da un file HDF5.
    """
    def __init__(self, h5_file_path: str, metadata_df: pd.DataFrame, label_map: dict):
        self.h5_file_path = h5_file_path
        self.metadata = metadata_df
        self.label_map = label_map
        self.h5_file = h5py.File(h5_file_path, 'r')

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int):
        # Ottieni l'ID e la label dal DataFrame dei metadati
        row = self.metadata.iloc[idx]
        embedding_id = str(row['id'])  # 'id' deve essere una stringa per HDF5
        label_str = row['oncotree_code']
        
        # Converte la stringa della label nell'ID numerico
        label = self.label_map[label_str]
        
        # Accedi all'embedding corrispondente nel file HDF5
        embedding = self.h5_file[embedding_id][()]  # [()] carica i dati in memoria come array NumPy

        return {'embedding': torch.tensor(embedding, dtype=torch.float32), 'label': torch.tensor(label, dtype=torch.int64)}
    
    def close(self):
        self.h5_file.close()
    
# --- Funzioni ausiliarie modificate per usare il nuovo attributo labels ---

def stratified_split(input_dataset, test_size=0.1, val_size=0.1, random_state=42):
    """
    Splits the input_dataset (which can be a full Dataset or a Subset)
    into train, val, and test subsets in a stratified manner.
    It accesses the original dataset's labels attribute for stratification.
    """
    # Determine the underlying full dataset and the effective indices
    if isinstance(input_dataset, Subset):
        original_dataset = input_dataset.dataset # Access the original dataset from the Subset
        effective_indices = input_dataset.indices # Get the indices of the Subset
    else: # It's already the full dataset
        original_dataset = input_dataset
        effective_indices = list(range(len(input_dataset)))

    # Get the labels corresponding to the effective_indices from the original_dataset
    # This is efficient because original_dataset.labels is pre-computed
    labels_for_splitting = [original_dataset.labels[i] for i in effective_indices]

    # Perform the first split: train+val vs test
    trainval_local_idx, test_local_idx = train_test_split(
        list(range(len(effective_indices))), # Split local indices of the input_dataset
        test_size=test_size,
        stratify=labels_for_splitting,
        random_state=random_state
    )

    # Get labels for the second split using the local indices
    trainval_local_labels = [labels_for_splitting[i] for i in trainval_local_idx]

    # Perform the second split: train vs val
    train_local_idx, val_local_idx = train_test_split(
        trainval_local_idx,
        test_size=val_size / (1 - test_size),
        stratify=trainval_local_labels,
        random_state=random_state
    )

    # Convert local indices back to original dataset indices
    train_global_idx = [effective_indices[i] for i in train_local_idx]
    val_global_idx = [effective_indices[i] for i in val_local_idx]
    test_global_idx = [effective_indices[i] for i in test_local_idx]

    return train_global_idx, val_global_idx, test_global_idx

def get_labels_from_indices(indices, full_dataset):
    """
    Recupera le label dai subset di indici usando l'attributo .labels del dataset.
    """
    return [full_dataset.labels[i] for i in indices]



def download_dataset(repo_id = "MahmoodLab/hest", dest_folder = "/equilibrium/datasets/TCGA-histological-data/hest/patches", subfolder = "patches"):
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=dest_folder,
        allow_patterns=[f"{subfolder}/*"],
        local_dir_use_symlinks=False
    )

def plot_class_distributions(train_dataset, val_dataset, test_dataset, full_dataset, class_names=None):
    """
    Plot the class distributions in train, val, and test sets.
    """
    # Extract labels
    train_labels = get_labels_from_indices(train_dataset.indices, full_dataset) # Access .indices of Subset
    val_labels = get_labels_from_indices(val_dataset.indices, full_dataset)     # Access .indices of Subset
    test_labels = get_labels_from_indices(test_dataset.indices, full_dataset)   # Access .indices of Subset

    # Count occurrences
    def count_labels(labels):
        return dict(Counter(labels))

    train_counts = count_labels(train_labels)
    val_counts = count_labels(val_labels)
    test_counts = count_labels(test_labels)

    # Ordered classes
    all_classes_indices = sorted(set(train_counts) | set(val_counts) | set(test_counts))
    
    # Use class_names from the full_dataset if not provided
    if class_names is None:
        class_names_map = full_dataset.class_names # Get the list of string names
        # Map indices to names
        class_names_dict = {i: class_names_map[i] for i in all_classes_indices}
    else:
        class_names_dict = {i: str(i) for i in all_classes_indices} # Fallback if class_names is provided but not mapping

    # Create DataFrame for plotting
    data = []
    for split_name, counts in zip(['Train', 'Validation', 'Test'], [train_counts, val_counts, test_counts]):
        for cls_idx in all_classes_indices:
            data.append({
                'Split': split_name,
                'Class': class_names_dict.get(cls_idx, str(cls_idx)), # Use mapped class name
                'Count': counts.get(cls_idx, 0)
            })
    df = pd.DataFrame(data)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Class', y='Count', hue='Split')
    plt.title("Distribuzione delle classi nei set di Train/Val/Test")
    plt.tight_layout()
    plt.show()