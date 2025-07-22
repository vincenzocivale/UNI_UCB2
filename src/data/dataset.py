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
    def __init__(self, h5_dir, transform=None):
        self.h5_dir = h5_dir
        self.transform = transform

        metadata = pd.read_csv("hf://datasets/MahmoodLab/hest/HEST_v1_1_0.csv")
        metadata['oncotree_code'] = metadata['oncotree_code'].fillna('Healthy')
        metadata['oncotree_code'] = metadata['oncotree_code'].astype(str)

        self.labels_str = metadata['oncotree_code'].unique().tolist() # Nomi stringa delle classi
        self.sample_to_label_str = dict(zip(metadata['id'], metadata['oncotree_code']))
        self.label_str_to_idx = {label: i for i, label in enumerate(self.labels_str)}

        self.class_names = [None] * len(self.label_str_to_idx)
        for label, idx in self.label_str_to_idx.items():
            self.class_names[idx] = label



        self.data_index = []
        # Aggiungeremo una lista per le label numeriche corrispondenti a data_index
        self.labels = [] # Questo sarà il nostro array NumPy delle label pre-calcolate

        for file in os.listdir(h5_dir):
            if file.endswith(".h5"):
                sample_id = file.replace('.h5', '')
                if sample_id in self.sample_to_label_str:
                    h5_path = os.path.join(h5_dir, file)
                    with h5py.File(h5_path, 'r') as f:
                        n_patches = len(f['img'])
                        label_for_file = self.label_str_to_idx[self.sample_to_label_str[sample_id]]
                        for i in range(n_patches):
                            self.data_index.append((file, i))
                            self.labels.append(label_for_file) # Salva la label numerica per ogni patch
                else:
                    print(f"Skipping {file}: label not in selected classes.")

        # Converti la lista di label in un array NumPy alla fine
        self.labels = np.array(self.labels, dtype=np.long) # Usa np.long per coerenza con torch.long
       

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        file_name, patch_idx = self.data_index[idx]
        
        # Recupera la patch
        h5_path = os.path.join(self.h5_dir, file_name)
        with h5py.File(h5_path, 'r') as f:
            patch = f['img'][patch_idx]

        patch = patch.astype(np.float32)
        patch = torch.tensor(patch)

        if patch.ndim == 2:
            patch = patch.unsqueeze(0)
        elif patch.shape[-1] == 3:
            patch = patch.permute(2, 0, 1)
        # else: Aggiungi un controllo più robusto qui se ci sono altre forme inattese
        #    print(f"Unexpected patch shape: {patch.shape}") # Debugging

        if self.transform:
            patch = self.transform(patch)

        # Recupera la label direttamente dall'array pre-calcolato
        label_idx = self.labels[idx]
        return patch, torch.tensor(label_idx, dtype=torch.long)
    
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