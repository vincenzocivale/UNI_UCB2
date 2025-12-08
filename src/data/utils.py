
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Subset
from huggingface_hub import snapshot_download
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def stratified_split(input_dataset, test_size=0.1, val_size=0.1, random_state=42):
    """
    Splits the input_dataset into train, val, and test subsets in a stratified manner.
    """
    if isinstance(input_dataset, Subset):
        original_dataset = input_dataset.dataset
        effective_indices = input_dataset.indices
    else:
        original_dataset = input_dataset
        effective_indices = list(range(len(input_dataset)))

    labels_for_splitting = [original_dataset.labels[i] for i in effective_indices]

    trainval_local_idx, test_local_idx = train_test_split(
        list(range(len(effective_indices))),
        test_size=test_size,
        stratify=labels_for_splitting,
        random_state=random_state
    )

    trainval_local_labels = [labels_for_splitting[i] for i in trainval_local_idx]

    train_local_idx, val_local_idx = train_test_split(
        trainval_local_idx,
        test_size=val_size / (1 - test_size),
        stratify=trainval_local_labels,
        random_state=random_state
    )

    train_global_idx = [effective_indices[i] for i in train_local_idx]
    val_global_idx = [effective_indices[i] for i in val_local_idx]
    test_global_idx = [effective_indices[i] for i in test_local_idx]

    return train_global_idx, val_global_idx, test_global_idx

def get_labels_from_indices(indices, full_dataset):
    """
    Retrieves labels from subsets of indices using the .labels attribute of the dataset.
    """
    return [full_dataset.labels[i] for i in indices]

def download_dataset(repo_id="MahmoodLab/hest", dest_folder="/equilibrium/datasets/TCGA-histological-data/hest/patches", subfolder="patches"):
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
    train_labels = get_labels_from_indices(train_dataset.indices, full_dataset)
    val_labels = get_labels_from_indices(val_dataset.indices, full_dataset)
    test_labels = get_labels_from_indices(test_dataset.indices, full_dataset)

    def count_labels(labels):
        return dict(Counter(labels))

    train_counts = count_labels(train_labels)
    val_counts = count_labels(val_labels)
    test_counts = count_labels(test_labels)

    all_classes_indices = sorted(set(train_counts) | set(val_counts) | set(test_counts))
    
    if class_names is None:
        class_names_map = full_dataset.class_names
        class_names_dict = {i: class_names_map[i] for i in all_classes_indices}
    else:
        class_names_dict = {i: str(i) for i in all_classes_indices}

    data = []
    for split_name, counts in zip(['Train', 'Validation', 'Test'], [train_counts, val_counts, test_counts]):
        for cls_idx in all_classes_indices:
            data.append({
                'Split': split_name,
                'Class': class_names_dict.get(cls_idx, str(cls_idx)),
                'Count': counts.get(cls_idx, 0)
            })
    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Class', y='Count', hue='Split')
    plt.title("Class Distribution in Train/Val/Test Sets")
    plt.tight_layout()
    plt.show()
