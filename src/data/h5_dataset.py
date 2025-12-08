
from torch.utils.data import Dataset
import os
import h5py
import pandas as pd
import numpy as np
import torch

class H5PatchDataset(Dataset):
    def __init__(self, h5_dir, transform=None):
        self.h5_dir = h5_dir
        self.transform = transform

        metadata = pd.read_csv("hf://datasets/MahmoodLab/hest/HEST_v1_1_0.csv")
        metadata['oncotree_code'] = metadata['oncotree_code'].fillna('Healthy')
        metadata['oncotree_code'] = metadata['oncotree_code'].astype(str)

        self.labels_str = metadata['oncotree_code'].unique().tolist()
        self.sample_to_label_str = dict(zip(metadata['id'], metadata['oncotree_code']))
        self.label_str_to_idx = {label: i for i, label in enumerate(self.labels_str)}

        self.class_names = [None] * len(self.label_str_to_idx)
        for label, idx in self.label_str_to_idx.items():
            self.class_names[idx] = label

        self.data_index = []
        self.labels = [] 

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
                            self.labels.append(label_for_file)
                else:
                    print(f"Skipping {file}: label not in selected classes.")

        self.labels = np.array(self.labels, dtype=np.long)

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        file_name, patch_idx = self.data_index[idx]
        
        h5_path = os.path.join(self.h5_dir, file_name)
        with h5py.File(h5_path, 'r') as f:
            patch = f['img'][patch_idx]

        patch = patch.astype(np.float32)
        patch = torch.tensor(patch)

        if patch.ndim == 2:
            patch = patch.unsqueeze(0)
        elif patch.shape[-1] == 3:
            patch = patch.permute(2, 0, 1)

        if self.transform:
            patch = self.transform(patch)

        label_idx = self.labels[idx]
        return patch, torch.tensor(label_idx, dtype=torch.long)
