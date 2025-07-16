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
from torch.utils.data import Dataset, Subset # Mantieni Subset se lo usi

class PatchFromH5Dataset(Dataset):
    def __init__(self, h5_dir, transform=None):
        self.h5_dir = h5_dir
        self.transform = transform

        metadata = pd.read_csv("hf://datasets/MahmoodLab/hest/HEST_v1_1_0.csv")
        # metadata['oncotree_code'] = metadata['oncotree_code'].fillna('Healthy')
        # metadata['oncotree_code'] = metadata['oncotree_code'].astype(str)

        # # Il numero di classi è 6, come previsto.
        # self.labels = ['SKCM', 'PRAD', 'COAD', 'SOC', 'CESC', 'BRCA']
        # # Filtra i metadati solo per le tue 6 classi
        # metadata = metadata[metadata['oncotree_code'].isin(self.labels)] 

        # self.sample_to_label = dict(zip(metadata['id'], metadata['oncotree_code']))
        # self.label_to_idx = {label: i for i, label in enumerate(self.labels)}

        self.labels = ['Healthy', 'Cancer']
        self.sample_to_label = dict(zip(metadata['id'], metadata['disease_state']))
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}

        self.data_index = [] 

        for file in os.listdir(h5_dir):
            if file.endswith(".h5"):
                sample_id = file.replace('.h5', '')
                # Aggiungi un controllo qui per assicurarti che il sample_id abbia una label che ti interessa
                if sample_id in self.sample_to_label:
                    h5_path = os.path.join(h5_dir, file)
                    with h5py.File(h5_path, 'r') as f:
                        n_patches = len(f['img']) 
                        for i in range(n_patches):
                            self.data_index.append((file, i)) 
                else:
                    # Opzionale: logga se un file .h5 viene saltato perché la sua label non è tra le 6
                    # print(f"Skipping {file_name}: label not in selected classes.")
                    pass
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

        if patch.ndim == 2: 
            patch = patch.unsqueeze(0) 
        elif patch.shape[-1] == 3: 
            patch = patch.permute(2, 0, 1) 

        if self.transform:
            patch = self.transform(patch)

        # Restituisci l'indice di classe come un tensore di tipo long (necessario per CrossEntropyLoss)
        return patch, torch.tensor(label_idx, dtype=torch.long)