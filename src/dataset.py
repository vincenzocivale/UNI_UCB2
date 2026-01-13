from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm.auto  import tqdm
from datasets import load_from_disk

class HistologicalImageDataset(Dataset):
    """Optimized dataset with pre-extracted images for fast training"""
    
    def __init__(self, data_dir, transform=None, split='train', test_size=0.15, val_size=0.15, seed=42):
        self.transform = transform
        self.hf_dataset = load_from_disk(data_dir)
        
        # Label mapping
        unique_labels = sorted(set(self.hf_dataset['label']))
        self.class_names = unique_labels
        self.label_str_to_idx = {label: i for i, label in enumerate(unique_labels)}
        
        all_labels = np.array([self.label_str_to_idx[l] for l in self.hf_dataset['label']], dtype=np.int64)
        
        # 3-way split
        indices = np.arange(len(self.hf_dataset))
        train_idx, temp_idx = train_test_split(
            indices, test_size=test_size+val_size, stratify=all_labels, random_state=seed
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, stratify=all_labels[temp_idx], random_state=seed
        )
        
        if split == 'train':
            self.indices = train_idx
        elif split == 'val':
            self.indices = val_idx
        else:  # test
            self.indices = test_idx
        
        self.labels = all_labels[self.indices]
        print(f"{split.upper()}: {len(self.indices)} samples")
        
        # Pre-load images
        self.images = [np.array(self.hf_dataset[i]['image']) for i in tqdm(self.indices, desc="Loading")]
        
       
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Fast: direct access to pre-loaded numpy array
        image = self.images[idx]
        
        # Convert to tensor
        patch = torch.from_numpy(image).float()
        
        # HWC -> CHW
        if patch.ndim == 3 and patch.shape[2] == 3:
            patch = patch.permute(2, 0, 1)
        
        # Normalize to [0, 1]
        patch = patch / 255.0
        
        # Apply transform
        if self.transform:
            patch = self.transform(patch)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return patch, label