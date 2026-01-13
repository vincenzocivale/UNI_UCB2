"""
Simplified dataset class for pre-split histological images
"""

from torch.utils.data import Dataset
from datasets import load_from_disk
import numpy as np
import torch
from tqdm.auto import tqdm


class HistologicalImageDataset(Dataset):
    """
    Simple dataset for pre-split histological images.
    Expects dataset structure:
        data/DATASET_NAME/
            train/
            val/
            test/
    """
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Path to specific split (e.g., "data/BACH/train")
            transform: Optional transforms
        """
        self.transform = transform
        
        # Load dataset
        print(f"Loading from {data_dir}...")
        self.hf_dataset = load_from_disk(data_dir)
        
        # Extract class names
        label_feature = self.hf_dataset.features.get('label')
        if hasattr(label_feature, 'names'):
            self.class_names = list(label_feature.names)
        else:
            unique_labels = sorted(set(self.hf_dataset['label']))
            if isinstance(unique_labels[0], str):
                self.class_names = unique_labels
            else:
                self.class_names = [f"Class_{i}" for i in unique_labels]
        
        # Pre-load images and labels
        print(f"Pre-loading {len(self.hf_dataset)} images...")
        self.images = []
        self.labels = []
        
        for i in tqdm(range(len(self.hf_dataset)), desc="Loading"):
            self.images.append(np.array(self.hf_dataset[i]['image']))
            self.labels.append(self.hf_dataset[i]['label'])
        
        self.labels = np.array(self.labels, dtype=np.int64)
        
        print(f"Loaded {len(self)} samples, {len(self.class_names)} classes")
        self._print_class_distribution()
    
    def _print_class_distribution(self):
        """Print class distribution"""
        unique, counts = np.unique(self.labels, return_counts=True)
        print("Class distribution:")
        for cls_idx, count in zip(unique, counts):
            print(f"  {self.class_names[cls_idx]}: {count} ({count/len(self)*100:.1f}%)")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image
        image = self.images[idx]
        
        # Convert to tensor
        img = torch.from_numpy(image).float()
        
        # HWC -> CHW
        if img.ndim == 3 and img.shape[2] == 3:
            img = img.permute(2, 0, 1)
        
        # Normalize to [0, 1]
        if img.max() > 1.0:
            img = img / 255.0
        
        # Apply transform
        if self.transform:
            img = self.transform(img)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return img, label