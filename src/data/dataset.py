"""
8-class Dataset: Defect_1..Defect_6 + Clean + Other (min 8 classes for Phase 1)
"""
import os
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import torch

# Class order: Defect_1, Defect_2, Defect_3, Defect_4, Defect_5, Defect_6, Clean, Other
CLASS_NAMES = [
    'bridge',      # Defect_1
    'cmp_scratch', # Defect_2
    'ler',         # Defect_3
    'opens',       # Defect_4
    'crack',      # Defect_5 (via_crack)
    'short',       # Defect_6
    'clean',       # Clean
    'other',       # Other
]
NUM_CLASSES = 8


class Defect8ClassDataset(Dataset):
    """Dataset for 8-class semiconductor defect classification."""

    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Root with subdirs: bridge, cmp_scratch, ler, opens, crack, short, clean, other
            transform: Optional transform
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []  # (path, class_idx)
        self.class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
        self.idx_to_class = {i: name for i, name in enumerate(CLASS_NAMES)}
        self._load_samples()

    def _load_samples(self):
        for class_name in CLASS_NAMES:
            folder = self.data_dir / class_name
            if not folder.exists():
                continue
            idx = self.class_to_idx[class_name]
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                for img_file in sorted(folder.glob(ext)):
                    self.samples.append((str(img_file), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
