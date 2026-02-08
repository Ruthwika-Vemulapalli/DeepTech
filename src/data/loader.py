"""
Data loading utilities for 8-class dataset
"""
import torch
from torch.utils.data import DataLoader, Subset
from .dataset import Defect8ClassDataset, NUM_CLASSES
from .augmentation import get_train_transforms, get_val_transforms


def create_dataloaders(data_dir, batch_size=32, val_split=0.15, test_split=0.15, num_workers=4):
    """Create train, val, test dataloaders with stratified split."""
    train_ds = Defect8ClassDataset(data_dir=data_dir, transform=get_train_transforms())
    val_ds = Defect8ClassDataset(data_dir=data_dir, transform=get_val_transforms())
    test_ds = Defect8ClassDataset(data_dir=data_dir, transform=get_val_transforms())

    n = len(train_ds)
    gen = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=gen).tolist()
    test_size = int(n * test_split)
    val_size = int(n * val_split)
    train_size = n - val_size - test_size

    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = perm[train_size + val_size:]

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        Subset(test_ds, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    return train_loader, val_loader, test_loader
