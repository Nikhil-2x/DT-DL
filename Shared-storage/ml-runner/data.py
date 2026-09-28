"""Dataset loading for the training script.

Default is a synthetic CIFAR-10-shaped dataset so the pipeline runs
offline and deterministically. Pass --data cifar10 to download the real
dataset via torchvision instead (needs internet on first run).
"""
import os
import re

import torch
from torch.utils.data import Dataset


class SyntheticImageDataset(Dataset):
    """Deterministic random 3x32x32 images with 10 classes."""

    def __init__(self, size: int = 2000, num_classes: int = 10, seed: int = 42):
        g = torch.Generator().manual_seed(seed)
        self.images = torch.randn(size, 3, 32, 32, generator=g)
        self.labels = torch.randint(0, num_classes, (size,), generator=g)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class ShardedFileDataset(Dataset):
    """Wraps a list of local files (this rank's own pre-downloaded shard -
    see dataset_shard.py) as a dataset. Each file's index (parsed from its
    filename, e.g. img_0007.bin -> 7) deterministically seeds a synthetic
    3x32x32 image + label, so results are reproducible without needing the
    file's actual byte content to mean anything."""

    def __init__(self, local_paths, num_classes: int = 10):
        self.local_paths = local_paths
        self.num_classes = num_classes

    def __len__(self):
        return len(self.local_paths)

    def __getitem__(self, idx):
        path = self.local_paths[idx]
        match = re.search(r"(\d+)", os.path.basename(path))
        file_index = int(match.group(1)) if match else idx
        g = torch.Generator().manual_seed(file_index)
        image = torch.randn(3, 32, 32, generator=g)
        label = torch.randint(0, self.num_classes, (1,), generator=g).item()
        return image, label


def get_dataset(kind: str = "synthetic", root: str = "./data"):
    if kind == "cifar10":
        from torchvision import datasets, transforms

        transform = transforms.Compose([transforms.ToTensor()])
        return datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    return SyntheticImageDataset()
