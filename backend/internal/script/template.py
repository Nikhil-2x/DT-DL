"""Starting point for a training project on the platform.

You write ONLY what to train and on what data. The platform handles the rest:
torchrun, DistributedDataParallel, splitting the data across workers,
checkpoints, resuming after a crash, and logging.

Contract
--------
Required:
  build_model()          -> torch.nn.Module
      (or build_model(dataset) to size the model from your data, e.g. the
       number of classes: len(dataset.classes))
  get_dataset(data_dir)  -> torch Dataset yielding (input, target) pairs
Optional:
  build_optimizer(model, lr) -> torch.optim.Optimizer   (default: SGD, momentum 0.9)
  build_loss()               -> callable(outputs, targets) (default: CrossEntropyLoss)

data_dir is the folder with the dataset attached to the job (a .zip upload is
unpacked into it), or None when the job has no dataset. Do not shard the data
yourself, and do not wrap the model in DDP. Epochs, batch size and learning
rate are job settings, not part of this file.

This template runs as-is on random data, so you can check that the pipeline
works before plugging in your own model and data.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset

NUM_CLASSES = 10


class RandomImages(Dataset):
    """Placeholder data: random 3x32x32 images with random labels."""

    def __init__(self, size=2000):
        g = torch.Generator().manual_seed(42)
        self.images = torch.randn(size, 3, 32, 32, generator=g)
        self.labels = torch.randint(0, NUM_CLASSES, (size,), generator=g)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.images[i], self.labels[i]


def build_model():
    # TODO: replace with your model
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, NUM_CLASSES),
    )


def get_dataset(data_dir):
    # TODO: load your data from data_dir (None means "no dataset attached")
    return RandomImages()
