"""Example user project: CIFAR-10 (the classic 10-class image benchmark).

No dataset upload needed: get_dataset() downloads CIFAR-10 itself, so create
the job with a script only. The training container needs internet access for
the first download (~170 MB).

SUBSET keeps a CPU-only demo quick; set it to None to train on all 50,000 images.
"""
import torch.nn as nn
from torch.utils.data import Subset
from torchvision import datasets, transforms

SUBSET = 5000

_MEAN, _STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def get_dataset(data_dir):
    tfm = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    # Every worker calls this; the folder is per-container, so they may race to
    # write it. download=True is safe when the files already exist.
    full = datasets.CIFAR10(root="/tmp/cifar10", train=True, download=True, transform=tfm)
    return Subset(full, range(SUBSET)) if SUBSET else full


def build_model():
    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2))

    return nn.Sequential(
        block(3, 32), block(32, 64), block(64, 128),
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 10),
    )
