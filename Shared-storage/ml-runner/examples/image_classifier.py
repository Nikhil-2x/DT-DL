"""Example user project: an image classifier for ANY folder-per-class dataset.

Upload a .zip whose folders are the class names:

    my_images.zip
      cats/  001.jpg 002.png ...
      dogs/  001.jpg ...
      birds/ ...

The number of classes and their names come from the folders, so the same
script works for 2 classes or 200. Images of any size are resized to
IMAGE_SIZE x IMAGE_SIZE. A single top-level folder inside the zip
(my_images/cats/...) is handled too.

Try it with the generated test set:  python examples/make_shapes_dataset.py
"""
import os

import torch.nn as nn
from torchvision import datasets, transforms

IMAGE_SIZE = 32


def _dataset_root(data_dir):
    """Descend through wrapper folders (e.g. the zip's top-level folder)."""
    root = data_dir
    while True:
        entries = [e for e in os.listdir(root) if not e.startswith((".", "__"))]
        dirs = [e for e in entries if os.path.isdir(os.path.join(root, e))]
        if len(entries) == 1 and len(dirs) == 1:
            root = os.path.join(root, dirs[0])
        else:
            return root


def get_dataset(data_dir):
    if data_dir is None:
        raise ValueError("attach a zip of images (one folder per class) to the job")
    tfm = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return datasets.ImageFolder(_dataset_root(data_dir), transform=tfm)


def build_model(dataset):
    # `dataset` is what get_dataset() returned: the output layer is sized
    # from the real number of classes found in the data.
    num_classes = len(dataset.classes)

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2))

    return nn.Sequential(
        block(3, 32), block(32, 64), block(64, 128),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(128, num_classes),
    )
