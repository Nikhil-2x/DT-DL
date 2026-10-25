"""Example user project: a tabular classifier trained on an uploaded CSV.

Upload this file as a script, upload a CSV (see make_blobs_csv.py) as a
dataset, then create a job with both. The platform trains it with DDP on any
number of workers; this file only describes the model and the data.

CSV format: optional header row, N numeric feature columns, and the integer
class label (0..NUM_CLASSES-1) in the LAST column.
"""
import csv
import glob
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset

NUM_FEATURES = 4
NUM_CLASSES = 3


class CsvDataset(Dataset):
    def __init__(self, data_dir):
        paths = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
        if not paths:
            raise FileNotFoundError(f"no .csv file found in the dataset folder {data_dir}")
        features, labels = [], []
        for path in paths:
            with open(path, newline="") as f:
                for row in csv.reader(f):
                    try:
                        values = [float(v) for v in row]
                    except ValueError:
                        continue  # header or blank line
                    if len(values) != NUM_FEATURES + 1:
                        raise ValueError(f"{path}: expected {NUM_FEATURES + 1} columns, got {len(values)}")
                    features.append(values[:-1])
                    labels.append(int(values[-1]))
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def build_model():
    return nn.Sequential(
        nn.Linear(NUM_FEATURES, 32),
        nn.ReLU(),
        nn.Linear(32, NUM_CLASSES),
    )


def get_dataset(data_dir):
    if data_dir is None:
        raise ValueError("this project needs a CSV dataset attached to the job")
    return CsvDataset(data_dir)


# Optional: override the defaults (SGD with momentum, cross-entropy loss).
def build_optimizer(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr)
