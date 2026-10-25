"""Tests for the user-project contract (project_loader.py). No Docker, S3 or
distributed setup needed:

    python test_project_loader.py
"""
import os
import sys
import tempfile
import textwrap
import unittest
import zipfile

import project_loader as pl

HERE = os.path.dirname(os.path.abspath(__file__))

GOOD = """
import torch, torch.nn as nn
from torch.utils.data import Dataset

class DS(Dataset):
    def __len__(self): return 10
    def __getitem__(self, i): return torch.ones(4) * i, i % 3

def build_model(): return nn.Linear(4, 3)
def get_dataset(data_dir): return DS()
"""


class Case(unittest.TestCase):
    def script(self, src):
        d = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        path = os.path.join(d, "project.py")
        with open(path, "w") as f:
            f.write(textwrap.dedent(src))
        return path

    def assertContract(self, src, fragment):
        with self.assertRaises(pl.ContractError) as cm:
            pl.load_project(self.script(src), None)
        self.assertIn(fragment, str(cm.exception))


class ContractTests(Case):
    def test_valid_project(self):
        p = pl.load_project(self.script(GOOD), None)
        self.assertEqual(len(p.dataset), 10)
        opt = p.build_optimizer(p.model, 0.1)  # default optimizer
        self.assertEqual(type(opt).__name__, "SGD")

    def test_custom_optimizer_and_loss(self):
        src = GOOD + """
def build_optimizer(model, lr): return torch.optim.Adam(model.parameters(), lr=lr)
def build_loss(): return nn.CrossEntropyLoss(label_smoothing=0.1)
"""
        p = pl.load_project(self.script(src), None)
        self.assertEqual(type(p.build_optimizer(p.model, 0.1)).__name__, "Adam")
        self.assertEqual(p.criterion.label_smoothing, 0.1)

    def test_build_model_can_take_the_dataset(self):
        src = GOOD.replace("def build_model(): return nn.Linear(4, 3)",
                           "def build_model(dataset): return nn.Linear(4, len(set(int(dataset[i][1]) for i in range(len(dataset)))))")
        p = pl.load_project(self.script(src), None)
        self.assertEqual(p.model.out_features, 3)  # derived from the data, not hard-coded

    def test_missing_functions(self):
        self.assertContract("def build_model(): pass", "get_dataset()")
        self.assertContract("x = 1", "build_model() and get_dataset()")

    def test_syntax_error(self):
        self.assertContract("def build_model(:\n", "syntax error")

    def test_import_error(self):
        self.assertContract("import definitely_not_installed_pkg\n", "failed while being imported")

    def test_model_wrong_type(self):
        self.assertContract(GOOD.replace("return nn.Linear(4, 3)", "return 42"), "must return a torch.nn.Module")

    def test_empty_dataset(self):
        self.assertContract(GOOD.replace("return 10", "return 0"), "empty dataset")

    def test_dataset_wrong_type(self):
        self.assertContract(GOOD.replace("return DS()", "return 5"), "map-style torch Dataset")

    def test_model_data_mismatch_has_helpful_message(self):
        # model expects 8 features, data has 4
        self.assertContract(GOOD.replace("nn.Linear(4, 3)", "nn.Linear(8, 3)"), "input shape (4,)")

    def test_item_not_a_pair(self):
        self.assertContract(GOOD.replace("return torch.ones(4) * i, i % 3", "return torch.ones(4) * i"),
                            "(input, target) pair")

    def test_user_exception_is_wrapped(self):
        self.assertContract(GOOD.replace("return DS()", "raise KeyError('boom')"), "get_dataset() raised KeyError")

    def test_bad_optimizer(self):
        # the optimizer is built later (after DDP wraps the model), so the check is lazy
        p = pl.load_project(self.script(GOOD + "\ndef build_optimizer(model, lr): return 1\n"), None)
        with self.assertRaises(pl.ContractError) as cm:
            p.build_optimizer(p.model, 0.1)
        self.assertIn("torch.optim.Optimizer", str(cm.exception))


class ExampleTests(Case):
    def test_csv_example_end_to_end_without_s3(self):
        sys.path.insert(0, os.path.join(HERE, "examples"))
        try:
            import make_blobs_csv
        finally:
            sys.path.pop(0)
        data_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(data_dir, ignore_errors=True))
        make_blobs_csv.main(os.path.join(data_dir, "blobs.csv"), per_class=20)
        p = pl.load_project(os.path.join(HERE, "examples", "csv_classifier.py"), data_dir)
        self.assertEqual(len(p.dataset), 60)

    def test_csv_example_requires_dataset(self):
        with self.assertRaises(pl.ContractError) as cm:
            pl.load_project(os.path.join(HERE, "examples", "csv_classifier.py"), None)
        self.assertIn("needs a CSV dataset", str(cm.exception))


class FakeS3:
    """Just enough of a boto3 client for download_dataset/list_keys."""

    def __init__(self, objects):
        self.objects = objects  # key -> bytes

    def get_paginator(self, _):
        outer = self

        class P:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k} for k in outer.objects if k.startswith(Prefix)]}

        return P()

    def download_file(self, bucket, key, dest):
        with open(dest, "wb") as f:
            f.write(self.objects[key])


def make_zip(files):
    d = tempfile.mkdtemp()
    path = os.path.join(d, "z.zip")
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    with open(path, "rb") as f:
        return f.read()


class StagingTests(Case):
    def tmp(self):
        d = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        return d

    def test_plain_file_download(self):
        s3 = FakeS3({"abc/": b"", "abc/data.csv": b"1,2\n", "other/x": b"no"})
        local = self.tmp()
        data_dir = pl.stage_dataset(s3, "datasets", "abc/", local, True, lambda: None)
        self.assertEqual(open(os.path.join(data_dir, "data.csv")).read(), "1,2\n")
        self.assertFalse(os.path.exists(os.path.join(data_dir, "x")))

    def test_zip_is_unpacked_and_removed(self):
        s3 = FakeS3({"abc/imgs.zip": make_zip({"cat/1.txt": "a", "dog/2.txt": "b"})})
        data_dir = pl.stage_dataset(s3, "datasets", "abc/", self.tmp(), True, lambda: None)
        self.assertTrue(os.path.exists(os.path.join(data_dir, "cat", "1.txt")))
        self.assertTrue(os.path.exists(os.path.join(data_dir, "dog", "2.txt")))
        self.assertFalse(os.path.exists(os.path.join(data_dir, "imgs.zip")))

    def test_zip_slip_is_rejected(self):
        s3 = FakeS3({"abc/evil.zip": make_zip({"../escape.txt": "x"})})
        local = self.tmp()
        with self.assertRaises(pl.ContractError):
            pl.stage_dataset(s3, "datasets", "abc/", local, True, lambda: None)
        self.assertFalse(os.path.exists(os.path.join(local, "escape.txt")))

    def test_empty_dataset(self):
        with self.assertRaises(pl.ContractError):
            pl.stage_dataset(FakeS3({}), "datasets", "abc/", self.tmp(), True, lambda: None)

    def test_follower_waits_then_uses_leader_files(self):
        s3 = FakeS3({"abc/data.csv": b"x"})
        local = self.tmp()
        calls = []
        # leader downloads, then both reach the barrier
        pl.stage_dataset(s3, "datasets", "abc/", local, True, lambda: calls.append("leader"))
        data_dir = pl.stage_dataset(None, "datasets", "abc/", local, False, lambda: calls.append("follower"))
        self.assertEqual(calls, ["leader", "follower"])
        self.assertTrue(os.path.exists(os.path.join(data_dir, "data.csv")))

    def test_leader_failure_reaches_barrier_and_follower_fails_too(self):
        local = self.tmp()
        reached = []
        with self.assertRaises(pl.ContractError):
            pl.stage_dataset(FakeS3({}), "datasets", "abc/", local, True, lambda: reached.append(1))
        self.assertEqual(reached, [1])  # barrier reached despite the failure
        with self.assertRaises(pl.ContractError):
            pl.stage_dataset(None, "datasets", "abc/", local, False, lambda: None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
