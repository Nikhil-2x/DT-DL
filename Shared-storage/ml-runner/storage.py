"""Checkpoint storage abstraction.

Two implementations of the same CheckpointStore interface:
  - LocalCheckpointStore: local disk (used in earlier standalone tests).
  - S3CheckpointStore: any S3-compatible object store (SeaweedFS locally,
    could equally be real AWS S3 or another S3-compatible server later).
Both implement save()/load_latest(), so train.py never has to know or
care which one is actually being used.
"""
import glob
import io
import os
from abc import ABC, abstractmethod
from typing import Optional

import torch


class CheckpointStore(ABC):
    @abstractmethod
    def save(self, state: dict, epoch: int) -> None:
        ...

    @abstractmethod
    def load_latest(self) -> Optional[dict]:
        ...


class LocalCheckpointStore(CheckpointStore):
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)

    def save(self, state: dict, epoch: int) -> None:
        path = os.path.join(self.run_dir, f"epoch_{epoch}.pt")
        tmp_path = path + ".tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)  # atomic on both POSIX and Windows

    def load_latest(self) -> Optional[dict]:
        files = glob.glob(os.path.join(self.run_dir, "epoch_*.pt"))
        if not files:
            return None

        def epoch_of(path: str) -> int:
            name = os.path.basename(path)
            return int(name[len("epoch_"):-len(".pt")])

        latest = max(files, key=epoch_of)
        return torch.load(latest, map_location="cpu")


class S3CheckpointStore(CheckpointStore):
    """Stores checkpoints as objects in an S3-compatible bucket, under
    key prefix `<prefix>/epoch_N.pt`. Creates the bucket if it doesn't
    exist yet."""

    def __init__(
        self,
        bucket: str,
        prefix: str,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
    ):
        import boto3
        from botocore.exceptions import ClientError

        self._ClientError = ClientError
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except self._ClientError:
            self.client.create_bucket(Bucket=self.bucket)

    def save(self, state: dict, epoch: int) -> None:
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        key = f"{self.prefix}/epoch_{epoch}.pt"
        self.client.upload_fileobj(buffer, self.bucket, key)

    def load_latest(self) -> Optional[dict]:
        resp = self.client.list_objects_v2(Bucket=self.bucket, Prefix=self.prefix + "/")
        contents = resp.get("Contents", [])
        if not contents:
            return None

        def epoch_of(obj: dict) -> int:
            name = obj["Key"].rsplit("/", 1)[-1]
            return int(name[len("epoch_"):-len(".pt")])

        latest = max(contents, key=epoch_of)
        buffer = io.BytesIO()
        self.client.download_fileobj(self.bucket, latest["Key"], buffer)
        buffer.seek(0)
        return torch.load(buffer, map_location="cpu")
