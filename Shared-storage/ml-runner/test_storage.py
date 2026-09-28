"""End-to-end read/write test against the S3-compatible store (SeaweedFS
locally). Proves the exact flow discussed: one 'machine' uploads a
dataset, any other 'machine' can read it back, and checkpoints can be
saved/loaded the same way training pods will do it later.

Run: python test_storage.py
"""
import io

import boto3
from botocore.exceptions import ClientError

from storage import S3CheckpointStore

ENDPOINT_URL = "http://127.0.0.1:8333"
ACCESS_KEY = "any"       # SeaweedFS default S3 gateway has no auth enforced
SECRET_KEY = "any"
DATASET_BUCKET = "datasets"


def ensure_bucket(client, bucket: str) -> None:
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError:
        client.create_bucket(Bucket=bucket)


def test_dataset_upload_download():
    print("== Dataset upload/download test ==")
    client = boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )
    ensure_bucket(client, DATASET_BUCKET)

    # Simulate "PC-A uploads a dataset"
    fake_dataset_content = b"fake cifar10 zip bytes - just testing upload/download plumbing"
    key = "cifar10/train.zip"
    client.upload_fileobj(io.BytesIO(fake_dataset_content), DATASET_BUCKET, key)
    print(f"uploaded {len(fake_dataset_content)} bytes to s3://{DATASET_BUCKET}/{key}")

    # Simulate "PC-B (or a training pod) downloads the same dataset"
    buffer = io.BytesIO()
    client.download_fileobj(DATASET_BUCKET, key, buffer)
    downloaded = buffer.getvalue()

    assert downloaded == fake_dataset_content, "downloaded content does not match what was uploaded!"
    print("downloaded content matches exactly - dataset read/write works")

    # List to confirm it shows up like a real dataset listing would
    listing = client.list_objects_v2(Bucket=DATASET_BUCKET)
    keys = [obj["Key"] for obj in listing.get("Contents", [])]
    print("bucket contents:", keys)


def test_checkpoint_roundtrip():
    print("\n== Checkpoint save/load roundtrip test (via S3CheckpointStore) ==")
    store = S3CheckpointStore(
        bucket="checkpoints",
        prefix="run-storage-test",
        endpoint_url=ENDPOINT_URL,
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
    )

    fake_state = {"model_state": {"weight": [1, 2, 3]}, "optimizer_state": {}, "epoch": 0}
    store.save(fake_state, epoch=0)
    store.save({**fake_state, "epoch": 1}, epoch=1)
    print("saved checkpoints for epoch 0 and epoch 1")

    loaded = store.load_latest()
    assert loaded is not None, "load_latest() returned nothing!"
    assert loaded["epoch"] == 1, f"expected latest epoch=1, got {loaded['epoch']}"
    assert loaded["model_state"] == fake_state["model_state"]
    print(f"load_latest() correctly returned epoch {loaded['epoch']} with matching contents")


if __name__ == "__main__":
    test_dataset_upload_download()
    test_checkpoint_roundtrip()
    print("\nAll storage tests passed.")
