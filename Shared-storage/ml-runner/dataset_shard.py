"""Pre-sharded dataset access: instead of every pod downloading the FULL
dataset and then logically picking a slice in-memory (what
DistributedSampler does), each rank lists the dataset's files in S3 and
downloads ONLY the files assigned to it - so a rank never has more than
its own 1/world_size share of the data on local disk at all.

Trade-off worth knowing: because the split now happens physically at
download time, a given rank sees the SAME slice of files every epoch
(no cross-epoch reshuffling across the whole dataset, since a rank
literally never has the other ranks' files available to shuffle into).
For very large datasets this is normal and desirable (nobody could fit
the full dataset locally anyway); for smaller datasets, the "copy full
dataset, shuffle every epoch" approach in data.py trains slightly better.
"""
import os
from typing import List

import boto3


def list_keys(client, bucket: str, prefix: str) -> List[str]:
    keys = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return sorted(keys)


def shard_for_rank(keys: List[str], rank: int, world_size: int) -> List[str]:
    """Interleaved split: rank i gets keys[i], keys[i+world_size], ...
    Deterministic, no coordination needed between ranks, and every key
    goes to exactly one rank."""
    return keys[rank::world_size]


def download_shard(client, bucket: str, keys: List[str], local_dir: str) -> List[str]:
    os.makedirs(local_dir, exist_ok=True)
    local_paths = []
    for key in keys:
        filename = key.rsplit("/", 1)[-1]
        local_path = os.path.join(local_dir, filename)
        client.download_file(bucket, key, local_path)
        local_paths.append(local_path)
    return local_paths


def fetch_rank_shard(
    endpoint_url: str,
    access_key: str,
    secret_key: str,
    bucket: str,
    prefix: str,
    rank: int,
    world_size: int,
    local_dir: str,
) -> List[str]:
    """Full flow: list all dataset files, pick this rank's slice, download
    only those. Returns local file paths for this rank's shard."""
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    all_keys = list_keys(client, bucket, prefix)
    my_keys = shard_for_rank(all_keys, rank, world_size)
    local_paths = download_shard(client, bucket, my_keys, local_dir)
    return local_paths
