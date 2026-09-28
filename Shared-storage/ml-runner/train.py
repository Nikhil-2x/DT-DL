"""DDP training core. Two ways to launch this:

  - On Linux (real deployment target, Phase 2 containers):
        torchrun --nproc_per_node=4 train.py --epochs 5 --run-id demo
    torchrun sets RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORT and
    this file's main() picks them up from the environment.

  - On this Windows dev machine: use run_local.py instead. torchrun's
    launcher has a bug on this PyTorch/Windows build where it hardcodes
    TCPStore creation in a way that ignores USE_LIBUV and always fails
    with "PyTorch was built without libuv support" even when the env
    var is set. run_local.py sidesteps torchrun entirely and uses
    torch.multiprocessing.spawn + dist.init_process_group's plain env://
    path instead, which handles this correctly on Windows.
"""
import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data import ShardedFileDataset, get_dataset
from dataset_shard import fetch_rank_shard
from model import SmallCNN
from storage import LocalCheckpointStore, S3CheckpointStore


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--data", choices=["synthetic", "cifar10", "sharded"], default="synthetic",
                         help="'sharded' downloads only this rank's slice of a dataset from S3 (see dataset_shard.py)")
    parser.add_argument("--dataset-bucket", default="datasets")
    parser.add_argument("--dataset-prefix", default="demo100")
    parser.add_argument("--dataset-local-dir", default="./shard_data",
                         help="Where this rank's downloaded shard files get saved")
    parser.add_argument("--run-id", default="demo")
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--nproc", type=int, default=1,
                         help="Number of local worker processes (run_local.py only; ignored by torchrun)")
    parser.add_argument("--storage", choices=["local", "s3"], default="local",
                         help="Where checkpoints are saved/loaded from")
    parser.add_argument("--s3-endpoint", default="http://127.0.0.1:8333")
    parser.add_argument("--s3-bucket", default="checkpoints")
    parser.add_argument("--s3-access-key", default="any")
    parser.add_argument("--s3-secret-key", default="any")
    return parser.parse_args()


def build_checkpoint_store(args):
    if args.storage == "s3":
        return S3CheckpointStore(
            bucket=args.s3_bucket,
            prefix=args.run_id,
            endpoint_url=args.s3_endpoint,
            access_key=args.s3_access_key,
            secret_key=args.s3_secret_key,
        )
    return LocalCheckpointStore(os.path.join(args.checkpoint_dir, args.run_id))


def params_checksum(model: nn.Module) -> float:
    """Sum of all parameter values - used only to prove weights are in
    sync across ranks after each DDP allreduce + optimizer step."""
    return sum(p.detach().sum().item() for p in model.parameters())


def run(rank: int, world_size: int, args) -> None:
    """One worker's full training loop. Called directly by torchrun's
    per-process main() below, or by run_local.py via mp.spawn."""
    torch.manual_seed(0)  # same initial weights on every rank before DDP broadcast
    model = SmallCNN()
    ddp_model = DDP(model)

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    if args.data == "sharded":
        # Physical sharding: this rank downloads ONLY its own slice of the
        # dataset from S3 - it never has the other ranks' files at all,
        # unlike synthetic/cifar10 below where every rank gets the full
        # dataset and DistributedSampler picks a slice in-memory.
        local_dir = os.path.join(args.dataset_local_dir, f"rank{rank}")
        local_paths = fetch_rank_shard(
            endpoint_url=args.s3_endpoint,
            access_key=args.s3_access_key,
            secret_key=args.s3_secret_key,
            bucket=args.dataset_bucket,
            prefix=args.dataset_prefix,
            rank=rank,
            world_size=world_size,
            local_dir=local_dir,
        )
        print(f"[rank {rank}] downloaded {len(local_paths)} files (its own shard only): "
              f"{[os.path.basename(p) for p in local_paths]}", flush=True)
        dataset = ShardedFileDataset(local_paths)
        sampler = None
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        dataset = get_dataset(args.data)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=0)
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    store = build_checkpoint_store(args)
    start_epoch = 0
    ckpt = store.load_latest()
    if ckpt is not None:
        ddp_model.module.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        if rank == 0:
            print(f"[rank {rank}] resumed from checkpoint at epoch {ckpt['epoch']}, "
                  f"continuing at epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        ddp_model.train()

        epoch_start = time.time()
        total_loss, batches = 0.0, 0
        for images, labels in loader:
            optimizer.zero_grad()
            outputs = ddp_model(images)
            loss = criterion(outputs, labels)
            loss.backward()  # DDP allreduces gradients across ranks here
            optimizer.step()
            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(batches, 1)
        elapsed = time.time() - epoch_start
        checksum = params_checksum(ddp_model.module)
        print(f"[rank {rank}] epoch {epoch} avg_loss={avg_loss:.4f} "
              f"param_checksum={checksum:.4f} time={elapsed:.2f}s", flush=True)

        if rank == 0:
            store.save(
                {
                    "model_state": ddp_model.module.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                },
                epoch,
            )
        dist.barrier()  # make sure rank 0's checkpoint write finishes before anyone resumes

    dist.destroy_process_group()


def main():
    """torchrun entrypoint (Linux/Phase 2 containers): rank/world_size
    come from the environment torchrun sets up."""
    args = parse_args()
    dist.init_process_group(backend="gloo")
    run(dist.get_rank(), dist.get_world_size(), args)


if __name__ == "__main__":
    main()
