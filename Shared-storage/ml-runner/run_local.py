"""Local multi-process DDP launcher for Windows dev machines.

torchrun's launcher hardcodes a TCPStore creation path on Windows that
ignores USE_LIBUV and always fails ("PyTorch was built without libuv
support"), even with the env var set correctly. This script sidesteps
torchrun's elastic launcher entirely and uses torch.multiprocessing.spawn
plus dist.init_process_group's plain env:// rendezvous, which resolves
use_libuv correctly on Windows on its own.

Usage:
    python run_local.py --nproc 4 --epochs 5 --run-id demo

On Linux (Phase 2 containers), use `torchrun` directly against train.py
instead - this launcher is a Windows-only workaround.
"""
import os

import torch.distributed as dist
import torch.multiprocessing as mp

from train import parse_args, run

MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29500"


def worker(local_rank: int, world_size: int, args) -> None:
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # Gloo needs an explicit network interface on this machine; without it,
    # it tries to resolve this PC's hostname and picks up a bogus address
    # from Docker Desktop's local DNS override (kubernetes.docker.internal).
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "Loopback Pseudo-Interface 1")

    dist.init_process_group(backend="gloo", rank=local_rank, world_size=world_size)
    run(local_rank, world_size, args)


def main():
    args = parse_args()
    mp.spawn(worker, args=(args.nproc, args), nprocs=args.nproc, join=True)


if __name__ == "__main__":
    main()
