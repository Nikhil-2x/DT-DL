"""Plug-in contract for user-supplied training projects (--data custom).

A user uploads ONE Python file. The platform owns everything distributed
(torchrun, DDP, DistributedSampler, checkpoints, resume, logging); the file
only says WHAT to train and on WHAT data.

Required functions in the user's file:

    def build_model() -> torch.nn.Module
    def build_model(dataset) -> torch.nn.Module
        Return the (un-wrapped) model. Do not wrap it in DDP or move it to a
        device; the platform does that. If you declare a parameter, you get
        the dataset from get_dataset(), so the model can size itself from the
        data (e.g. number of classes: len(dataset.classes)).

    def get_dataset(data_dir: str | None) -> torch.utils.data.Dataset
        Return the TRAINING dataset. Each item must be an (input, target)
        pair. `data_dir` is the folder holding the dataset the user attached
        to the job (a .zip upload is unpacked into it), or None if the job
        has no dataset (e.g. the data is generated or downloaded in code).
        Do NOT shard the data yourself: each rank gets its own slice via
        DistributedSampler.

Optional functions:

    def build_optimizer(model, lr) -> torch.optim.Optimizer
        Default: SGD(lr=lr, momentum=0.9).

    def build_loss() -> callable(outputs, targets) -> scalar tensor
        Default: torch.nn.CrossEntropyLoss().

Batch size, learning rate and epochs come from the job (CLI flags), not from
the file.
"""
import importlib.util
import inspect
import os
import shutil
import tempfile
import traceback
import zipfile
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

REQUIRED_FUNCTIONS = ("build_model", "get_dataset")


class ContractError(Exception):
    """The user's project does not follow the contract. The message is meant
    to be read by the user in the job logs."""


@dataclass
class UserProject:
    module: object
    model: nn.Module
    dataset: Dataset
    criterion: Callable
    build_optimizer: Callable  # (model, lr) -> Optimizer


# ---------------------------------------------------------------- loading


def load_module(path: str):
    """Import the user's file. Import-time errors become ContractError so the
    job log shows a clear message instead of a bare traceback."""
    spec = importlib.util.spec_from_file_location("user_project", path)
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot load {path} as a Python module")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except SyntaxError as e:
        raise ContractError(f"syntax error in your script: {e.msg} (line {e.lineno})") from e
    except Exception as e:
        traceback.print_exc()
        raise ContractError(f"your script failed while being imported: {type(e).__name__}: {e}") from e
    return module


def check_functions(module) -> None:
    missing = [n for n in REQUIRED_FUNCTIONS if not callable(getattr(module, n, None))]
    if missing:
        raise ContractError(
            "your script must define " + " and ".join(f"{n}()" for n in missing)
            + " (see the contract in project_loader.py or GET /api/v1/scripts/template)"
        )


def _call(module, name: str, *args):
    try:
        return getattr(module, name)(*args)
    except ContractError:
        raise
    except Exception as e:
        traceback.print_exc()
        raise ContractError(f"{name}() raised {type(e).__name__}: {e}") from e


def make_model(module, dataset: Optional[Dataset] = None) -> nn.Module:
    wants_dataset = any(
        p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        for p in inspect.signature(module.build_model).parameters.values()
    )
    model = _call(module, "build_model", dataset) if wants_dataset else _call(module, "build_model")
    if not isinstance(model, nn.Module):
        raise ContractError(f"build_model() must return a torch.nn.Module, got {type(model).__name__}")
    if not any(p.requires_grad for p in model.parameters()):
        raise ContractError("build_model() returned a model with no trainable parameters")
    return model


def make_dataset(module, data_dir: Optional[str]) -> Dataset:
    ds = _call(module, "get_dataset", data_dir)
    if not (hasattr(ds, "__len__") and hasattr(ds, "__getitem__")):
        raise ContractError(
            f"get_dataset() must return a map-style torch Dataset (with __len__ and __getitem__), "
            f"got {type(ds).__name__}"
        )
    if len(ds) == 0:
        raise ContractError("get_dataset() returned an empty dataset")
    return ds


def make_criterion(module) -> Callable:
    if not hasattr(module, "build_loss"):
        return nn.CrossEntropyLoss()
    criterion = _call(module, "build_loss")
    if not callable(criterion):
        raise ContractError("build_loss() must return a callable (outputs, targets) -> loss")
    return criterion


def make_optimizer_factory(module) -> Callable:
    fn = getattr(module, "build_optimizer", None)
    if fn is None:
        return lambda model, lr: torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    def factory(model, lr):
        opt = _call(module, "build_optimizer", model, lr)
        if not isinstance(opt, torch.optim.Optimizer):
            raise ContractError(f"build_optimizer() must return a torch.optim.Optimizer, got {type(opt).__name__}")
        return opt

    return factory


def smoke_test(model: nn.Module, dataset: Dataset, criterion: Callable) -> None:
    """Run one forward pass + loss on two real samples BEFORE DDP starts, so a
    mismatch between the model, the data and the loss fails fast with a clear
    message (and on every rank identically) rather than deep inside DDP."""
    first = None
    try:
        first = dataset[0]
        if not (isinstance(first, (tuple, list)) and len(first) == 2):
            raise ContractError("each dataset item must be an (input, target) pair")
        batch = default_collate([dataset[0], dataset[min(1, len(dataset) - 1)]])
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                outputs = model(batch[0])
                loss = criterion(outputs, batch[1])
        finally:
            model.train(was_training)
    except ContractError:
        raise
    except Exception as e:
        traceback.print_exc()
        hint = ""
        if isinstance(first, (tuple, list)) and len(first) == 2:
            hint = f" A sample has input shape {_shape(first[0])} and target shape {_shape(first[1])}."
        raise ContractError(f"model/data/loss do not fit together: {type(e).__name__}: {e}.{hint}") from e
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        raise ContractError("the loss must be a single scalar tensor")
    if not torch.isfinite(loss):
        raise ContractError(f"the loss is {loss.item()} on the first batch; check your data and model")


def _shape(x) -> str:
    return str(tuple(x.shape)) if hasattr(x, "shape") else type(x).__name__


def load_project(script_path: str, data_dir: Optional[str]) -> UserProject:
    """Import the script, validate the contract and build model/data/loss."""
    module = load_module(script_path)
    check_functions(module)
    dataset = make_dataset(module, data_dir)  # first: build_model(dataset) may size itself from it
    model = make_model(module, dataset)
    criterion = make_criterion(module)
    smoke_test(model, dataset, criterion)
    return UserProject(module, model, dataset, criterion, make_optimizer_factory(module))


# --------------------------------------------------------- fetching files


def fetch_script(client, bucket: str, key: str, dest_dir: str) -> str:
    """Download the user's script from object storage. Each process uses its
    own dest_dir, so concurrent ranks never write the same file."""
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, "project.py")
    client.download_file(bucket, key, path)
    return path


def _safe_join(root: str, rel: str) -> str:
    dest = os.path.abspath(os.path.join(root, rel))
    if os.path.commonpath([os.path.abspath(root), dest]) != os.path.abspath(root):
        raise ContractError(f"refusing to write outside the data folder: {rel!r}")
    return dest


def download_dataset(client, bucket: str, prefix: str, data_dir: str) -> List[str]:
    """Download EVERY object under `prefix` into data_dir (keeping paths
    relative to the prefix) and unpack .zip files in place. Returns the
    downloaded object keys."""
    from dataset_shard import list_keys  # local import: keeps this module light

    keys = list_keys(client, bucket, prefix)
    if not keys:
        raise ContractError(f"dataset is empty: nothing under s3://{bucket}/{prefix}")
    os.makedirs(data_dir, exist_ok=True)
    for key in keys:
        rel = key[len(prefix):] if key.startswith(prefix) else key.rsplit("/", 1)[-1]
        if not rel or rel.endswith("/"):
            continue
        dest = _safe_join(data_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        client.download_file(bucket, key, dest)
        if dest.lower().endswith(".zip"):
            _unzip(dest, data_dir)
    return keys


def _unzip(zip_path: str, data_dir: str) -> None:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.infolist():
                _safe_join(data_dir, member.filename)  # reject ../ and absolute paths
            zf.extractall(data_dir)
    except zipfile.BadZipFile as e:
        raise ContractError(f"{os.path.basename(zip_path)} is not a valid zip file") from e
    os.remove(zip_path)


def stage_dataset(
    client,
    bucket: str,
    prefix: str,
    local_dir: str,
    is_local_leader: bool,
    barrier: Callable[[], None],
) -> str:
    """Make the dataset available at <local_dir>/data on this machine.

    Only the local leader (LOCAL_RANK 0) downloads; the other ranks on the
    same machine wait at `barrier` and then read the same folder. The barrier
    is always reached, even if the download failed, so no rank is left
    hanging; everyone then raises the failure instead.
    """
    data_dir = os.path.join(local_dir, "data")
    marker = os.path.join(local_dir, ".ready")
    failure: Optional[Exception] = None
    if is_local_leader:
        try:
            shutil.rmtree(local_dir, ignore_errors=True)
            download_dataset(client, bucket, prefix, data_dir)
            with open(marker, "w"):
                pass
        except Exception as e:  # noqa: BLE001 - re-raised after the barrier
            failure = e
    barrier()
    if failure is not None:
        raise failure
    if not os.path.exists(marker):
        raise ContractError("dataset download failed on this machine's leader process")
    return data_dir


def default_script_dir(rank: int) -> str:
    return os.path.join(tempfile.gettempdir(), "dtdl_user_code", f"rank{rank}")
