# Distributed ML Training Platform

**The big goal:** a platform where you upload a dataset + training code,
click "Start Training," and it automatically splits the work across
multiple machines/pods on a Kubernetes cluster - like a mini version of
what real ML companies use, not just "train one model on one computer."

**Where we are right now:** before building the website, the backend
server, and Kubernetes, we first proved that the two hardest, riskiest
pieces actually work:

1. **Real distributed training** - one model trained by multiple
   worker processes at once, genuinely staying in sync with each other
   (not just running the same script 4 times separately).
2. **Shared storage** - a central place to upload datasets/checkpoints
   that any machine can read from or write to over the network.

Everything below is how to run and verify that proof for yourself.
**No Kubernetes, Go backend, or website exists yet** - see the bottom of
this file for what's still ahead.

---

## Folder guide - what each file actually does

```
DML/
  README.md                    <- you are here
  ml-runner/
    model.py                    A small AI model (CNN) used for test training runs
    data.py                     Where training data comes from (fake data by default, or real CIFAR-10)
    storage.py                  Save/load checkpoints - works with local disk OR shared network storage
    train.py                    THE MAIN FILE - runs the actual distributed training
    dataset_shard.py            Splits a dataset so each worker downloads only ITS slice, not the whole thing
    make_demo_dataset.py        Uploads 100 sample files to storage, for testing dataset_shard.py
    dataset_api.py              A basic website/API to upload files - proves "upload button" works
    test_storage.py             A script that checks storage upload/download works correctly
    run_local.py                Windows-only backup plan (see "Known issues" below) - not important
    Dockerfile                  Recipe for building the container that actually runs training
    requirements.txt            List of Python packages needed
```

**If you only remember one file: `train.py`.** That's the actual
distributed-training engine. Everything else supports it or tests it.

---

## Before you start: what you need installed

- **Docker Desktop** - training runs inside a Linux container, not directly
  on Windows (see "Known issues" for why). Just needs to be installed and
  startable.
- **Python 3.12+** on your Windows machine itself, with the packages in
  `ml-runner/requirements.txt` installed (`pip install -r
  ml-runner/requirements.txt`). This is only needed for the small helper
  scripts (`dataset_api.py`, `test_storage.py`, `make_demo_dataset.py`) that
  run directly on Windows - the actual training always runs inside Docker.

---

## Quick Start - copy-paste, in order

Open a terminal in the `ml-runner` folder for all of this.

### Step 1 - Start Docker Desktop

Open Docker Desktop normally (Start Menu), and wait until it says it's
running. Everything else needs this.

### Step 2 - Start the storage server (SeaweedFS)

This is your "central storage" - the one place datasets and checkpoints
actually live.

```bash
docker volume create seaweed-data      # only needed the very first time ever

docker run -d --name seaweedfs \
  -p 8333:8333 -p 9333:9333 -p 8080:8080 -p 8888:8888 \
  -v seaweed-data:/data \
  chrislusf/seaweedfs server -s3 -dir=/data
```

> If you're on Windows using Git Bash and this fails with a weird path
> error, put `MSYS_NO_PATHCONV=1` right before `docker run` - Git Bash
> sometimes mangles the `/data` path.

Leave this running in the background. If you ever restart your computer
and the container is gone, just re-run the same command above - your
actual data is safe inside `seaweed-data` and comes right back.

### Step 3 - Build the training container (only needed once, or after code changes to requirements)

```bash
docker build -t ml-runner:dev .
```

Takes a few minutes the first time (downloading PyTorch). After that, it's
cached and instant unless you change `requirements.txt`.

### Step 4 - Run a real distributed training job

This is the main event. It starts **4 separate worker processes**, all
training **one shared model together**, and shows proof that they're
actually syncing up with each other:

```bash
docker run --rm -v "$(pwd):/workspace" -w /workspace ml-runner:dev \
  bash -c "torchrun --nproc_per_node=4 train.py --epochs 3 --run-id demo4"
```

**What to look for in the output:** each of the 4 workers (`rank 0`
through `rank 3`) prints a different `avg_loss` number (because each one
trained on different data), but they all print the **exact same
`param_checksum`** every round. That matching number is the proof - it
means all 4 workers really merged their learning into one shared model,
not 4 separate untouched ones.

(On Windows Git Bash, put `MSYS_NO_PATHCONV=1` before this `docker run`
too, same reason as Step 2.)

### Step 5 - Prove it can resume after a crash

Run the exact same command again, but with a bigger `--epochs` number:

```bash
docker run --rm -v "$(pwd):/workspace" -w /workspace ml-runner:dev \
  bash -c "torchrun --nproc_per_node=4 train.py --epochs 5 --run-id demo4"
```

You should see a line like `resumed from checkpoint at epoch 2, continuing
at epoch 3` - meaning it picked up where it left off instead of starting
over from scratch. This matters because Kubernetes will restart crashed
training pods automatically, and we don't want to lose all the progress
when that happens.

### Step 6 - When you're done, shut everything down

```bash
docker stop seaweedfs
```

Then just close Docker Desktop normally. Nothing is lost - your data sits
safely in the `seaweed-data` volume and comes back next time you start
SeaweedFS again with the same command from Step 2.

---

## Other things you can try

### Save checkpoints to shared storage instead of local disk

Same training command, but checkpoints go into SeaweedFS over the network
instead of a local folder - this is closer to how it'll really work once
Kubernetes pods are involved:

```bash
docker run --rm -v "$(pwd):/workspace" -w /workspace ml-runner:dev \
  bash -c "torchrun --nproc_per_node=4 train.py --epochs 3 --run-id s3demo \
           --storage s3 --s3-endpoint http://host.docker.internal:8333"
```

### Try the "upload dataset in a browser" demo

This runs directly on Windows, no Docker needed:

```bash
python -m uvicorn dataset_api:app --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000** in your browser. Pick a file, click
upload, and it lands in SeaweedFS - proving the real "click a button to
upload a dataset" flow works, not just a Python script talking to storage
internally.

### Try "each worker downloads only its own slice of the dataset"

First upload 100 sample files:
```bash
python make_demo_dataset.py
```

Then run training in "sharded" mode - each of the 4 workers will download
only ~25 files (its own slice), never the other 75:
```bash
docker run --rm -v "$(pwd):/workspace" -w /workspace ml-runner:dev \
  bash -c "torchrun --nproc_per_node=4 train.py --epochs 2 --run-id shard-demo \
           --data sharded --dataset-bucket datasets --dataset-prefix demo100 \
           --dataset-local-dir /tmp/shard_data --s3-endpoint http://host.docker.internal:8333"
```
Watch the log lines like `[rank 0] downloaded 25 files (its own shard
only)` - that's proof each worker really only fetched its own piece. (Note:
this is a specialized alternative, not the default - see "Known issues"
below for why the normal mode above is usually the better choice.)

### Quick sanity check that storage is working at all

```bash
python test_storage.py
```
Uploads a test file, downloads it back, and checks the bytes match exactly.
Good first thing to run if something else seems broken.

---

## Known issues & decisions we had to make along the way

- **Native Windows PyTorch can't do distributed training at all** - this
  is a real bug/limitation in the Windows version of PyTorch, not a mistake
  in this code. Even a single training process fails with `unsupported
  gloo device`. That's why every real training run happens inside a Linux
  Docker container instead of running Python directly on Windows.
  (`run_local.py` was an attempted workaround before we discovered Docker
  was the cleaner fix - it's not needed anymore.)

- **MinIO (the storage system originally planned) is discontinued.** Its
  own server told us directly: *"the open-source MinIO Server... are
  archived and no longer maintained."* We switched to **SeaweedFS**
  instead - still free, still open-source, still actively maintained, and
  it speaks the exact same "S3" storage language MinIO used, so nothing
  else about the design had to change.

- **The "each worker downloads only its slice" mode trains slightly
  worse** than the normal mode, because a worker only ever sees its fixed
  25 files instead of getting reshuffled against the full dataset every
  round. It's a real technique used for huge datasets that can't fit on
  one machine - but our dataset is small, so the normal full-copy mode
  (Step 4) is the better default. The sharded mode is there to demonstrate
  we understand the technique, not because it's what we'd actually use day
  to day.

- **The browser upload demo (`dataset_api.py`) loads the whole file into
  memory before uploading it** - fine for small test files, not fine for
  huge datasets. This is a known limitation of this quick prototype; the
  real backend (built in Go later) will stream uploads properly instead.

---

## What's NOT built yet

- No Kubernetes cluster
- No Go backend / API server
- No website/frontend
- No monitoring dashboards (Prometheus/Grafana/TensorBoard)
- No user accounts / login
- No database for job history

This README only covers proving the hard ML + storage pieces work. The
full plan for everything above is in the project plan document.
