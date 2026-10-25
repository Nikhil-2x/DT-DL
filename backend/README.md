# Go Backend (Control Plane)

The Go backend is the "manager" of the Distributed ML Training Platform. It
does **not** train models. It:

- accepts dataset uploads and stores them in SeaweedFS (S3 API),
- records datasets and training jobs in a SQLite database,
- starts / stops / monitors training jobs through a `JobRunner`,
- exposes all of this as a JSON REST API.

The PyTorch training code (`../Shared-storage/ml-runner`) is unchanged. The
backend just launches it, the same way you would by hand:

```
torchrun --nproc_per_node=4 train.py --epochs 3 --run-id <job-id> --storage s3 ...
```

## Status

| Piece | State |
|---|---|
| REST API (health, datasets, scripts, jobs, logs) | Done |
| User-supplied training projects (`build_model()` / `get_dataset()` script) | Done, see below |
| SeaweedFS storage via S3 API (streaming upload) | Done |
| SQLite persistence + migrations | Done |
| `mock` runner (simulated training) | Done |
| `docker` runner (real `ml-runner:dev` container) | Done, only unit-tested so far |
| Unit + API tests | Done |
| Dockerfile / docker-compose | Not yet |
| Kubernetes runner | Not yet (interface is ready) |
| Authentication, frontend | Not yet |

## Requirements

- Go 1.24+ (`go version`)
- SeaweedFS running with the S3 API on port 8333, exactly as in
  `../Shared-storage/README.md`:
  ```bash
  docker run -d --name seaweedfs -p 8333:8333 -p 9333:9333 -p 8080:8080 -p 8888:8888 \
    -v seaweed-data:/data chrislusf/seaweedfs server -s3 -dir=/data
  ```
- Docker + the `ml-runner:dev` image, **only** if you use `RUNNER=docker`.

## Run it

```bash
cd backend
go run ./cmd/server
```

The server listens on http://127.0.0.1:8080. By default it uses the **mock**
runner, so you can try the whole job lifecycle without Docker or PyTorch.

To run real training containers instead:

```bash
docker build -t ml-runner:dev ../Shared-storage/ml-runner
RUNNER=docker go run ./cmd/server
```

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `SERVER_PORT` | `8080` | HTTP port |
| `DATABASE_URL` | `sqlite:./data/backend.db` | SQLite file (`sqlite:<path>`) |
| `S3_ENDPOINT` | `http://127.0.0.1:8333` | SeaweedFS S3 endpoint |
| `S3_ACCESS_KEY` / `S3_SECRET_KEY` | `any` / `any` | S3 credentials (SeaweedFS started with `-s3` accepts any) |
| `S3_REGION` | `us-east-1` | S3 region |
| `DATASET_BUCKET` | `datasets` | Bucket for uploaded datasets |
| `CHECKPOINT_BUCKET` | `checkpoints` | Bucket where `train.py` saves checkpoints |
| `MAX_UPLOAD_BYTES` | `10737418240` (10 GiB) | Upload size limit |
| `RUNNER` | `mock` | `mock` or `docker` |
| `TRAINING_IMAGE` | `ml-runner:dev` | Image used by the docker runner |
| `RUNNER_S3_ENDPOINT` | `http://host.docker.internal:8333` | S3 endpoint as seen **from inside** training containers |
| `SCRIPT_BUCKET` | `scripts` | Bucket for uploaded project scripts |
| `MAX_SCRIPT_BYTES` | `1048576` (1 MiB) | Max size of a project script |
| `MAX_WORKERS` | `8` | Max workers per job |
| `MOCK_EPOCH_DURATION` | `2s` | How long one fake epoch takes (mock runner) |
| `RECONCILE_INTERVAL` | `2s` | How often job status is refreshed |
| `LOG_LEVEL` | `info` | `debug`, `info`, `warn`, `error` |

Never commit real credentials. Put them in your environment or in an
untracked `.env` file.

## Try the API

```bash
# health
curl localhost:8080/api/v1/health

# upload a dataset (streamed; put "name" BEFORE "file")
echo "a,b,c" > sample.csv
curl -F name=sample -F file=@sample.csv localhost:8080/api/v1/datasets

# list datasets / get one / delete one
curl localhost:8080/api/v1/datasets
curl localhost:8080/api/v1/datasets/<dataset-id>
curl -X DELETE localhost:8080/api/v1/datasets/<dataset-id>

# create a job (synthetic data, like the README demo)
curl -X POST localhost:8080/api/v1/jobs -H 'Content-Type: application/json' \
  -d '{"name":"cifar-demo","epochs":3,"workers":4}'

# start it, check it, read its logs, stop it
curl -X POST localhost:8080/api/v1/jobs/<job-id>/start
curl localhost:8080/api/v1/jobs/<job-id>
curl "localhost:8080/api/v1/jobs/<job-id>/logs?tail=50"
curl -X POST localhost:8080/api/v1/jobs/<job-id>/stop
```

Errors always look like:

```json
{"error": {"code": "JOB_NOT_FOUND", "message": "job not found"}}
```

## Bring your own model and data (project scripts)

Different projects need different models and datasets, so a job can train a
**user-supplied Python file** instead of the built-in CNN. The file only says
*what* to train; the platform adds distributed training (torchrun + DDP), data
splitting across workers, checkpoints, resume and logging.

The file must define two top-level functions (optional ones in brackets):

```python
def build_model():             # -> torch.nn.Module   (or build_model(dataset) to size it from the data)
def get_dataset(data_dir):     # -> torch Dataset yielding (input, target); data_dir is None if no dataset
[def build_optimizer(model, lr)]   # default SGD(momentum=0.9)
[def build_loss()]                 # default CrossEntropyLoss
```

```bash
# get a runnable starting point
curl localhost:8080/api/v1/scripts/template > project.py

# upload it (the file must be .py and define the two functions)
curl -F name=my-model -F file=@project.py localhost:8080/api/v1/scripts

# optional: upload data (a .zip is unpacked into data_dir; other files are copied as-is)
curl -F name=blobs -F file=@blobs.csv localhost:8080/api/v1/datasets

# train it on 4 workers
curl -X POST localhost:8080/api/v1/jobs -H 'Content-Type: application/json' \
  -d '{"name":"mine","epochs":5,"workers":4,"script_id":"<script-id>","dataset_id":"<dataset-id>"}'
curl -X POST localhost:8080/api/v1/jobs/<job-id>/start
```

Other script endpoints: `GET /scripts`, `GET /scripts/{id}`,
`GET /scripts/{id}/content`, `DELETE /scripts/{id}` (blocked while an unfinished
job uses it). Examples live in `../Shared-storage/ml-runner/examples/`:
`image_classifier.py` (any zip of images, one folder per class; try it with
`shapes.zip` from `make_shapes_dataset.py`), `cifar10_classifier.py` (downloads
CIFAR-10 itself, no dataset upload needed) and `csv_classifier.py` (tabular).

How it works: the script is stored in SeaweedFS as `scripts/<id>/project.py`.
For a `custom` job the backend starts the same `torchrun ... train.py`
container with `--data custom --script-key ...` (and `--attach-dataset` if a
dataset is chosen). Inside, rank 0 of each machine downloads the dataset, every
rank imports the script and checks the contract, runs one sample forward pass
to catch model/data/loss mismatches early, and then trains with DDP. A contract
problem is reported in the job logs as `PROJECT ERROR: ...` and the job FAILS.

**Security:** the script is arbitrary Python and runs inside the training
container, with the container's S3 credentials in its environment. Only
trusted users should be able to upload scripts until authentication and
sandboxing exist. The API still accepts no image or command from clients, and
the backend itself never executes the script.

### Job lifecycle

```
PENDING -> QUEUED -> RUNNING -> COMPLETED
                        |  \--> FAILED
                        \-----> STOPPING -> STOPPED
```

### Job request fields

`name` and `epochs` are required. Optional: `workers` (default 1),
`dataset_id`, `script_id` (a project script; implies `data_mode` `custom`),
`data_mode` (`synthetic`, `cifar10`, `sharded`, `custom`), `gpus`,
`cpu_cores`, `memory_mb`. Unknown fields are rejected. There is deliberately
**no** field for image or command; `script_id` can only reference a script
that was uploaded through the scripts API.

## Run the tests

```bash
cd backend
go test ./...          # all unit + API tests (no SeaweedFS or Docker needed)
go test -race ./...    # same, with the race detector
go vet ./...

# optional: check the S3 layer against a real SeaweedFS
S3_TEST_ENDPOINT=http://127.0.0.1:8333 go test ./internal/storage -run S3 -v
```

## Project layout

```
backend/
  cmd/server/main.go        entry point: config -> wiring -> HTTP server
  internal/
    config/                 environment-variable settings
    apperr/                 shared error type -> JSON error + HTTP status
    http/handlers/          request decoding + JSON responses (no business logic)
    http/middleware/        request ID, access log, panic recovery
    dataset/                dataset model + upload/list/delete service
    job/                    job model, state machine, start/stop/reconcile service
    script/                 user project scripts: upload/validate/list/delete + template
    storage/                ObjectStorage interface, S3 (SeaweedFS) + in-memory fake
    runner/                 JobRunner interface, mock runner, docker runner
    repository/             SQLite implementations of the dataset/job stores
  migrations/               SQL schema (embedded into the binary)
```

## Known limitations (things the existing Python code imposes)

- **Built-in modes ignore uploaded data contents.** `--data sharded` treats
  every object under a prefix as one sample. To really train on uploaded data,
  use a project script (`script_id`), whose `get_dataset(data_dir)` reads it.
- **Sharded mode needs one file per worker**, otherwise a worker with no data
  would hang PyTorch's synchronisation, so built-in sharded jobs are limited to
  `workers=1`. Project-script jobs have no such limit: each worker downloads
  the whole dataset and `DistributedSampler` gives it a different slice.
- **Script validation in the backend is only a quick check** (the two
  functions exist at top level). The authoritative check happens when the
  training container imports the script, so a broken script fails at job time.
- **"Workers" are processes in one container** (`torchrun --nproc_per_node`),
  on CPU with gloo. They are not separate pods or GPUs yet.
- **S3 credentials are passed to `train.py` as command-line arguments**
  (that is how `train.py` takes them), so they are visible via
  `docker inspect`. Fine for local dev; a follow-up should let `train.py`
  read environment variables or a Kubernetes Secret.
- One backend instance only (state changes are guarded by an in-process lock).
