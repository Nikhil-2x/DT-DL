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
| REST API (health, datasets, jobs, logs) | Done |
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

### Job lifecycle

```
PENDING -> QUEUED -> RUNNING -> COMPLETED
                        |  \--> FAILED
                        \-----> STOPPING -> STOPPED
```

### Job request fields

`name` and `epochs` are required. Optional: `workers` (default 1),
`dataset_id`, `data_mode` (`synthetic`, `cifar10`, `sharded`), `gpus`,
`cpu_cores`, `memory_mb`. Unknown fields are rejected. There is deliberately
**no** field for image, command or script, so the API cannot run arbitrary
code.

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
    storage/                ObjectStorage interface, S3 (SeaweedFS) + in-memory fake
    runner/                 JobRunner interface, mock runner, docker runner
    repository/             SQLite implementations of the dataset/job stores
  migrations/               SQL schema (embedded into the binary)
```

## Known limitations (things the existing Python code imposes)

- **Uploaded data is not really used by training yet.** `train.py` only reads
  data via `--data sharded`, which treats every object under a prefix as one
  sample and ignores file contents. A real data loader is future Python work.
- **Sharded mode needs one file per worker**, otherwise a worker with no data
  would hang PyTorch's synchronisation. Since each upload is one file, jobs
  with a dataset are limited to `workers=1` for now.
- **"Workers" are processes in one container** (`torchrun --nproc_per_node`),
  on CPU with gloo. They are not separate pods or GPUs yet.
- **S3 credentials are passed to `train.py` as command-line arguments**
  (that is how `train.py` takes them), so they are visible via
  `docker inspect`. Fine for local dev; a follow-up should let `train.py`
  read environment variables or a Kubernetes Secret.
- One backend instance only (state changes are guarded by an in-process lock).
