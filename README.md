# DT-DL — SeaweedFS Shared Storage Integration

## What Has Been Implemented

This stage implements the basic shared-storage and training flow for DT-DL.

- Replaced the planned MinIO storage with **SeaweedFS**.
- SeaweedFS is running through Docker.
- SeaweedFS provides an **S3-compatible endpoint**.
- The Go backend connects to SeaweedFS using the S3 API.
- `datasets` bucket is used to store uploaded datasets.
- `checkpoints` bucket is used to store training checkpoints.
- The backend can upload and download datasets.
- The backend can create and start training jobs.
- The Docker runner starts the `ml-runner:dev` training container.
- The training container can access SeaweedFS.
- The training container downloads the required dataset.
- Training runs successfully using PyTorch.
- Training checkpoints are uploaded back to SeaweedFS.
- Job status is updated to `COMPLETED` after successful training.

## Current Tested Flow

```text
Go Backend
    |
    | Upload Dataset
    v
SeaweedFS
    |
    | Dataset
    v
Docker Training Container
    |
    | Download Dataset
    | Run Training
    v
SeaweedFS
    |
    | Checkpoint
    v
checkpoints/<job-id>/epoch_0.pt
```

The complete flow has been tested on a single machine using Docker containers to simulate the training machine.

## Requirements

The following are required:

- Docker
- Go
- Python 3
- SeaweedFS Docker image
- `ml-runner:dev` Docker image

## 1. Start SeaweedFS

Start the SeaweedFS container:

```bash
docker run -d \
  --name seaweedfs \
  -p 8333:8333 \
  -p 9333:9333 \
  -p 8080:8080 \
  -p 8888:8888 \
  -v seaweed-data:/data \
  chrislusf/seaweedfs \
  server -s3 -dir=/data
```

Check that it is running:

```bash
docker ps
```

The S3 endpoint is:

```text
http://127.0.0.1:8333
```

## 2. Build the ML Training Image

Go to the ML runner directory:

```bash
cd Shared-storage/ml-runner
```

Build the image:

```bash
docker build -t ml-runner:dev .
```

Check the image:

```bash
docker images | grep ml-runner
```

## 3. Run the Go Backend

Go to the backend:

```bash
cd backend
```

Run the backend with Docker as the runner:

```bash
SERVER_PORT=8081 \
RUNNER=docker \
S3_ENDPOINT=http://127.0.0.1:8333 \
S3_ACCESS_KEY=dtdl-access \
S3_SECRET_KEY=dtdl-secret \
RUNNER_S3_ENDPOINT=http://host.docker.internal:8333 \
go run ./cmd/server
```

The backend will run on:

```text
http://127.0.0.1:8081
```

Check the health endpoint:

```bash
curl -s http://127.0.0.1:8081/api/v1/health
```

Expected result:

```json
{
  "status": "ok"
}
```

## 4. Upload a Test Dataset

Create a small CSV:

```bash
printf 'x,y\n1,2\n3,4\n' > /tmp/flow-test.csv
```

Upload it:

```bash
curl -s -X POST \
  -F 'name=flow-test' \
  -F 'file=@/tmp/flow-test.csv' \
  http://127.0.0.1:8081/api/v1/datasets
```

The response contains the dataset ID.

Example:

```text
938f5c03010fde68d985cd61
```

## 5. Create a Training Job

Replace `<DATASET_ID>` with the ID returned above:

```bash
curl -s -X POST \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "docker-flow-test",
    "dataset_id": "<DATASET_ID>",
    "epochs": 1,
    "workers": 1,
    "data_mode": "sharded",
    "gpus": 0,
    "cpu_cores": 2,
    "memory_mb": 1024
  }' \
  http://127.0.0.1:8081/api/v1/jobs
```

Save the returned job ID.

## 6. Start the Training Job

Replace `<JOB_ID>` with the job ID:

```bash
curl -s -X POST \
  http://127.0.0.1:8081/api/v1/jobs/<JOB_ID>/start
```

Check the running Docker container:

```bash
docker ps -a --filter name=dtdl-job-<JOB_ID>
```

View the training logs:

```bash
docker logs dtdl-job-<JOB_ID>
```

A successful run produces output similar to:

```text
[rank 0] downloaded 1 files (its own shard only): ['flow-test.csv']
[rank 0] epoch 0 avg_loss=2.3928 param_checksum=1.5496 time=0.01s
```

## 7. Check the Job Status

```bash
curl -s \
  http://127.0.0.1:8081/api/v1/jobs/<JOB_ID>
```

After successful training, the status should be:

```text
COMPLETED
```

The output location will point to:

```text
s3://checkpoints/<JOB_ID>/
```

## 8. Verify the Checkpoint in SeaweedFS

Open the SeaweedFS shell:

```bash
docker exec -it seaweedfs weed shell -master=127.0.0.1:9333
```

Check the checkpoint:

```text
fs.ls /buckets/checkpoints/<JOB_ID>
```

Expected:

```text
epoch_0.pt
```

This confirms that the training container downloaded the dataset from shared storage and uploaded the training checkpoint back to shared storage.

## Current Status

The following flow has been successfully tested:

```text
Dataset Upload
      ↓
Go Backend
      ↓
SeaweedFS
      ↓
Docker Training Container
      ↓
Dataset Download
      ↓
PyTorch Training
      ↓
Checkpoint Upload
      ↓
SeaweedFS
      ↓
Job COMPLETED
```

## Current Limitation

The current test uses one worker on one machine.

For sharded datasets, the backend currently restricts the job to one worker because the uploaded dataset is currently stored as a single object.

The next stage is to test multiple workers on the same machine and then move the worker containers to Kubernetes.

## Future Architecture

The final system is planned to use:

```text
              Go Backend
                   |
             Job Management
                   |
             Kubernetes
          /        |        \
       Worker    Worker    Worker
         |         |         |
         +---------+---------+
                   |
              SeaweedFS
             /          \
        Datasets      Checkpoints
```

Multiple training machines or Kubernetes pods will use the same SeaweedFS storage so that datasets and checkpoints can be shared between workers.
