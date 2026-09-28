"""Minimal dataset upload/download API, backed by SeaweedFS (S3-compatible).

This is a throwaway prototype to prove the real "user uploads a file
through a browser/API -> it lands in shared object storage -> anyone
else can read it back" flow end to end. The real platform's backend
(Go, per the plan) replaces this later with the same underlying idea:
stream the upload straight into the object store, don't buffer whole
datasets in application memory/disk.

Run:
    uvicorn dataset_api:app --reload --port 8000
Then open http://127.0.0.1:8000 in a browser.
"""
import io

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

ENDPOINT_URL = "http://127.0.0.1:8333"
ACCESS_KEY = "any"
SECRET_KEY = "any"
DATASET_BUCKET = "datasets"

app = FastAPI(title="Dataset Storage Prototype")

client = boto3.client(
    "s3",
    endpoint_url=ENDPOINT_URL,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
)


@app.on_event("startup")
def ensure_bucket():
    try:
        client.head_bucket(Bucket=DATASET_BUCKET)
    except ClientError:
        client.create_bucket(Bucket=DATASET_BUCKET)


@app.get("/", response_class=HTMLResponse)
def upload_page():
    return """
    <html>
      <body style="font-family: sans-serif; max-width: 480px; margin: 40px auto;">
        <h2>Dataset Upload</h2>
        <form action="/datasets/upload" method="post" enctype="multipart/form-data">
          <input type="file" name="file" required />
          <button type="submit">Upload</button>
        </form>
        <h3>Existing datasets</h3>
        <ul id="list"></ul>
        <script>
          fetch('/datasets').then(r => r.json()).then(data => {
            const ul = document.getElementById('list');
            data.datasets.forEach(d => {
              const li = document.createElement('li');
              li.innerHTML = `<a href="/datasets/${encodeURIComponent(d.key)}/download">${d.key}</a> (${d.size} bytes)`;
              ul.appendChild(li);
            });
          });
        </script>
      </body>
    </html>
    """


@app.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    contents = await file.read()
    client.upload_fileobj(io.BytesIO(contents), DATASET_BUCKET, file.filename)
    return {"filename": file.filename, "size": len(contents), "bucket": DATASET_BUCKET}


@app.get("/datasets")
def list_datasets():
    resp = client.list_objects_v2(Bucket=DATASET_BUCKET)
    datasets = [
        {"key": obj["Key"], "size": obj["Size"]}
        for obj in resp.get("Contents", [])
    ]
    return {"datasets": datasets}


@app.get("/datasets/{key}/download")
def download_dataset(key: str):
    try:
        obj = client.get_object(Bucket=DATASET_BUCKET, Key=key)
    except ClientError:
        raise HTTPException(status_code=404, detail=f"dataset '{key}' not found")
    return StreamingResponse(
        obj["Body"].iter_chunks(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{key}"'},
    )
