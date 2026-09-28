"""Uploads 100 individual dummy 'image' files to SeaweedFS, so we can
demonstrate pre-sharded downloading (each rank fetching only its slice)
instead of one big zip every pod downloads in full.

Run: python make_demo_dataset.py
"""
import io

import boto3
from botocore.exceptions import ClientError

ENDPOINT_URL = "http://127.0.0.1:8333"
ACCESS_KEY = "any"
SECRET_KEY = "any"
BUCKET = "datasets"
PREFIX = "demo100"
NUM_FILES = 100


def main():
    client = boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )
    try:
        client.head_bucket(Bucket=BUCKET)
    except ClientError:
        client.create_bucket(Bucket=BUCKET)

    for i in range(NUM_FILES):
        key = f"{PREFIX}/img_{i:04d}.bin"
        # content just needs to be unique and deterministic per index
        content = f"dummy image content for index {i}".encode()
        client.upload_fileobj(io.BytesIO(content), BUCKET, key)

    print(f"Uploaded {NUM_FILES} files to s3://{BUCKET}/{PREFIX}/")


if __name__ == "__main__":
    main()
