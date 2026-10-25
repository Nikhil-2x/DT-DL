FROM python:3.12-slim

WORKDIR /workspace

# Slow or flaky networks: wait longer per read and retry more before giving up.
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir boto3

COPY . /workspace
