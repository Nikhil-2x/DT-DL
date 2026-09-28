FROM python:3.12-slim

WORKDIR /workspace

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir boto3

COPY . /workspace
