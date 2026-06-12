FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg build-essential ninja-build curl ca-certificates apt-transport-https gnupg \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
       | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
       > /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-train.txt requirements-orchestration.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-train.txt -r requirements-orchestration.txt

COPY . .
RUN pip install --no-cache-dir -e .

RUN chmod +x infra/docker/train/entrypoint.sh

ENTRYPOINT ["infra/docker/train/entrypoint.sh"]
