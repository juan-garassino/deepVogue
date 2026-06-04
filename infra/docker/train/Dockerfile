FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg build-essential ninja-build && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-train.txt requirements-orchestration.txt ./
RUN pip install -r requirements.txt -r requirements-train.txt -r requirements-orchestration.txt
COPY . .
RUN pip install -e .
CMD ["python", "deepVogue/train.py", "--help"]
