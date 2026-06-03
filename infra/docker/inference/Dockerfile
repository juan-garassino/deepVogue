# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-serve.txt requirements-orchestration.txt ./
RUN pip install -r requirements.txt -r requirements-serve.txt -r requirements-orchestration.txt

COPY . .
RUN pip install -e .

ENV PORT=8080 \
    DV_FASTAPI_HOST=0.0.0.0 \
    DV_FASTAPI_PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request, sys; \
        sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health', timeout=3).status==200 else 1)"

CMD ["uvicorn", "deepVogue.serve.api:app", "--host", "0.0.0.0", "--port", "8080"]
