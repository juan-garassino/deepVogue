# syntax=docker/dockerfile:1.7
# Devel base: ships nvcc + CUDA headers so SG3 custom ops can JIT-compile at
# runtime (bias_act, upfirdn2d). The runtime image is ~2-3GB smaller but the
# inference server raises on first model load without nvcc.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg build-essential ninja-build && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-serve.txt requirements-orchestration.txt ./
RUN pip install -r requirements.txt -r requirements-serve.txt -r requirements-orchestration.txt

COPY . .
RUN pip install -e .

# Warm the custom-ops import path so any source-level errors surface at build
# time. The real .so cache is built on first GPU call at runtime — CPU-only
# build environments can't pre-compile the kernels.
RUN python -c "from deepVogue.pytorch_utils.ops import bias_act, upfirdn2d; print('ops import OK')"

ENV PORT=8080 \
    DV_FASTAPI_HOST=0.0.0.0 \
    DV_FASTAPI_PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request, sys; \
        sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health', timeout=3).status==200 else 1)"

CMD ["uvicorn", "deepVogue.serve.api:app", "--host", "0.0.0.0", "--port", "8080"]
