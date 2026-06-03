FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev build-essential && rm -rf /var/lib/apt/lists/*

RUN pip install "mlflow[extras]==2.16.2" "psycopg2-binary>=2.9" \
    "google-cloud-storage>=2.18" "boto3>=1.34"

EXPOSE 5000

CMD ["sh", "-c", "mlflow server \
    --host 0.0.0.0 --port 5000 \
    --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
    --artifacts-destination ${MLFLOW_ARTIFACTS_DESTINATION} \
    --serve-artifacts"]
