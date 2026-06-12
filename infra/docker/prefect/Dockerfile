FROM prefecthq/prefect:2-python3.11

ARG TARGET=server
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt requirements-orchestration.txt ./
RUN pip install -r requirements.txt -r requirements-orchestration.txt

COPY . .
RUN pip install -e .

EXPOSE 4200

# server target runs the API; worker target starts a process pool worker.
# Switch with: docker build --build-arg TARGET=worker
ENV TARGET=${TARGET}
CMD ["sh", "-c", "if [ \"$TARGET\" = \"worker\" ]; then \
    prefect worker start --pool default-process-pool; \
  else \
    prefect server start --host 0.0.0.0 --port 4200; \
  fi"]
