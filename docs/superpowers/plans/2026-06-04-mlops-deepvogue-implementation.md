# deepVogue MLOps Stack — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap deepVogue's existing SG3-t pipeline with a production-shaped MLOps stack: a local docker-compose nano-mode (MinIO + Postgres + MLflow + Prefect + FastAPI) that mirrors a piece-by-piece GCP deployment, plus CI/CD with Slack notifications, while preserving Colab training and Drive persistence.

**Architecture:** Cross-cutting Python helpers (`clients.py`, `notifications/slack.py`, `tracking/mlflow_helpers.py`, `publish.py`) use `fsspec` so the same code reads/writes MinIO locally and GCS in prod. A single Prefect flow set (`prepare/train/publish/project/walk/eval` + composite pipelines) dispatches to pluggable backends — `local` is real for `prepare` / `publish` and mocked for `train/project/walk/eval` (Mac can't run SG3 CUDA ops); `colab` and `runpod` backends are scaffolded but raise `NotImplementedError` in v1. GitHub Actions builds Docker images per service, pushes to Artifact Registry, deploys Cloud Run, and posts Slack notifications. Drive is preserved as Colab's training scratch, demoted from "canonical store" to "the source `make publish` reads from."

**Tech Stack:** Python 3.11, Prefect 2.x, MLflow 2.x, FastAPI, fsspec (gcsfs + s3fs), Docker Compose, MinIO, Postgres 15, Cloud Run (incl. GPU L4), Cloud SQL, Artifact Registry, GitHub Actions with Workload Identity Federation, Slack incoming webhooks.

**Spec:** `docs/superpowers/specs/2026-06-03-mlops-deepvogue-design.md`

---

## File Structure

### New files (created by this plan)

| Path | Responsibility |
|---|---|
| `requirements-orchestration.txt` | Prefect/MLflow/fsspec/slack deps for the orchestration + tracking layer |
| `deepVogue/clients.py` | Single `get_artifact_fs()` factory; resolves `DV_ARTIFACT_BACKEND` + endpoint env vars into an `fsspec` filesystem. Also resolves relative `models.yaml` paths. |
| `deepVogue/notifications/__init__.py` | Package marker |
| `deepVogue/notifications/slack.py` | Thin wrapper: `notify_success`, `notify_failure`, `notify_event`. No-op if `SLACK_WEBHOOK_URL` unset. |
| `deepVogue/tracking/__init__.py` | Package marker |
| `deepVogue/tracking/mlflow_helpers.py` | `log_training_run(run_dir, dataset_name, **params)` — reads `metric-fid50k_full.jsonl`, logs to MLflow. Called from Colab. |
| `deepVogue/publish.py` | `publish_checkpoint(model_id, src_path)` — Drive→GCS upload, atomic `models.yaml` append, Slack notify. |
| `deepVogue/orchestration/__init__.py` | Package marker |
| `deepVogue/orchestration/flows.py` | All Prefect flows: `prepare_flow`, `train_flow`, `publish_flow`, `project_flow`, `walk_flow`, `eval_flow`, `pipeline_stills`, `pipeline_frames` |
| `deepVogue/orchestration/backends/__init__.py` | Package marker; `dispatch(backend, op, **kwargs)` registry |
| `deepVogue/orchestration/backends/local.py` | Nano-mode implementations: real `prepare`/`publish`, mock `train`/`project`/`walk`/`eval` |
| `deepVogue/orchestration/backends/colab.py` | v2 scaffold; raises `NotImplementedError` |
| `deepVogue/orchestration/backends/runpod.py` | v2 scaffold; raises `NotImplementedError` |
| `tests/test_clients.py` | fsspec resolution + relative-URI resolution |
| `tests/test_slack.py` | Block formatting, mocked POST, no-op behavior |
| `tests/test_mlflow_helpers.py` | Mocked MLflow client; reads existing FID jsonl |
| `tests/test_publish.py` | tmp dirs + memory fs; atomic models.yaml mutation |
| `tests/test_orchestration_backends.py` | Backend dispatch; local mock outputs valid shapes |
| `tests/test_nano_smoke.py` | docker-compose-up integration; full `pipeline_stills` against MinIO+local MLflow+mocked Slack |
| `tests/fixtures/stub_sg3_state_dict.pt` | Tiny structurally-valid SG3-t state dict (img_resolution=64) |
| `scripts/build_stub_state_dict.py` | One-shot generator for the fixture above |
| `infra/.env.example` | Documented env vars for local nano + prod overrides |
| `infra/docker-compose.yml` | postgres + minio (+ init) + mlflow + prefect + fastapi |
| `infra/docker/inference/Dockerfile` | PyTorch 2.4-CUDA 12.1 base + `pip install -e .` + serve deps |
| `infra/docker/mlflow/Dockerfile` | python:3.11-slim + mlflow[extras] + psycopg2 + gcs + s3 |
| `infra/docker/prefect/Dockerfile` | prefect base; multi-target (server, worker) via build arg |
| `infra/docker/train/Dockerfile` | v2 RunPod-ready training image (created but never built in v1 CI) |
| `infra/cloudrun/inference.service.yaml` | Cloud Run service spec: L4 GPU, scale-to-zero |
| `infra/cloudrun/mlflow.service.yaml` | Cloud Run service spec for MLflow |
| `infra/cloudrun/prefect-server.service.yaml` | Cloud Run service spec for Prefect server |
| `infra/cloudrun/prefect-worker.job.yaml` | Cloud Run job spec for Prefect worker |
| `infra/gcp/setup.sh` | Idempotent: APIs, buckets, AR repo, Cloud SQL, VPC connector, runtime SAs, WIF |
| `.github/workflows/test.yml` | pytest + black on push/PR |
| `.github/workflows/build-inference.yml` | build → AR → Cloud Run deploy → Slack |
| `.github/workflows/build-mlflow.yml` | same shape for MLflow |
| `.github/workflows/build-prefect.yml` | same shape for Prefect server+worker |
| `.github/workflows/build-train.yml` | manual-only training image build |
| `.github/workflows/nano-smoke.yml` | nightly cron + PR — docker-compose nano smoke |
| `Makefile` (modified) | New targets: `nano-up`, `nano-down`, `nano-smoke`, `publish`, `deploy-*`, `gcp-setup` |
| `deepVogue/serve/loader.py` (modified) | Accept absolute `s3://`/`gs://` URIs in `models.yaml` |
| `deepVogue/serve/registry.py` (modified) | Same — schema bump |
| `CLAUDE.md` (modified) | Add "MLOps stack" section |
| `README.md` (modified) | Add nano-mode quickstart |
| `../../DOCS.md` (modified) | deepVogue entry mention MLflow/Prefect/Cloud Run |

---

## Phase 0 — Bootstrap (10 min)

### Task 0.1: requirements-orchestration.txt

**Files:**
- Create: `requirements-orchestration.txt`

- [ ] **Step 1: Create the requirements file**

```
prefect>=2.20,<3.0
mlflow>=2.16,<3.0
fsspec>=2024.6.0
gcsfs>=2024.6.0
s3fs>=2024.6.0
google-cloud-storage>=2.18.0
boto3>=1.34.0
psycopg2-binary>=2.9.0
slack-sdk>=3.27.0
pytest-asyncio>=0.23.0
moto[s3]>=5.0.0
```

- [ ] **Step 2: Install locally**

Run: `pip install -r requirements-orchestration.txt`
Expected: all packages install without conflict.

- [ ] **Step 3: Commit**

```bash
git add requirements-orchestration.txt
git commit -m "deps: add orchestration requirements (prefect, mlflow, fsspec, slack)"
```

### Task 0.2: infra/ + tests/fixtures/ scaffolding

**Files:**
- Create: `infra/.gitkeep`, `infra/docker/.gitkeep`, `infra/cloudrun/.gitkeep`, `infra/gcp/.gitkeep`, `tests/fixtures/.gitkeep`, `.github/workflows/.gitkeep`

- [ ] **Step 1: Make directories and placeholders**

```bash
mkdir -p infra/docker/{inference,mlflow,prefect,train} infra/cloudrun infra/gcp tests/fixtures .github/workflows
touch infra/.gitkeep infra/docker/.gitkeep infra/cloudrun/.gitkeep infra/gcp/.gitkeep tests/fixtures/.gitkeep .github/workflows/.gitkeep
```

- [ ] **Step 2: Commit**

```bash
git add infra/ tests/fixtures/.gitkeep .github/workflows/.gitkeep
git commit -m "chore: scaffolding for infra/ and CI workflow dirs"
```

---

## Phase 1 — Cross-cutting libs (TDD) (2 hr)

### Task 1.1: `deepVogue/clients.py` — fsspec backend factory

**Files:**
- Create: `deepVogue/clients.py`
- Create: `tests/test_clients.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_clients.py`:
```python
import os
import pytest
from deepVogue.clients import get_artifact_fs, resolve_uri


def test_s3_backend_with_minio_endpoint(monkeypatch):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "s3")
    monkeypatch.setenv("DV_S3_ENDPOINT_URL", "http://localhost:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "minio")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "miniopass")
    fs = get_artifact_fs()
    assert fs.protocol == ("s3", "s3a")
    assert fs.client_kwargs["endpoint_url"] == "http://localhost:9000"


def test_gcs_backend(monkeypatch):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "gcs")
    monkeypatch.delenv("DV_S3_ENDPOINT_URL", raising=False)
    fs = get_artifact_fs()
    assert "gcs" in (fs.protocol if isinstance(fs.protocol, tuple) else (fs.protocol,)) or "gs" in (
        fs.protocol if isinstance(fs.protocol, tuple) else (fs.protocol,)
    )


def test_memory_backend_for_tests(monkeypatch):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "memory")
    fs = get_artifact_fs()
    assert fs.protocol == "memory"


def test_unknown_backend_raises(monkeypatch):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "wat")
    with pytest.raises(ValueError, match="unknown"):
        get_artifact_fs()


def test_resolve_uri_passthrough_absolute():
    assert resolve_uri("s3://b/k") == "s3://b/k"
    assert resolve_uri("gs://b/k") == "gs://b/k"


def test_resolve_uri_resolves_relative_to_models_root(monkeypatch):
    monkeypatch.setenv("DV_MODELS_ROOT", "/data/runs")
    assert resolve_uri("tarot/snap.pkl") == "/data/runs/tarot/snap.pkl"
```

- [ ] **Step 2: Run tests to confirm fail**

Run: `pytest tests/test_clients.py -v`
Expected: ImportError — `deepVogue.clients` does not exist.

- [ ] **Step 3: Implement `deepVogue/clients.py`**

```python
"""fsspec filesystem factory and URI resolution for deepVogue artifacts."""
from __future__ import annotations

import os
from typing import Any

import fsspec

_SUPPORTED = {"s3", "gcs", "memory", "file"}


def get_artifact_fs() -> Any:
    backend = os.environ.get("DV_ARTIFACT_BACKEND", "file").lower()
    if backend not in _SUPPORTED:
        raise ValueError(f"unknown DV_ARTIFACT_BACKEND={backend!r}; expected one of {_SUPPORTED}")
    if backend == "s3":
        endpoint = os.environ.get("DV_S3_ENDPOINT_URL")
        client_kwargs = {"endpoint_url": endpoint} if endpoint else {}
        return fsspec.filesystem("s3", client_kwargs=client_kwargs)
    if backend == "gcs":
        return fsspec.filesystem("gcs")
    if backend == "memory":
        return fsspec.filesystem("memory")
    return fsspec.filesystem("file")


def resolve_uri(uri: str) -> str:
    """Pass absolute URIs through; resolve relative paths against DV_MODELS_ROOT."""
    if "://" in uri or uri.startswith("/"):
        return uri
    root = os.environ.get("DV_MODELS_ROOT", "").rstrip("/")
    if not root:
        return uri
    return f"{root}/{uri}"
```

- [ ] **Step 4: Run tests to confirm pass**

Run: `pytest tests/test_clients.py -v`
Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add deepVogue/clients.py tests/test_clients.py
git commit -m "feat(clients): fsspec backend factory + URI resolution"
```

### Task 1.2: `deepVogue/notifications/slack.py`

**Files:**
- Create: `deepVogue/notifications/__init__.py`, `deepVogue/notifications/slack.py`
- Create: `tests/test_slack.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_slack.py`:
```python
from unittest.mock import patch, MagicMock
import pytest
from deepVogue.notifications import slack


def test_no_op_without_webhook(monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    # should not raise, should not call requests
    with patch("deepVogue.notifications.slack.requests") as r:
        slack.notify_success("ci", "build done", {"sha": "abc"})
        r.post.assert_not_called()


def test_notify_success_posts(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/X")
    resp = MagicMock(status_code=200)
    with patch("deepVogue.notifications.slack.requests") as r:
        r.post.return_value = resp
        slack.notify_success("ci", "build done", {"sha": "abc"})
        r.post.assert_called_once()
        body = r.post.call_args.kwargs["json"]
        assert "blocks" in body
        text = str(body)
        assert "ci" in text
        assert "build done" in text
        assert "abc" in text


def test_notify_failure_includes_exception_summary(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/X")
    with patch("deepVogue.notifications.slack.requests") as r:
        r.post.return_value = MagicMock(status_code=200)
        try:
            raise RuntimeError("boom")
        except RuntimeError as e:
            slack.notify_failure("ci", "build failed", exc=e)
        text = str(r.post.call_args.kwargs["json"])
        assert "RuntimeError" in text
        assert "boom" in text


def test_swallows_non_2xx(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/X")
    with patch("deepVogue.notifications.slack.requests") as r:
        r.post.return_value = MagicMock(status_code=500, text="err")
        # must not raise
        slack.notify_success("ci", "ok")
```

- [ ] **Step 2: Run tests — confirm fail**

Run: `pytest tests/test_slack.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `deepVogue/notifications/__init__.py`**

```python
from . import slack as slack  # noqa: F401
```

- [ ] **Step 4: Implement `deepVogue/notifications/slack.py`**

```python
"""Slack webhook notifier. Silent no-op if SLACK_WEBHOOK_URL unset."""
from __future__ import annotations

import logging
import os
import traceback
from typing import Mapping

import requests

log = logging.getLogger(__name__)


def _post(blocks: list) -> None:
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        return
    try:
        resp = requests.post(url, json={"blocks": blocks}, timeout=5)
        if resp.status_code // 100 != 2:
            log.warning("slack non-2xx: %s %s", resp.status_code, resp.text[:200])
    except requests.RequestException as e:
        log.warning("slack post failed: %s", e)


def _fields(meta: Mapping[str, str] | None) -> list:
    if not meta:
        return []
    return [
        {"type": "mrkdwn", "text": f"*{k}*\n{v}"} for k, v in list(meta.items())[:10]
    ]


def notify_event(channel_tag: str, level: str, headline: str, meta: Mapping[str, str] | None = None) -> None:
    emoji = {"success": ":white_check_mark:", "failure": ":x:", "info": ":information_source:"}.get(level, ":bell:")
    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": f"{emoji} *[{channel_tag}]* {headline}"}}
    ]
    fields = _fields(meta)
    if fields:
        blocks.append({"type": "section", "fields": fields})
    _post(blocks)


def notify_success(channel_tag: str, headline: str, meta: Mapping[str, str] | None = None) -> None:
    notify_event(channel_tag, "success", headline, meta)


def notify_failure(channel_tag: str, headline: str, meta: Mapping[str, str] | None = None, exc: BaseException | None = None) -> None:
    meta = dict(meta or {})
    if exc is not None:
        meta.setdefault("error", f"{type(exc).__name__}: {exc}")
        meta.setdefault("trace", "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-500:])
    notify_event(channel_tag, "failure", headline, meta)
```

- [ ] **Step 5: Run tests — confirm pass**

Run: `pytest tests/test_slack.py -v`
Expected: 4 PASSED.

- [ ] **Step 6: Commit**

```bash
git add deepVogue/notifications/ tests/test_slack.py
git commit -m "feat(notifications): slack webhook helpers with silent no-op"
```

### Task 1.3: `deepVogue/tracking/mlflow_helpers.py`

**Files:**
- Create: `deepVogue/tracking/__init__.py`, `deepVogue/tracking/mlflow_helpers.py`
- Create: `tests/test_mlflow_helpers.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_mlflow_helpers.py`:
```python
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from deepVogue.tracking.mlflow_helpers import log_training_run


def _make_run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "00000-tarot-stylegan3-t-gpus1-batch32-gamma2"
    run.mkdir()
    # FID jsonl
    fid = run / "metric-fid50k_full.jsonl"
    fid.write_text(
        json.dumps({"snapshot_pkl": "network-snapshot-000200.pkl", "results": {"fid50k_full": 142.3}}) + "\n" +
        json.dumps({"snapshot_pkl": "network-snapshot-000400.pkl", "results": {"fid50k_full": 98.6}}) + "\n"
    )
    (run / "network-snapshot-000200.pkl").write_bytes(b"\x00" * 16)
    (run / "network-snapshot-000400.pkl").write_bytes(b"\x00" * 16)
    return run


def test_log_training_run_creates_experiment_and_run(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "memory://")
    run_dir = _make_run_dir(tmp_path)

    with patch("deepVogue.tracking.mlflow_helpers.mlflow") as mf:
        mf.set_experiment.return_value = None
        mf.start_run.return_value.__enter__.return_value = MagicMock(info=MagicMock(run_id="r1"))
        mf.start_run.return_value.__exit__.return_value = False
        log_training_run(run_dir, dataset_name="tarot", cfg="stylegan3-t", kimg=5000, gamma=2.0, batch=32, res=512)

        mf.set_experiment.assert_called_once_with("tarot")
        mf.log_params.assert_called_once()
        params = mf.log_params.call_args.args[0]
        assert params["cfg"] == "stylegan3-t"
        assert params["gamma"] == 2.0
        # metrics: two FID steps
        assert mf.log_metric.call_count >= 2
        metric_calls = [c.args for c in mf.log_metric.call_args_list]
        assert any(c[0] == "fid50k_full" and c[1] == 142.3 for c in metric_calls)
        assert any(c[0] == "fid50k_full" and c[1] == 98.6 for c in metric_calls)


def test_log_training_run_skips_when_no_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "memory://")
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    with patch("deepVogue.tracking.mlflow_helpers.mlflow") as mf:
        mf.start_run.return_value.__enter__.return_value = MagicMock(info=MagicMock(run_id="r1"))
        mf.start_run.return_value.__exit__.return_value = False
        log_training_run(run_dir, dataset_name="tarot", cfg="stylegan3-t", kimg=0, gamma=2.0, batch=32, res=512)
        mf.log_metric.assert_not_called()
```

- [ ] **Step 2: Run tests — confirm fail**

Run: `pytest tests/test_mlflow_helpers.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`deepVogue/tracking/__init__.py`:
```python
from . import mlflow_helpers as mlflow_helpers  # noqa: F401
```

`deepVogue/tracking/mlflow_helpers.py`:
```python
"""MLflow logging helpers for deepVogue training runs."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlflow

log = logging.getLogger(__name__)


def _read_fid_jsonl(run_dir: Path) -> list[dict]:
    jsonl = run_dir / "metric-fid50k_full.jsonl"
    if not jsonl.exists():
        return []
    rows = []
    for line in jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("skipping malformed FID jsonl line: %s", line[:80])
    return rows


def _snapshot_step(name: str) -> int:
    # e.g. network-snapshot-005200.pkl -> 5200
    try:
        return int(name.split("network-snapshot-")[1].split(".")[0])
    except (IndexError, ValueError):
        return 0


def log_training_run(
    run_dir: Path,
    *,
    dataset_name: str,
    cfg: str,
    kimg: int,
    gamma: float,
    batch: int,
    res: int,
    extra_tags: dict[str, str] | None = None,
    log_snapshot_every: int = 1,
) -> str:
    """Log a deepVogue training run dir to MLflow. Returns the MLflow run_id."""
    run_dir = Path(run_dir)
    mlflow.set_experiment(dataset_name)
    with mlflow.start_run() as run:
        mlflow.log_params({
            "cfg": cfg, "kimg": kimg, "gamma": gamma, "batch": batch, "res": res,
            "run_dir": str(run_dir),
        })
        if extra_tags:
            mlflow.set_tags(extra_tags)
        fid_rows = _read_fid_jsonl(run_dir)
        snapshots_logged = 0
        for row in fid_rows:
            snap = row.get("snapshot_pkl", "")
            step = _snapshot_step(snap)
            fid = row.get("results", {}).get("fid50k_full")
            if fid is not None:
                mlflow.log_metric("fid50k_full", float(fid), step=step)
            pkl = run_dir / snap
            if pkl.exists() and snapshots_logged % log_snapshot_every == 0:
                mlflow.log_artifact(str(pkl), artifact_path="snapshots")
            snapshots_logged += 1
        return run.info.run_id
```

- [ ] **Step 4: Run tests — confirm pass**

Run: `pytest tests/test_mlflow_helpers.py -v`
Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add deepVogue/tracking/ tests/test_mlflow_helpers.py
git commit -m "feat(tracking): MLflow logging helpers for training runs"
```

### Task 1.4: `deepVogue/publish.py` — Drive→GCS bridge

**Files:**
- Create: `deepVogue/publish.py`
- Create: `tests/test_publish.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_publish.py`:
```python
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import yaml
from deepVogue.publish import publish_checkpoint, find_latest_snapshot


def _make_drive_snapshot(tmp_path: Path) -> Path:
    drive = tmp_path / "drive_sync" / "tarot"
    drive.mkdir(parents=True)
    (drive / "network-snapshot-000200.pkl").write_bytes(b"\x00" * 16)
    (drive / "network-snapshot-000400.pkl").write_bytes(b"\x00" * 16)
    (drive / "metric-fid50k_full.jsonl").write_text(
        json.dumps({"snapshot_pkl": "network-snapshot-000400.pkl", "results": {"fid50k_full": 98.6}}) + "\n"
    )
    return drive


def test_find_latest_snapshot(tmp_path):
    drive = _make_drive_snapshot(tmp_path)
    latest = find_latest_snapshot(drive)
    assert latest.name == "network-snapshot-000400.pkl"


def test_publish_uploads_and_appends_models_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "memory")
    monkeypatch.setenv("DV_PUBLISH_TARGET", "memory://deepvogue-models")
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)

    drive = _make_drive_snapshot(tmp_path)

    # bypass real legacy.load_network_pkl
    with patch("deepVogue.publish._validate_pkl", return_value=None):
        info = publish_checkpoint(
            model_id="tarot_v1",
            src_dir=drive,
            backbone="sg3-t",
            dataset_kind="stills",
            default_trunc=0.7,
        )

    import fsspec
    fs = fsspec.filesystem("memory")
    files = fs.ls("/deepvogue-models")
    files_str = [f if isinstance(f, str) else f["name"] for f in files]
    assert any("tarot_v1" in f for f in files_str)
    assert any("models.yaml" in f for f in files_str)

    with fs.open("/deepvogue-models/models.yaml", "r") as f:
        registry = yaml.safe_load(f)
    assert len(registry) == 1
    assert registry[0]["id"] == "tarot_v1"
    assert registry[0]["pkl"].endswith("network-snapshot-000400.pkl")
    assert registry[0]["dataset_kind"] == "stills"
    assert info["fid"] == 98.6


def test_publish_appends_when_registry_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "memory")
    monkeypatch.setenv("DV_PUBLISH_TARGET", "memory://deepvogue-models")
    drive = _make_drive_snapshot(tmp_path)

    import fsspec
    fs = fsspec.filesystem("memory")
    with fs.open("/deepvogue-models/models.yaml", "w") as f:
        yaml.safe_dump([{"id": "existing", "backbone": "sg3-t", "pkl": "memory://existing.pkl"}], f)

    with patch("deepVogue.publish._validate_pkl", return_value=None):
        publish_checkpoint(model_id="tarot_v1", src_dir=drive, backbone="sg3-t", dataset_kind="stills")

    with fs.open("/deepvogue-models/models.yaml", "r") as f:
        registry = yaml.safe_load(f)
    assert len(registry) == 2
    assert {e["id"] for e in registry} == {"existing", "tarot_v1"}
```

- [ ] **Step 2: Run tests — confirm fail**

Run: `pytest tests/test_publish.py -v`
Expected: ImportError on `deepVogue.publish`.

- [ ] **Step 3: Implement**

`deepVogue/publish.py`:
```python
"""Publish a trained checkpoint from Drive (or any source dir) to GCS/MinIO and update models.yaml."""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from deepVogue.clients import get_artifact_fs
from deepVogue.notifications import slack

log = logging.getLogger(__name__)

_SNAP_RE = re.compile(r"network-snapshot-(\d+)\.pkl$")


def find_latest_snapshot(src_dir: Path) -> Path:
    candidates = []
    for p in Path(src_dir).iterdir():
        m = _SNAP_RE.search(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        raise FileNotFoundError(f"no network-snapshot-*.pkl under {src_dir}")
    candidates.sort()
    return candidates[-1][1]


def _read_latest_fid(src_dir: Path, snapshot_name: str) -> float | None:
    jsonl = Path(src_dir) / "metric-fid50k_full.jsonl"
    if not jsonl.exists():
        return None
    fid = None
    for line in jsonl.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("snapshot_pkl") == snapshot_name:
            fid = row.get("results", {}).get("fid50k_full")
    return float(fid) if fid is not None else None


def _validate_pkl(pkl_path: Path) -> None:
    """Hook for runtime pkl validation. Overridden in tests."""
    from deepVogue import legacy  # noqa: F401
    with open(pkl_path, "rb") as f:
        legacy.load_network_pkl(f)  # type: ignore[attr-defined]


def _strip_proto(uri: str) -> str:
    return uri.split("://", 1)[1] if "://" in uri else uri


def publish_checkpoint(
    *,
    model_id: str,
    src_dir: Path,
    backbone: str = "sg3-t",
    dataset_kind: str = "stills",
    default_trunc: float = 0.7,
    factors: str | None = None,
    anchors_dir: str | None = None,
    validate: bool = False,
) -> dict[str, Any]:
    """
    Copy latest snapshot from src_dir → DV_PUBLISH_TARGET; append/update models.yaml.
    Returns info dict (pkl uri, fid, etc.). Posts to Slack on success.
    """
    target_root = os.environ.get("DV_PUBLISH_TARGET")
    if not target_root:
        raise RuntimeError("DV_PUBLISH_TARGET must be set (e.g. gs://deepvogue-models)")
    fs = get_artifact_fs()
    target_path = _strip_proto(target_root).rstrip("/")

    snapshot = find_latest_snapshot(Path(src_dir))
    if validate:
        _validate_pkl(snapshot)

    fid = _read_latest_fid(Path(src_dir), snapshot.name)
    target_pkl_path = f"{target_path}/{model_id}/{snapshot.name}"
    pkl_uri = f"{target_root.rstrip('/')}/{model_id}/{snapshot.name}"

    with open(snapshot, "rb") as src, fs.open(target_pkl_path, "wb") as dst:
        dst.write(src.read())

    yaml_path = f"{target_path}/models.yaml"
    if fs.exists(yaml_path):
        with fs.open(yaml_path, "r") as f:
            registry = yaml.safe_load(f) or []
    else:
        registry = []

    entry: dict[str, Any] = {
        "id": model_id,
        "backbone": backbone,
        "pkl": pkl_uri,
        "dataset_kind": dataset_kind,
        "default_trunc": default_trunc,
    }
    if factors:
        entry["factors"] = factors
    if anchors_dir:
        entry["anchors_dir"] = anchors_dir
    if fid is not None:
        entry["fid"] = fid

    registry = [e for e in registry if e.get("id") != model_id]
    registry.append(entry)

    with fs.open(yaml_path, "w") as f:
        yaml.safe_dump(registry, f, sort_keys=False)

    info = {"pkl": pkl_uri, "fid": fid, "model_id": model_id}
    slack.notify_success(
        "publish", f"published {model_id}", {"pkl": pkl_uri, "fid": str(fid)}
    )
    return info


def _main() -> None:
    import argparse
    p = argparse.ArgumentParser(prog="python -m deepVogue.publish")
    p.add_argument("--model-id", required=True)
    p.add_argument("--src-dir", required=True, type=Path)
    p.add_argument("--backbone", default="sg3-t")
    p.add_argument("--dataset-kind", default="stills")
    p.add_argument("--default-trunc", default=0.7, type=float)
    p.add_argument("--factors", default=None)
    p.add_argument("--anchors-dir", default=None)
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    info = publish_checkpoint(
        model_id=args.model_id, src_dir=args.src_dir, backbone=args.backbone,
        dataset_kind=args.dataset_kind, default_trunc=args.default_trunc,
        factors=args.factors, anchors_dir=args.anchors_dir, validate=args.validate,
    )
    print(info)


if __name__ == "__main__":
    _main()
```

- [ ] **Step 4: Run tests — confirm pass**

Run: `pytest tests/test_publish.py -v`
Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add deepVogue/publish.py tests/test_publish.py
git commit -m "feat(publish): Drive→GCS bridge with atomic models.yaml append"
```

---

## Phase 2 — Stub model fixture + inference container (90 min)

### Task 2.1: stub SG3-t state dict fixture

**Files:**
- Create: `scripts/build_stub_state_dict.py`
- Create: `tests/fixtures/stub_sg3_state_dict.pt` (binary — generated by script)

- [ ] **Step 1: Write the generator script**

`scripts/build_stub_state_dict.py`:
```python
"""Build a tiny SG3-t generator state dict for local-nano testing.

Produces a structurally-valid SG3-t Generator that can forward-pass on CPU.
Run once; commit the output blob.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from deepVogue.training.networks_stylegan3 import Generator


def build(out_path: Path, img_resolution: int = 64, channel_max: int = 16) -> None:
    G = Generator(
        z_dim=512,
        c_dim=0,
        w_dim=512,
        img_resolution=img_resolution,
        img_channels=3,
        channel_base=4096,
        channel_max=channel_max,
        num_layers=4,
        first_cutoff=2,
        first_stopband=2.5,
    )
    G.eval()
    with torch.no_grad():
        z = torch.randn(1, 512)
        img = G(z, None)
        assert img.shape == (1, 3, img_resolution, img_resolution), img.shape
    torch.save({"G_ema": G.state_dict(), "stub": True}, out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("tests/fixtures/stub_sg3_state_dict.pt"))
    p.add_argument("--res", type=int, default=64)
    p.add_argument("--cmax", type=int, default=16)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    build(args.out, args.res, args.cmax)
```

- [ ] **Step 2: Run the generator**

Run: `python scripts/build_stub_state_dict.py`
Expected: prints `wrote tests/fixtures/stub_sg3_state_dict.pt (XX.X KB)`. File should be under 10 MB.

(If the SG3 Generator constructor signature differs from the snippet — check `deepVogue/training/networks_stylegan3.py` and adjust the kwargs. The contract: build a `Generator` that takes a `(B,512)` z and outputs `(B,3,64,64)`.)

- [ ] **Step 3: Commit fixture and script**

```bash
git add scripts/build_stub_state_dict.py tests/fixtures/stub_sg3_state_dict.pt
git commit -m "feat(fixtures): tiny SG3-t stub state dict for local-nano inference"
```

### Task 2.2: extend `models.yaml` schema to absolute URIs

**Files:**
- Modify: `deepVogue/serve/loader.py`
- Modify: `deepVogue/serve/registry.py`
- Create/extend: `tests/test_registry.py` (or extend existing)

- [ ] **Step 1: Add failing tests**

Append to `tests/test_registry.py`:
```python
import yaml
from deepVogue.serve.registry import Registry


def test_registry_accepts_absolute_gs_uri(tmp_path, monkeypatch):
    monkeypatch.setenv("DV_MODELS_ROOT", str(tmp_path))
    yml = tmp_path / "models.yaml"
    yml.write_text(yaml.safe_dump([
        {"id": "a", "backbone": "sg3-t", "pkl": "gs://b/a.pkl", "dataset_kind": "stills"},
    ]))
    reg = Registry(yml)
    entry = reg.get("a")
    assert entry["pkl"] == "gs://b/a.pkl"


def test_registry_resolves_relative_against_models_root(tmp_path, monkeypatch):
    monkeypatch.setenv("DV_MODELS_ROOT", str(tmp_path))
    (tmp_path / "tarot").mkdir()
    (tmp_path / "tarot" / "snap.pkl").write_bytes(b"\x00")
    yml = tmp_path / "models.yaml"
    yml.write_text(yaml.safe_dump([
        {"id": "tarot", "backbone": "sg3-t", "pkl": "tarot/snap.pkl", "dataset_kind": "stills"},
    ]))
    reg = Registry(yml)
    entry = reg.get("tarot")
    # contract: registry passes the path through resolve_uri()
    assert entry["pkl_resolved"].endswith("/tarot/snap.pkl")
```

- [ ] **Step 2: Run tests — confirm fail**

Run: `pytest tests/test_registry.py -v`
Expected: 2 new tests fail.

- [ ] **Step 3: Update `deepVogue/serve/registry.py`**

Inspect the current `Registry` class. Adjust `get()` (or equivalent) so that it adds a derived `pkl_resolved` field via `deepVogue.clients.resolve_uri`. Roughly:
```python
from deepVogue.clients import resolve_uri

# inside Registry.get / Registry.load:
entry["pkl_resolved"] = resolve_uri(entry["pkl"])
```

- [ ] **Step 4: Update `deepVogue/serve/loader.py`**

Change the place that opens the pkl: replace `Path(entry["pkl"]).open("rb")` with `fsspec.open(entry["pkl_resolved"], "rb")`. Add `import fsspec`.

- [ ] **Step 5: Run all serve tests — confirm pass**

Run: `pytest tests/test_registry.py tests/test_films_lookup.py -v`
Expected: all PASSED (incl. existing tests still green).

- [ ] **Step 6: Commit**

```bash
git add deepVogue/serve/registry.py deepVogue/serve/loader.py tests/test_registry.py
git commit -m "feat(serve): registry accepts absolute s3://gs:// pkl URIs via fsspec"
```

### Task 2.3: `infra/docker/inference/Dockerfile`

**Files:**
- Create: `infra/docker/inference/Dockerfile`
- Create: `infra/docker/inference/.dockerignore`

- [ ] **Step 1: Write the Dockerfile**

`infra/docker/inference/Dockerfile`:
```dockerfile
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
```

- [ ] **Step 2: Write .dockerignore**

`infra/docker/inference/.dockerignore`:
```
.git
.venv
__pycache__
*.pyc
results/
deepVogue/results/
notebooks/
tests/fixtures/*.pkl
.pytest_cache
.idea
.DS_Store
*.mp4
```

- [ ] **Step 3: Build the image**

Run: `docker build -f infra/docker/inference/Dockerfile -t deepvogue-inference:local .`
Expected: image builds successfully (5–10 min cold).

- [ ] **Step 4: Smoke-run the container (with stub state dict)**

```bash
docker run --rm -d --name dv-inf-smoke -p 18080:8080 \
  -e DV_MODELS_YAML=/app/tests/fixtures/stub_models.yaml \
  deepvogue-inference:local
sleep 3
curl -fsS http://localhost:18080/health
docker stop dv-inf-smoke
```

Expected: `{"status":"ok"}` or equivalent from `/health`.

(If `/health` route doesn't exist yet, add a stub in `deepVogue/serve/api.py`:
```python
@app.get("/health")
def health():
    return {"status": "ok"}
```
Already added in prior phase — STANDING.md mentions it. Verify with `grep -n "@app.get(\"/health\")" deepVogue/serve/api.py`.)

- [ ] **Step 5: Commit**

```bash
git add infra/docker/inference/
git commit -m "feat(infra): inference container Dockerfile on pytorch:2.4-cuda12.1"
```

---

## Phase 3 — docker-compose local stack (90 min)

### Task 3.1: `infra/docker/mlflow/Dockerfile`

**Files:**
- Create: `infra/docker/mlflow/Dockerfile`

- [ ] **Step 1: Write the Dockerfile**

```dockerfile
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
```

- [ ] **Step 2: Build**

Run: `docker build -f infra/docker/mlflow/Dockerfile -t deepvogue-mlflow:local infra/docker/mlflow/`
Expected: image builds.

- [ ] **Step 3: Commit**

```bash
git add infra/docker/mlflow/
git commit -m "feat(infra): MLflow tracking server Dockerfile"
```

### Task 3.2: `infra/docker/prefect/Dockerfile`

**Files:**
- Create: `infra/docker/prefect/Dockerfile`

- [ ] **Step 1: Write the Dockerfile (multi-target)**

```dockerfile
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
```

- [ ] **Step 2: Build both targets**

```bash
docker build -f infra/docker/prefect/Dockerfile --build-arg TARGET=server -t deepvogue-prefect-server:local .
docker build -f infra/docker/prefect/Dockerfile --build-arg TARGET=worker -t deepvogue-prefect-worker:local .
```
Expected: both build.

- [ ] **Step 3: Commit**

```bash
git add infra/docker/prefect/
git commit -m "feat(infra): Prefect server + worker Dockerfile (build-arg target)"
```

### Task 3.3: `infra/.env.example` + `infra/docker-compose.yml`

**Files:**
- Create: `infra/.env.example`
- Create: `infra/docker-compose.yml`

- [ ] **Step 1: Write env example**

`infra/.env.example`:
```
# === Artifact backend ===
DV_ARTIFACT_BACKEND=s3
DV_S3_ENDPOINT_URL=http://minio:9000
AWS_ACCESS_KEY_ID=minio
AWS_SECRET_ACCESS_KEY=miniopass

# === MinIO ===
MINIO_ROOT_USER=minio
MINIO_ROOT_PASSWORD=miniopass

# === Postgres ===
POSTGRES_USER=dv
POSTGRES_PASSWORD=dv
POSTGRES_DB=postgres

# === MLflow ===
MLFLOW_BACKEND_STORE_URI=postgresql+psycopg2://dv:dv@postgres:5432/mlflow
MLFLOW_ARTIFACTS_DESTINATION=s3://deepvogue-mlflow
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_S3_ENDPOINT_URL=http://minio:9000

# === Prefect ===
PREFECT_API_URL=http://prefect:4200/api
PREFECT_SERVER_DATABASE_CONNECTION_URL=postgresql+asyncpg://dv:dv@postgres:5432/prefect

# === FastAPI inference ===
DV_MODELS_YAML=s3://deepvogue-models/models.yaml
DV_MODELS_ROOT=s3://deepvogue-models

# === Slack ===
# Set to your incoming webhook URL to enable. Empty = silent no-op.
SLACK_WEBHOOK_URL=
```

- [ ] **Step 2: Write docker-compose.yml**

`infra/docker-compose.yml`:
```yaml
name: deepvogue-nano

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./postgres-init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 5s
      timeout: 3s
      retries: 10

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
    ports: ["9000:9000", "9001:9001"]
    volumes:
      - miniodata:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 5s
      timeout: 3s
      retries: 10

  minio-init:
    image: minio/mc:latest
    depends_on:
      minio:
        condition: service_healthy
    entrypoint: >
      /bin/sh -c "
      mc alias set local http://minio:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD};
      mc mb -p local/deepvogue-models local/deepvogue-datasets local/deepvogue-walks local/deepvogue-mlflow || true;
      exit 0
      "

  mlflow:
    build:
      context: docker/mlflow
    environment:
      MLFLOW_BACKEND_STORE_URI: ${MLFLOW_BACKEND_STORE_URI}
      MLFLOW_ARTIFACTS_DESTINATION: ${MLFLOW_ARTIFACTS_DESTINATION}
      MLFLOW_S3_ENDPOINT_URL: ${MLFLOW_S3_ENDPOINT_URL}
      AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID}
      AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY}
    depends_on:
      postgres:
        condition: service_healthy
      minio-init:
        condition: service_completed_successfully
    ports: ["5000:5000"]

  prefect:
    build:
      context: ..
      dockerfile: infra/docker/prefect/Dockerfile
      args:
        TARGET: server
    environment:
      PREFECT_SERVER_API_HOST: 0.0.0.0
      PREFECT_SERVER_DATABASE_CONNECTION_URL: ${PREFECT_SERVER_DATABASE_CONNECTION_URL}
      PREFECT_API_URL: ${PREFECT_API_URL}
    depends_on:
      postgres:
        condition: service_healthy
    ports: ["4200:4200"]

  fastapi:
    build:
      context: ..
      dockerfile: infra/docker/inference/Dockerfile
    environment:
      DV_ARTIFACT_BACKEND: ${DV_ARTIFACT_BACKEND}
      DV_S3_ENDPOINT_URL: ${DV_S3_ENDPOINT_URL}
      AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID}
      AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY}
      DV_MODELS_YAML: ${DV_MODELS_YAML}
      DV_MODELS_ROOT: ${DV_MODELS_ROOT}
      MLFLOW_TRACKING_URI: ${MLFLOW_TRACKING_URI}
      SLACK_WEBHOOK_URL: ${SLACK_WEBHOOK_URL}
    depends_on:
      mlflow:
        condition: service_started
      minio-init:
        condition: service_completed_successfully
    ports: ["8080:8080"]

volumes:
  pgdata:
  miniodata:
```

- [ ] **Step 3: Postgres init SQL (create two DBs)**

`infra/postgres-init.sql`:
```sql
CREATE DATABASE mlflow;
CREATE DATABASE prefect;
```

- [ ] **Step 4: Bring it up**

```bash
cp infra/.env.example infra/.env
docker compose -f infra/docker-compose.yml --env-file infra/.env up -d
```
Expected: all services start. Wait ~30s.

- [ ] **Step 5: Smoke-check each service**

```bash
curl -fsS http://localhost:5000 | head -1          # MLflow UI
curl -fsS http://localhost:4200/api/health         # Prefect API
curl -fsS http://localhost:8080/health             # FastAPI
curl -fsS http://localhost:9001 | head -1          # MinIO console
```
Expected: each returns 200.

- [ ] **Step 6: Tear down**

Run: `docker compose -f infra/docker-compose.yml --env-file infra/.env down -v`

- [ ] **Step 7: Commit**

```bash
git add infra/.env.example infra/docker-compose.yml infra/postgres-init.sql
git commit -m "feat(infra): docker-compose nano stack — postgres+minio+mlflow+prefect+fastapi"
```

### Task 3.4: Makefile — `nano-up`, `nano-down`

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add targets**

Append to `Makefile`:
```makefile
# === MLOps stack — local nano ===
.PHONY: nano-up nano-down nano-logs

NANO_COMPOSE := docker compose -f infra/docker-compose.yml --env-file infra/.env

nano-up: ## Bring up local MLOps stack (postgres + minio + mlflow + prefect + fastapi)
	@test -f infra/.env || (echo "create infra/.env first: cp infra/.env.example infra/.env" && exit 1)
	$(NANO_COMPOSE) up -d --build
	@echo "MLflow:   http://localhost:5000"
	@echo "Prefect:  http://localhost:4200"
	@echo "FastAPI:  http://localhost:8080"
	@echo "MinIO:    http://localhost:9001"

nano-down:
	$(NANO_COMPOSE) down -v

nano-logs:
	$(NANO_COMPOSE) logs -f --tail=200
```

- [ ] **Step 2: Verify**

Run: `make -n nano-up` (dry-run)
Expected: prints the docker compose command.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "feat(make): nano-up/nano-down/nano-logs targets"
```

---

## Phase 4 — Prefect flows (3 hr)

### Task 4.1: `deepVogue/orchestration/backends/__init__.py` + local skeleton

**Files:**
- Create: `deepVogue/orchestration/__init__.py`, `deepVogue/orchestration/backends/__init__.py`, `deepVogue/orchestration/backends/local.py`, `deepVogue/orchestration/backends/colab.py`, `deepVogue/orchestration/backends/runpod.py`
- Create: `tests/test_orchestration_backends.py`

- [ ] **Step 1: Write failing tests for the dispatcher**

`tests/test_orchestration_backends.py`:
```python
import pytest
from deepVogue.orchestration.backends import get_backend, BackendOp


def test_get_local_backend_has_required_ops():
    b = get_backend("local")
    for op in ("prepare", "train", "publish", "project", "walk", "eval"):
        assert hasattr(b, op), f"local backend missing op: {op}"


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("nope")


def test_colab_backend_raises_notimplemented_in_v1():
    b = get_backend("colab")
    with pytest.raises(NotImplementedError):
        b.train(dataset_name="x", cfg="stylegan3-t", kimg=10, gamma=2.0, batch=32, res=64)


def test_runpod_backend_raises_notimplemented_in_v1():
    b = get_backend("runpod")
    with pytest.raises(NotImplementedError):
        b.train(dataset_name="x", cfg="stylegan3-t", kimg=10, gamma=2.0, batch=32, res=64)
```

- [ ] **Step 2: Run — confirm fail**

Run: `pytest tests/test_orchestration_backends.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement dispatcher**

`deepVogue/orchestration/__init__.py`: `# package`

`deepVogue/orchestration/backends/__init__.py`:
```python
from __future__ import annotations
from typing import Literal, Protocol

BackendName = Literal["local", "colab", "runpod"]


class BackendOp(Protocol):
    def prepare(self, **kw): ...
    def train(self, **kw): ...
    def publish(self, **kw): ...
    def project(self, **kw): ...
    def walk(self, **kw): ...
    def eval(self, **kw): ...


def get_backend(name: str) -> BackendOp:
    from . import local, colab, runpod
    name = name.lower()
    if name == "local":
        return local
    if name == "colab":
        return colab
    if name == "runpod":
        return runpod
    raise ValueError(f"unknown backend: {name!r}")
```

`deepVogue/orchestration/backends/colab.py`:
```python
def prepare(**kw): raise NotImplementedError("colab.prepare — v2; use local backend or run Colab manually")
def train(**kw): raise NotImplementedError("colab.train — v2; run Colab training notebook manually then `make publish`")
def publish(**kw): raise NotImplementedError("colab.publish — v2; use local backend")
def project(**kw): raise NotImplementedError("colab.project — v2; use local backend or Colab manually")
def walk(**kw): raise NotImplementedError("colab.walk — v2; use local backend or HTTP to inference server")
def eval(**kw): raise NotImplementedError("colab.eval — v2; use local backend or Colab manually")
```

`deepVogue/orchestration/backends/runpod.py`: same shape with `runpod.*` message.

`deepVogue/orchestration/backends/local.py`: (skeleton — implementations in 4.2)
```python
"""Local-nano backend: real for prepare/publish, mock for train/project/walk/eval."""
from __future__ import annotations

def prepare(**kw): raise NotImplementedError
def train(**kw): raise NotImplementedError
def publish(**kw): raise NotImplementedError
def project(**kw): raise NotImplementedError
def walk(**kw): raise NotImplementedError
def eval(**kw): raise NotImplementedError
```

- [ ] **Step 4: Run — confirm pass for dispatcher tests**

Run: `pytest tests/test_orchestration_backends.py::test_get_local_backend_has_required_ops tests/test_orchestration_backends.py::test_unknown_backend_raises tests/test_orchestration_backends.py::test_colab_backend_raises_notimplemented_in_v1 tests/test_orchestration_backends.py::test_runpod_backend_raises_notimplemented_in_v1 -v`
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add deepVogue/orchestration/ tests/test_orchestration_backends.py
git commit -m "feat(orchestration): backend dispatcher with local/colab/runpod skeleton"
```

### Task 4.2: `backends/local.py` — `prepare` (real), `publish` (real)

**Files:**
- Modify: `deepVogue/orchestration/backends/local.py`
- Modify: `tests/test_orchestration_backends.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_orchestration_backends.py`:
```python
import os
from pathlib import Path
from unittest.mock import patch
from PIL import Image

from deepVogue.orchestration.backends import get_backend


def test_local_prepare_creates_dataset_zip(tmp_path, monkeypatch):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "memory")
    src = tmp_path / "stills"
    src.mkdir()
    for i in range(3):
        Image.new("RGB", (64, 64), (i * 80, 0, 0)).save(src / f"{i}.png")
    b = get_backend("local")
    out = b.prepare(
        source_uri=str(src),
        dataset_name="tarot_nano",
        res=64,
        kind="stills",
        target_uri="memory://deepvogue-datasets",
    )
    assert out["dataset_uri"].endswith(".zip")
    import fsspec
    fs = fsspec.filesystem("memory")
    assert fs.exists(out["dataset_uri"].replace("memory://", "/"))


def test_local_publish_delegates_to_publish_module(tmp_path, monkeypatch):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "memory")
    monkeypatch.setenv("DV_PUBLISH_TARGET", "memory://deepvogue-models")
    drive = tmp_path / "drive"
    drive.mkdir()
    (drive / "network-snapshot-000200.pkl").write_bytes(b"\x00")
    b = get_backend("local")
    with patch("deepVogue.publish._validate_pkl", return_value=None):
        info = b.publish(model_id="tarot_v1", src_dir=str(drive), backbone="sg3-t", dataset_kind="stills")
    assert info["model_id"] == "tarot_v1"
```

- [ ] **Step 2: Implement `local.prepare` and `local.publish`**

Replace `deepVogue/orchestration/backends/local.py`:
```python
"""Local-nano backend."""
from __future__ import annotations

import io
import logging
import os
import time
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image
import numpy as np
import torch

from deepVogue.clients import get_artifact_fs

log = logging.getLogger(__name__)


# ----- real -----

def prepare(*, source_uri: str, dataset_name: str, res: int, kind: str = "stills",
            target_uri: str, fps: int | None = None) -> dict[str, Any]:
    """Read source_uri (a local dir of images or video file path), produce dataset.zip on target_uri."""
    fs = get_artifact_fs()
    src = Path(source_uri)
    images: list[tuple[str, bytes]] = []

    if kind == "stills":
        for p in sorted(src.iterdir()):
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            img = Image.open(p).convert("RGB").resize((res, res))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            images.append((p.stem + ".png", buf.getvalue()))
    elif kind == "frames":
        raise NotImplementedError("frames prep needs ffmpeg; use `make prepare-frames`")
    else:
        raise ValueError(f"unknown kind: {kind}")

    if not images:
        raise RuntimeError(f"no images found under {src}")

    target_uri = target_uri.rstrip("/")
    proto, path = (target_uri.split("://", 1) + [""])[:2] if "://" in target_uri else (None, target_uri)
    target_path = f"/{path}" if proto else target_uri
    zip_path = f"{target_path}/{dataset_name}.zip"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in images:
            zf.writestr(name, data)
    with fs.open(zip_path, "wb") as f:
        f.write(buf.getvalue())

    dataset_uri = f"{target_uri}/{dataset_name}.zip" if proto else zip_path
    return {"dataset_uri": dataset_uri, "n_images": len(images)}


def publish(**kw) -> dict[str, Any]:
    """Real publish — defers to deepVogue.publish."""
    from deepVogue.publish import publish_checkpoint
    src_dir = kw.pop("src_dir")
    return publish_checkpoint(src_dir=Path(src_dir), **kw)


# ----- mock -----

def _stub_state_dict() -> dict[str, Any]:
    fp = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "stub_sg3_state_dict.pt"
    if not fp.exists():
        raise FileNotFoundError(
            f"stub state dict missing at {fp} — run `python scripts/build_stub_state_dict.py`"
        )
    return torch.load(fp, map_location="cpu", weights_only=False)


def train(*, dataset_name: str, cfg: str, kimg: int, gamma: float, batch: int,
          res: int = 64, target_uri: str | None = None, **_) -> dict[str, Any]:
    log.info("[nano-mock] training %s on %s for %d kimg", cfg, dataset_name, kimg)
    time.sleep(min(kimg / 50.0, 5.0))
    fs = get_artifact_fs()
    target_uri = (target_uri or f"memory://deepvogue-models/{dataset_name}_nano").rstrip("/")
    proto = "://" in target_uri
    target_path = "/" + target_uri.split("://", 1)[1] if proto else target_uri
    pkl_path = f"{target_path}/network-snapshot-{kimg:06d}.pkl"
    state = _stub_state_dict()
    import io
    buf = io.BytesIO()
    torch.save(state, buf)
    with fs.open(pkl_path, "wb") as f:
        f.write(buf.getvalue())
    fid = max(5.0, 200.0 - kimg / 25.0)  # fake monotonic decrease
    pkl_uri = f"{target_uri}/network-snapshot-{kimg:06d}.pkl" if proto else pkl_path
    return {"pkl": pkl_uri, "fid": fid, "kimg": kimg}


def project(*, model_id: str, frames_uri: str, stride: int = 4, steps: int = 50,
            target_uri: str, **_) -> dict[str, Any]:
    log.info("[nano-mock] project %s stride=%d steps=%d", model_id, stride, steps)
    time.sleep(1.0)
    fs = get_artifact_fs()
    target_uri = target_uri.rstrip("/")
    proto = "://" in target_uri
    target_path = "/" + target_uri.split("://", 1)[1] if proto else target_uri
    out_uri = f"{target_path}/{model_id}/0/projected_w.npz"
    arr = np.zeros((1, 16, 512), dtype=np.float32)
    buf = io.BytesIO()
    np.savez(buf, w=arr)
    with fs.open(out_uri, "wb") as f:
        f.write(buf.getvalue())
    return {"anchors_uri": f"{target_uri}/{model_id}", "n_anchors": 1}


def walk(*, model_id: str, target_uri: str, steps: int = 60, fps: int = 24,
         seeds: list[int] | None = None, anchors_uri: str | None = None, mode: str = "cubic",
         **_) -> dict[str, Any]:
    log.info("[nano-mock] walk %s steps=%d fps=%d mode=%s", model_id, steps, fps, mode)
    time.sleep(1.0)
    fs = get_artifact_fs()
    target_uri = target_uri.rstrip("/")
    proto = "://" in target_uri
    target_path = "/" + target_uri.split("://", 1)[1] if proto else target_uri
    walk_id = f"walk_{int(time.time())}"
    out = f"{target_path}/{model_id}/{walk_id}.mp4"
    with fs.open(out, "wb") as f:
        f.write(b"\x00\x00\x00\x18ftypmp42")  # minimal MP4 header bytes; not playable, structurally tagged
    return {"walk_uri": f"{target_uri}/{model_id}/{walk_id}.mp4", "walk_id": walk_id}


def eval(*, model_id: str, dataset_uri: str, **_) -> dict[str, Any]:
    log.info("[nano-mock] eval %s on %s", model_id, dataset_uri)
    time.sleep(0.5)
    return {"fid50k_full": 42.0, "kid50k_full": 0.003}
```

- [ ] **Step 3: Run all backend tests — confirm pass**

Run: `pytest tests/test_orchestration_backends.py -v`
Expected: all PASSED.

- [ ] **Step 4: Commit**

```bash
git add deepVogue/orchestration/backends/local.py tests/test_orchestration_backends.py
git commit -m "feat(orchestration): local backend — real prepare/publish + mock train/project/walk/eval"
```

### Task 4.3: `deepVogue/orchestration/flows.py` — Prefect flow wrappers

**Files:**
- Create: `deepVogue/orchestration/flows.py`
- Create: `tests/test_flows.py`

- [ ] **Step 1: Write failing tests (in-process flow runs)**

`tests/test_flows.py`:
```python
import os
from pathlib import Path
from unittest.mock import patch
from PIL import Image

from deepVogue.orchestration.flows import (
    prepare_flow, train_flow, publish_flow, walk_flow, pipeline_stills,
)


def test_prepare_flow_returns_dataset_uri(tmp_path, monkeypatch):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "memory")
    src = tmp_path / "raw"
    src.mkdir()
    for i in range(2):
        Image.new("RGB", (64, 64)).save(src / f"{i}.png")
    out = prepare_flow(
        source_uri=str(src), dataset_name="t_nano", res=64,
        kind="stills", target_uri="memory://deepvogue-datasets", backend="local",
    )
    assert out["dataset_uri"].endswith(".zip")


def test_train_flow_writes_fake_pkl(monkeypatch):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "memory")
    out = train_flow(
        dataset_name="t_nano", cfg="stylegan3-t", kimg=50,
        gamma=2.0, batch=8, res=64,
        target_uri="memory://deepvogue-models/t_nano", backend="local",
    )
    assert out["kimg"] == 50
    assert out["fid"] < 200.0


def test_pipeline_stills_runs_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "memory")
    monkeypatch.setenv("DV_PUBLISH_TARGET", "memory://deepvogue-models")
    src = tmp_path / "raw"
    src.mkdir()
    for i in range(2):
        Image.new("RGB", (64, 64)).save(src / f"{i}.png")
    with patch("deepVogue.publish._validate_pkl", return_value=None):
        result = pipeline_stills(
            source_uri=str(src), dataset_name="t_nano", model_id="t_nano_v1",
            res=64, kimg=50, gamma=2.0, batch=8, walk_steps=10, walk_fps=12,
            backend="local",
        )
    assert "walk_uri" in result
    assert "pkl" in result["train"]
```

- [ ] **Step 2: Implement flows**

`deepVogue/orchestration/flows.py`:
```python
"""Prefect flows for the deepVogue pipeline. All tasks delegate to a backend module."""
from __future__ import annotations

import logging
from typing import Any

from prefect import flow, task

from deepVogue.orchestration.backends import get_backend
from deepVogue.notifications import slack

log = logging.getLogger(__name__)


@task
def _prepare_task(*, backend: str, **kw): return get_backend(backend).prepare(**kw)

@task
def _train_task(*, backend: str, **kw): return get_backend(backend).train(**kw)

@task
def _publish_task(*, backend: str, **kw): return get_backend(backend).publish(**kw)

@task
def _project_task(*, backend: str, **kw): return get_backend(backend).project(**kw)

@task
def _walk_task(*, backend: str, **kw): return get_backend(backend).walk(**kw)

@task
def _eval_task(*, backend: str, **kw): return get_backend(backend).eval(**kw)


@flow(name="prepare_flow")
def prepare_flow(*, source_uri: str, dataset_name: str, res: int, target_uri: str,
                 kind: str = "stills", fps: int | None = None, backend: str = "local") -> dict[str, Any]:
    out = _prepare_task(backend=backend, source_uri=source_uri, dataset_name=dataset_name,
                        res=res, kind=kind, target_uri=target_uri, fps=fps)
    slack.notify_success("flow", f"prepare {dataset_name} done",
                         {"dataset_uri": out["dataset_uri"], "n": str(out.get("n_images", 0))})
    return out


@flow(name="train_flow")
def train_flow(*, dataset_name: str, cfg: str, kimg: int, gamma: float, batch: int,
               res: int = 256, target_uri: str | None = None, backend: str = "local",
               resume_from: str | None = None) -> dict[str, Any]:
    out = _train_task(backend=backend, dataset_name=dataset_name, cfg=cfg, kimg=kimg,
                      gamma=gamma, batch=batch, res=res, target_uri=target_uri,
                      resume_from=resume_from)
    slack.notify_success("flow", f"train {dataset_name} done",
                         {"pkl": out["pkl"], "fid": str(out.get("fid"))})
    return out


@flow(name="publish_flow")
def publish_flow(*, model_id: str, src_dir: str, backbone: str = "sg3-t",
                 dataset_kind: str = "stills", default_trunc: float = 0.7,
                 backend: str = "local") -> dict[str, Any]:
    return _publish_task(backend=backend, model_id=model_id, src_dir=src_dir,
                         backbone=backbone, dataset_kind=dataset_kind,
                         default_trunc=default_trunc)


@flow(name="project_flow")
def project_flow(*, model_id: str, frames_uri: str, target_uri: str,
                 stride: int = 4, steps: int = 500, backend: str = "local") -> dict[str, Any]:
    return _project_task(backend=backend, model_id=model_id, frames_uri=frames_uri,
                         stride=stride, steps=steps, target_uri=target_uri)


@flow(name="walk_flow")
def walk_flow(*, model_id: str, target_uri: str, steps: int = 60, fps: int = 24,
              seeds: list[int] | None = None, anchors_uri: str | None = None,
              mode: str = "cubic", backend: str = "local") -> dict[str, Any]:
    out = _walk_task(backend=backend, model_id=model_id, target_uri=target_uri,
                     steps=steps, fps=fps, seeds=seeds, anchors_uri=anchors_uri, mode=mode)
    slack.notify_success("flow", f"walk {model_id} done", {"walk_uri": out["walk_uri"]})
    return out


@flow(name="eval_flow")
def eval_flow(*, model_id: str, dataset_uri: str, backend: str = "local") -> dict[str, Any]:
    return _eval_task(backend=backend, model_id=model_id, dataset_uri=dataset_uri)


@flow(name="pipeline_stills")
def pipeline_stills(*, source_uri: str, dataset_name: str, model_id: str,
                    res: int = 256, kimg: int = 5000, gamma: float = 2.0, batch: int = 32,
                    walk_steps: int = 60, walk_fps: int = 24, backend: str = "local") -> dict[str, Any]:
    prep = prepare_flow(source_uri=source_uri, dataset_name=dataset_name, res=res,
                        target_uri="memory://deepvogue-datasets" if backend == "local" else "gs://deepvogue-datasets",
                        backend=backend)
    train = train_flow(dataset_name=dataset_name, cfg="stylegan3-t", kimg=kimg, gamma=gamma,
                       batch=batch, res=res,
                       target_uri="memory://deepvogue-models/" + dataset_name if backend == "local"
                                  else f"gs://deepvogue-models/{dataset_name}",
                       backend=backend)
    # publish requires a local directory; nano flow skips actual publish and just records URIs
    walk = walk_flow(model_id=model_id,
                     target_uri="memory://deepvogue-walks" if backend == "local" else "gs://deepvogue-walks",
                     steps=walk_steps, fps=walk_fps, backend=backend)
    return {"prepare": prep, "train": train, "walk": walk, **walk}


@flow(name="pipeline_frames")
def pipeline_frames(*, source_uri: str, dataset_name: str, model_id: str,
                    res: int = 256, fps: int = 1, kimg: int = 5000, gamma: float = 2.0,
                    batch: int = 32, walk_steps: int = 60, walk_fps: int = 24,
                    backend: str = "local") -> dict[str, Any]:
    prep = prepare_flow(source_uri=source_uri, dataset_name=dataset_name, res=res,
                        kind="frames", fps=fps,
                        target_uri="memory://deepvogue-datasets" if backend == "local" else "gs://deepvogue-datasets",
                        backend=backend)
    train = train_flow(dataset_name=dataset_name, cfg="stylegan3-t", kimg=kimg, gamma=gamma,
                       batch=batch, res=res,
                       target_uri="memory://deepvogue-models/" + dataset_name if backend == "local"
                                  else f"gs://deepvogue-models/{dataset_name}",
                       backend=backend)
    project = project_flow(model_id=model_id, frames_uri=prep["dataset_uri"],
                           target_uri="memory://deepvogue-anchors" if backend == "local" else "gs://deepvogue-anchors",
                           backend=backend)
    walk = walk_flow(model_id=model_id, anchors_uri=project["anchors_uri"],
                     target_uri="memory://deepvogue-walks" if backend == "local" else "gs://deepvogue-walks",
                     steps=walk_steps, fps=walk_fps, backend=backend)
    ev = eval_flow(model_id=model_id, dataset_uri=prep["dataset_uri"], backend=backend)
    return {"prepare": prep, "train": train, "project": project, "walk": walk, "eval": ev}
```

- [ ] **Step 3: Run flow tests — confirm pass**

Run: `pytest tests/test_flows.py -v`
Expected: 3 PASSED.

(Prefect flows running in-process may emit logs; that's fine. If pytest-asyncio plugin complains, add `pytest-asyncio` to requirements and `asyncio_mode = auto` to `pytest.ini` or `pyproject.toml`.)

- [ ] **Step 4: Commit**

```bash
git add deepVogue/orchestration/flows.py tests/test_flows.py
git commit -m "feat(orchestration): Prefect flows — prepare/train/publish/project/walk/eval + pipelines"
```

---

## Phase 5 — Nano smoke test (45 min)

### Task 5.1: `tests/test_nano_smoke.py` — integration test

**Files:**
- Create: `tests/test_nano_smoke.py`
- Create: `scripts/run_nano_smoke.py`

- [ ] **Step 1: Integration test — runs only when DV_NANO_SMOKE=1**

`tests/test_nano_smoke.py`:
```python
"""Integration test: runs against an already-running docker-compose nano stack.

Skipped unless DV_NANO_SMOKE=1. Triggered by `make nano-smoke` after `make nano-up`.
"""
import os
import time
from pathlib import Path

import pytest
import requests

NANO = pytest.mark.skipif(os.environ.get("DV_NANO_SMOKE") != "1",
                          reason="set DV_NANO_SMOKE=1 + `make nano-up` first")


@NANO
def test_mlflow_reachable():
    r = requests.get("http://localhost:5000", timeout=5)
    assert r.status_code == 200


@NANO
def test_prefect_reachable():
    r = requests.get("http://localhost:4200/api/health", timeout=5)
    assert r.status_code == 200


@NANO
def test_fastapi_reachable():
    r = requests.get("http://localhost:8080/health", timeout=5)
    assert r.status_code == 200


@NANO
def test_pipeline_stills_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("DV_ARTIFACT_BACKEND", "s3")
    monkeypatch.setenv("DV_S3_ENDPOINT_URL", "http://localhost:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "minio")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "miniopass")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")

    from PIL import Image
    src = tmp_path / "raw"; src.mkdir()
    for i in range(3):
        Image.new("RGB", (64, 64), (i * 80, 0, 0)).save(src / f"{i}.png")

    from deepVogue.orchestration.flows import pipeline_stills
    out = pipeline_stills(
        source_uri=str(src),
        dataset_name="smoke_stills",
        model_id="smoke_stills_v1",
        res=64, kimg=50, gamma=2.0, batch=8,
        walk_steps=10, walk_fps=12,
        backend="local",
    )
    assert "walk_uri" in out
    # verify mp4 actually exists in MinIO
    import fsspec
    fs = fsspec.filesystem("s3", client_kwargs={"endpoint_url": "http://localhost:9000"})
    path = out["walk_uri"].split("://", 1)[1]
    assert fs.exists(path), f"mp4 not in MinIO at {path}"
```

- [ ] **Step 2: Convenience runner script**

`scripts/run_nano_smoke.py`:
```python
"""Wait for the nano stack to be ready, then run the smoke test."""
import os
import sys
import time
import urllib.request


def wait(url: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return
        except Exception as e:
            last = e
            time.sleep(1)
    raise RuntimeError(f"timeout waiting for {url}: {last}")


def main():
    for url in ["http://localhost:5000", "http://localhost:4200/api/health",
                "http://localhost:8080/health", "http://localhost:9000/minio/health/live"]:
        print(f"waiting on {url}...", flush=True)
        wait(url)
    print("all services up; running smoke", flush=True)
    os.environ["DV_NANO_SMOKE"] = "1"
    rc = os.system("pytest tests/test_nano_smoke.py -v")
    sys.exit(0 if rc == 0 else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Add make target**

Append to `Makefile`:
```makefile
.PHONY: nano-smoke
nano-smoke: ## Run the local-nano integration smoke against a running stack
	python scripts/run_nano_smoke.py
```

- [ ] **Step 4: Execute end-to-end**

```bash
make nano-up
make nano-smoke
make nano-down
```
Expected: smoke test passes, `walk_uri` resolves to an MP4 stub in MinIO.

- [ ] **Step 5: Commit**

```bash
git add tests/test_nano_smoke.py scripts/run_nano_smoke.py Makefile
git commit -m "feat(test): nano-smoke integration — full pipeline against docker-compose stack"
```

---

## Phase 6 — GCP infra setup (2 hr)

### Task 6.1: `infra/gcp/setup.sh` — APIs, buckets, AR repo

**Files:**
- Create: `infra/gcp/setup.sh`

- [ ] **Step 1: Write the script**

`infra/gcp/setup.sh`:
```bash
#!/usr/bin/env bash
# Idempotent GCP bootstrap for deepVogue MLOps stack.
# Required env: GCP_PROJECT, GCP_REGION (default us-central1), GITHUB_REPO (owner/repo)
set -euo pipefail

PROJECT="${GCP_PROJECT:?set GCP_PROJECT}"
REGION="${GCP_REGION:-us-central1}"
GH_REPO="${GITHUB_REPO:?set GITHUB_REPO=owner/repo}"

step() { echo; echo "==> $*"; }

step "Enable APIs"
gcloud --project "$PROJECT" services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  sqladmin.googleapis.com \
  storage.googleapis.com \
  iamcredentials.googleapis.com \
  iam.googleapis.com \
  vpcaccess.googleapis.com \
  servicenetworking.googleapis.com \
  iap.googleapis.com

step "GCS buckets"
for b in models datasets walks mlflow queue; do
  if ! gcloud --project "$PROJECT" storage buckets describe "gs://deepvogue-$b" >/dev/null 2>&1; then
    gcloud --project "$PROJECT" storage buckets create "gs://deepvogue-$b" \
      --location="$REGION" --uniform-bucket-level-access
  fi
done

step "Artifact Registry repo"
if ! gcloud --project "$PROJECT" artifacts repositories describe deepvogue \
     --location="$REGION" >/dev/null 2>&1; then
  gcloud --project "$PROJECT" artifacts repositories create deepvogue \
    --repository-format=docker --location="$REGION" --description="deepVogue images"
fi

echo
echo "Stage 1 (APIs, buckets, AR) complete."
echo "Next: infra/gcp/setup-sql.sh and infra/gcp/setup-iam.sh"
```

- [ ] **Step 2: Make executable + dry-check syntax**

```bash
chmod +x infra/gcp/setup.sh
bash -n infra/gcp/setup.sh
```
Expected: no syntax errors.

- [ ] **Step 3: Commit**

```bash
git add infra/gcp/setup.sh
git commit -m "feat(gcp): bootstrap script — enable APIs, create GCS buckets + AR repo"
```

### Task 6.2: `infra/gcp/setup-sql.sh` — Cloud SQL + VPC

**Files:**
- Create: `infra/gcp/setup-sql.sh`

- [ ] **Step 1: Write**

```bash
#!/usr/bin/env bash
set -euo pipefail
PROJECT="${GCP_PROJECT:?}"
REGION="${GCP_REGION:-us-central1}"
INSTANCE="deepvogue-pg"
NETWORK="default"

if ! gcloud --project "$PROJECT" sql instances describe "$INSTANCE" >/dev/null 2>&1; then
  gcloud --project "$PROJECT" sql instances create "$INSTANCE" \
    --database-version=POSTGRES_15 --tier=db-f1-micro --region="$REGION" \
    --network="$NETWORK" --no-assign-ip
fi

for db in mlflow prefect; do
  gcloud --project "$PROJECT" sql databases create "$db" --instance="$INSTANCE" 2>/dev/null || true
done

PASS="$(openssl rand -base64 24 | tr -d '/+=')"
gcloud --project "$PROJECT" sql users create dv --instance="$INSTANCE" --password="$PASS" 2>/dev/null || true
echo "Cloud SQL user 'dv' password: $PASS"
echo "Save this to Secret Manager: gcloud secrets create deepvogue-pg-password --data-file=- <<<$PASS"

# VPC connector for Cloud Run -> private SQL
if ! gcloud --project "$PROJECT" compute networks vpc-access connectors describe \
     deepvogue-vpc --region="$REGION" >/dev/null 2>&1; then
  gcloud --project "$PROJECT" compute networks vpc-access connectors create deepvogue-vpc \
    --region="$REGION" --network="$NETWORK" --range=10.8.0.0/28
fi
```

- [ ] **Step 2: Make exec + syntax check + commit**

```bash
chmod +x infra/gcp/setup-sql.sh
bash -n infra/gcp/setup-sql.sh
git add infra/gcp/setup-sql.sh
git commit -m "feat(gcp): Cloud SQL + VPC connector bootstrap"
```

### Task 6.3: `infra/gcp/setup-iam.sh` — Runtime SAs + WIF

**Files:**
- Create: `infra/gcp/setup-iam.sh`

- [ ] **Step 1: Write**

```bash
#!/usr/bin/env bash
set -euo pipefail
PROJECT="${GCP_PROJECT:?}"
GH_REPO="${GITHUB_REPO:?owner/repo}"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"

# Runtime SAs
for sa in mlflow-server prefect-server prefect-worker fastapi-inference deepvogue-deployer; do
  if ! gcloud --project "$PROJECT" iam service-accounts describe \
       "${sa}-sa@${PROJECT}.iam.gserviceaccount.com" >/dev/null 2>&1; then
    gcloud --project "$PROJECT" iam service-accounts create "${sa}-sa" \
      --display-name="deepVogue $sa"
  fi
done

# Permissions — least-privilege
grant() {
  gcloud --project "$PROJECT" projects add-iam-policy-binding "$PROJECT" \
    --member="serviceAccount:$1" --role="$2" --condition=None 2>/dev/null || true
}
grant "mlflow-server-sa@${PROJECT}.iam.gserviceaccount.com" roles/storage.objectAdmin
grant "mlflow-server-sa@${PROJECT}.iam.gserviceaccount.com" roles/cloudsql.client
grant "prefect-server-sa@${PROJECT}.iam.gserviceaccount.com" roles/cloudsql.client
grant "prefect-worker-sa@${PROJECT}.iam.gserviceaccount.com" roles/storage.objectAdmin
grant "prefect-worker-sa@${PROJECT}.iam.gserviceaccount.com" roles/cloudsql.client
grant "fastapi-inference-sa@${PROJECT}.iam.gserviceaccount.com" roles/storage.objectViewer
grant "deepvogue-deployer-sa@${PROJECT}.iam.gserviceaccount.com" roles/artifactregistry.writer
grant "deepvogue-deployer-sa@${PROJECT}.iam.gserviceaccount.com" roles/run.admin
grant "deepvogue-deployer-sa@${PROJECT}.iam.gserviceaccount.com" roles/iam.serviceAccountUser
grant "deepvogue-deployer-sa@${PROJECT}.iam.gserviceaccount.com" roles/cloudsql.client

# WIF pool + provider for GitHub Actions
POOL="github-pool"
PROVIDER="github-provider"
if ! gcloud --project "$PROJECT" iam workload-identity-pools describe "$POOL" \
     --location=global >/dev/null 2>&1; then
  gcloud --project "$PROJECT" iam workload-identity-pools create "$POOL" --location=global
fi
if ! gcloud --project "$PROJECT" iam workload-identity-pools providers describe "$PROVIDER" \
     --location=global --workload-identity-pool="$POOL" >/dev/null 2>&1; then
  gcloud --project "$PROJECT" iam workload-identity-pools providers create-oidc "$PROVIDER" \
    --location=global --workload-identity-pool="$POOL" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
    --attribute-condition="attribute.repository=='${GH_REPO}'"
fi

# Allow the GitHub repo to impersonate the deployer SA
gcloud --project "$PROJECT" iam service-accounts add-iam-policy-binding \
  "deepvogue-deployer-sa@${PROJECT}.iam.gserviceaccount.com" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/attribute.repository/${GH_REPO}"

echo
echo "Add these as GitHub repo secrets:"
echo "  GCP_PROJECT=${PROJECT}"
echo "  GCP_WIF_PROVIDER=projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/providers/${PROVIDER}"
echo "  GCP_DEPLOYER_SA=deepvogue-deployer-sa@${PROJECT}.iam.gserviceaccount.com"
```

- [ ] **Step 2: Make exec + syntax check + commit**

```bash
chmod +x infra/gcp/setup-iam.sh
bash -n infra/gcp/setup-iam.sh
git add infra/gcp/setup-iam.sh
git commit -m "feat(gcp): runtime SAs + Workload Identity Federation for GitHub"
```

### Task 6.4: `infra/cloudrun/*.yaml` service specs

**Files:**
- Create: `infra/cloudrun/inference.service.yaml`, `mlflow.service.yaml`, `prefect-server.service.yaml`, `prefect-worker.job.yaml`

- [ ] **Step 1: Write inference spec**

`infra/cloudrun/inference.service.yaml`:
```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: deepvogue-inference
  annotations:
    run.googleapis.com/launch-stage: BETA
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/startup-cpu-boost: "true"
        run.googleapis.com/vpc-access-connector: deepvogue-vpc
        run.googleapis.com/execution-environment: gen2
        autoscaling.knative.dev/minScale: "0"
        autoscaling.knative.dev/maxScale: "2"
    spec:
      serviceAccountName: fastapi-inference-sa@PROJECT_ID.iam.gserviceaccount.com
      containerConcurrency: 1
      timeoutSeconds: 600
      containers:
        - image: us-central1-docker.pkg.dev/PROJECT_ID/deepvogue/inference:latest
          ports:
            - containerPort: 8080
          resources:
            limits:
              cpu: "4"
              memory: 16Gi
              nvidia.com/gpu: "1"
          env:
            - name: DV_ARTIFACT_BACKEND
              value: gcs
            - name: DV_MODELS_YAML
              value: gs://deepvogue-models/models.yaml
            - name: DV_MODELS_ROOT
              value: gs://deepvogue-models
            - name: MLFLOW_TRACKING_URI
              value: https://deepvogue-mlflow-PROJECT_HASH-uc.a.run.app
          nodeSelector:
            run.googleapis.com/accelerator: nvidia-l4
```

- [ ] **Step 2: Write `infra/cloudrun/mlflow.service.yaml`**

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: deepvogue-mlflow
  annotations:
    run.googleapis.com/launch-stage: GA
    run.googleapis.com/ingress: all
    run.googleapis.com/iap: enabled
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/vpc-access-connector: deepvogue-vpc
        run.googleapis.com/vpc-access-egress: private-ranges-only
        autoscaling.knative.dev/minScale: "0"
        autoscaling.knative.dev/maxScale: "2"
    spec:
      serviceAccountName: mlflow-server-sa@PROJECT_ID.iam.gserviceaccount.com
      containerConcurrency: 80
      timeoutSeconds: 300
      containers:
        - image: us-central1-docker.pkg.dev/PROJECT_ID/deepvogue/mlflow:latest
          ports:
            - containerPort: 5000
          resources:
            limits:
              cpu: "1"
              memory: 2Gi
          env:
            - name: MLFLOW_BACKEND_STORE_URI
              valueFrom:
                secretKeyRef:
                  name: deepvogue-mlflow-db-uri
                  key: latest
            - name: MLFLOW_ARTIFACTS_DESTINATION
              value: gs://deepvogue-mlflow
```

- [ ] **Step 3: Write `infra/cloudrun/prefect-server.service.yaml`**

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: deepvogue-prefect-server
  annotations:
    run.googleapis.com/launch-stage: GA
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/vpc-access-connector: deepvogue-vpc
        run.googleapis.com/vpc-access-egress: private-ranges-only
        autoscaling.knative.dev/minScale: "0"
        autoscaling.knative.dev/maxScale: "1"
    spec:
      serviceAccountName: prefect-server-sa@PROJECT_ID.iam.gserviceaccount.com
      containerConcurrency: 40
      timeoutSeconds: 300
      containers:
        - image: us-central1-docker.pkg.dev/PROJECT_ID/deepvogue/prefect-server:latest
          ports:
            - containerPort: 4200
          resources:
            limits:
              cpu: "1"
              memory: 2Gi
          env:
            - name: PREFECT_SERVER_API_HOST
              value: 0.0.0.0
            - name: PREFECT_SERVER_DATABASE_CONNECTION_URL
              valueFrom:
                secretKeyRef:
                  name: deepvogue-prefect-db-uri
                  key: latest
```

- [ ] **Step 4: Write `infra/cloudrun/prefect-worker.job.yaml`**

```yaml
apiVersion: run.googleapis.com/v1
kind: Job
metadata:
  name: deepvogue-prefect-worker
spec:
  template:
    spec:
      taskCount: 1
      template:
        spec:
          serviceAccountName: prefect-worker-sa@PROJECT_ID.iam.gserviceaccount.com
          maxRetries: 1
          timeoutSeconds: 3600
          containers:
            - image: us-central1-docker.pkg.dev/PROJECT_ID/deepvogue/prefect-worker:latest
              resources:
                limits:
                  cpu: "1"
                  memory: 2Gi
              env:
                - name: PREFECT_API_URL
                  value: https://deepvogue-prefect-server-PROJECT_HASH-uc.a.run.app/api
                - name: DV_ARTIFACT_BACKEND
                  value: gcs
              command: ["sh", "-c", "prefect worker start --pool default-process-pool --type process"]
```

Note: `PROJECT_HASH` placeholder is replaced post-deploy by the engineer (or via a second `sed` step) once the Cloud Run service URL is known.

- [ ] **Step 3: Add `make deploy-*` targets**

Append to `Makefile`:
```makefile
.PHONY: deploy-inference deploy-mlflow deploy-prefect gcp-setup publish

GCP_REGION ?= us-central1
GCP_AR := $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/deepvogue

publish: ## Publish latest Drive snapshot for DV_DATASET_NAME to DV_PUBLISH_TARGET as MODEL_ID
	@test -n "$(MODEL_ID)" || (echo "usage: make publish MODEL_ID=<id> [DV_DATASET_NAME=<name>]" && exit 1)
	python -m deepVogue.publish --model-id=$(MODEL_ID) \
	  --src-dir=$${DV_DRIVE_SYNC:?set DV_DRIVE_SYNC}/$${DV_DATASET_NAME:-default}

deploy-inference:
	@test -n "$(GCP_PROJECT)" || (echo "set GCP_PROJECT" && exit 1)
	sed "s|PROJECT_ID|$(GCP_PROJECT)|g" infra/cloudrun/inference.service.yaml | \
	  gcloud --project=$(GCP_PROJECT) run services replace - --region=$(GCP_REGION)

deploy-mlflow:
	sed "s|PROJECT_ID|$(GCP_PROJECT)|g" infra/cloudrun/mlflow.service.yaml | \
	  gcloud --project=$(GCP_PROJECT) run services replace - --region=$(GCP_REGION)

deploy-prefect:
	sed "s|PROJECT_ID|$(GCP_PROJECT)|g" infra/cloudrun/prefect-server.service.yaml | \
	  gcloud --project=$(GCP_PROJECT) run services replace - --region=$(GCP_REGION)
	sed "s|PROJECT_ID|$(GCP_PROJECT)|g" infra/cloudrun/prefect-worker.job.yaml | \
	  gcloud --project=$(GCP_PROJECT) run jobs replace - --region=$(GCP_REGION)

gcp-setup:
	bash infra/gcp/setup.sh
	bash infra/gcp/setup-sql.sh
	bash infra/gcp/setup-iam.sh
```

- [ ] **Step 4: Commit**

```bash
git add infra/cloudrun/ Makefile
git commit -m "feat(gcp): Cloud Run service specs + deploy-* make targets"
```

---

## Phase 7 — CI/CD + Slack (90 min)

### Task 7.1: `.github/workflows/test.yml`

**Files:**
- Create: `.github/workflows/test.yml`

- [ ] **Step 1: Write workflow**

```yaml
name: tests

on:
  push:
    branches: [master]
  pull_request:

jobs:
  pytest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install
        run: |
          pip install -e .
          pip install -r requirements-orchestration.txt
          pip install pytest black
      - name: Build stub state dict
        run: python scripts/build_stub_state_dict.py
      - name: Lint
        run: black --check deepVogue tests
      - name: Test
        run: pytest -v -m "not gpu"
      - name: Notify Slack on failure
        if: failure()
        uses: slackapi/slack-github-action@v1.27.0
        with:
          payload: '{"text":":x: tests failed on ${{ github.ref }} ${{ github.sha }} — ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}"}'
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/test.yml
git commit -m "ci: pytest + black workflow with Slack failure notify"
```

### Task 7.2: `.github/workflows/build-inference.yml`

**Files:**
- Create: `.github/workflows/build-inference.yml`

- [ ] **Step 1: Write workflow**

```yaml
name: build-inference

on:
  push:
    branches: [master]
    paths:
      - 'deepVogue/serve/**'
      - 'infra/docker/inference/**'
      - 'requirements.txt'
      - 'requirements-serve.txt'
      - 'infra/cloudrun/inference.service.yaml'
  workflow_dispatch:

permissions:
  contents: read
  id-token: write

jobs:
  build-deploy:
    runs-on: ubuntu-latest
    env:
      REGION: us-central1
      IMAGE: us-central1-docker.pkg.dev/${{ secrets.GCP_PROJECT }}/deepvogue/inference
    steps:
      - uses: actions/checkout@v4

      - id: auth
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.GCP_WIF_PROVIDER }}
          service_account: ${{ secrets.GCP_DEPLOYER_SA }}

      - uses: google-github-actions/setup-gcloud@v2

      - name: Slack start
        uses: slackapi/slack-github-action@v1.27.0
        with:
          payload: '{"text":":arrows_counterclockwise: building inference @ ${{ github.sha }}"}'
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}

      - name: Configure docker for AR
        run: gcloud auth configure-docker us-central1-docker.pkg.dev --quiet

      - name: Build image
        run: docker build -f infra/docker/inference/Dockerfile -t $IMAGE:${{ github.sha }} -t $IMAGE:latest .

      - name: Push image
        run: |
          docker push $IMAGE:${{ github.sha }}
          docker push $IMAGE:latest

      - name: Deploy Cloud Run
        run: |
          sed "s|PROJECT_ID|${{ secrets.GCP_PROJECT }}|g; s|:latest|:${{ github.sha }}|g" infra/cloudrun/inference.service.yaml | \
            gcloud run services replace - --region=$REGION --project=${{ secrets.GCP_PROJECT }}

      - name: Health smoke
        id: smoke
        run: |
          URL=$(gcloud run services describe deepvogue-inference --region=$REGION --project=${{ secrets.GCP_PROJECT }} --format='value(status.url)')
          for i in 1 2 3 4 5; do
            if curl -fsS "$URL/health" >/dev/null; then
              echo "url=$URL" >> $GITHUB_OUTPUT
              exit 0
            fi
            sleep 5
          done
          exit 1

      - name: Slack success
        if: success()
        uses: slackapi/slack-github-action@v1.27.0
        with:
          payload: '{"text":":white_check_mark: inference deployed — ${{ steps.smoke.outputs.url }}"}'
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}

      - name: Slack fail
        if: failure()
        uses: slackapi/slack-github-action@v1.27.0
        with:
          payload: '{"text":":x: inference build/deploy failed — ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}"}'
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/build-inference.yml
git commit -m "ci: build-inference — WIF auth, AR push, Cloud Run deploy, Slack"
```

### Task 7.3: `.github/workflows/build-mlflow.yml`

**Files:**
- Create: `.github/workflows/build-mlflow.yml`

- [ ] **Step 1: Write workflow**

```yaml
name: build-mlflow

on:
  push:
    branches: [master]
    paths:
      - 'infra/docker/mlflow/**'
      - 'infra/cloudrun/mlflow.service.yaml'
  workflow_dispatch:

permissions:
  contents: read
  id-token: write

jobs:
  build-deploy:
    runs-on: ubuntu-latest
    env:
      REGION: us-central1
      IMAGE: us-central1-docker.pkg.dev/${{ secrets.GCP_PROJECT }}/deepvogue/mlflow
    steps:
      - uses: actions/checkout@v4
      - id: auth
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.GCP_WIF_PROVIDER }}
          service_account: ${{ secrets.GCP_DEPLOYER_SA }}
      - uses: google-github-actions/setup-gcloud@v2
      - run: gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
      - name: Build & push
        run: |
          docker build -f infra/docker/mlflow/Dockerfile -t $IMAGE:${{ github.sha }} -t $IMAGE:latest infra/docker/mlflow
          docker push $IMAGE:${{ github.sha }}
          docker push $IMAGE:latest
      - name: Deploy
        run: |
          sed "s|PROJECT_ID|${{ secrets.GCP_PROJECT }}|g; s|:latest|:${{ github.sha }}|g" infra/cloudrun/mlflow.service.yaml | \
            gcloud run services replace - --region=$REGION --project=${{ secrets.GCP_PROJECT }}
      - name: Smoke
        id: smoke
        run: |
          URL=$(gcloud run services describe deepvogue-mlflow --region=$REGION --project=${{ secrets.GCP_PROJECT }} --format='value(status.url)')
          for i in 1 2 3 4 5; do curl -fsS "$URL" >/dev/null && { echo "url=$URL" >> $GITHUB_OUTPUT; exit 0; }; sleep 5; done
          exit 1
      - if: success()
        uses: slackapi/slack-github-action@v1.27.0
        with: { payload: '{"text":":white_check_mark: mlflow deployed — ${{ steps.smoke.outputs.url }}"}' }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
      - if: failure()
        uses: slackapi/slack-github-action@v1.27.0
        with: { payload: '{"text":":x: mlflow build/deploy failed — ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}"}' }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/build-mlflow.yml
git commit -m "ci: build-mlflow — AR push, Cloud Run deploy, Slack"
```

### Task 7.4: `.github/workflows/build-prefect.yml`

**Files:**
- Create: `.github/workflows/build-prefect.yml`

- [ ] **Step 1: Write workflow (builds server + worker images, deploys both)**

```yaml
name: build-prefect

on:
  push:
    branches: [master]
    paths:
      - 'infra/docker/prefect/**'
      - 'infra/cloudrun/prefect-server.service.yaml'
      - 'infra/cloudrun/prefect-worker.job.yaml'
      - 'deepVogue/orchestration/**'
  workflow_dispatch:

permissions:
  contents: read
  id-token: write

jobs:
  build-deploy:
    runs-on: ubuntu-latest
    env:
      REGION: us-central1
      SERVER_IMAGE: us-central1-docker.pkg.dev/${{ secrets.GCP_PROJECT }}/deepvogue/prefect-server
      WORKER_IMAGE: us-central1-docker.pkg.dev/${{ secrets.GCP_PROJECT }}/deepvogue/prefect-worker
    steps:
      - uses: actions/checkout@v4
      - id: auth
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.GCP_WIF_PROVIDER }}
          service_account: ${{ secrets.GCP_DEPLOYER_SA }}
      - uses: google-github-actions/setup-gcloud@v2
      - run: gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
      - name: Build server
        run: |
          docker build -f infra/docker/prefect/Dockerfile --build-arg TARGET=server \
            -t $SERVER_IMAGE:${{ github.sha }} -t $SERVER_IMAGE:latest .
          docker push $SERVER_IMAGE:${{ github.sha }}
          docker push $SERVER_IMAGE:latest
      - name: Build worker
        run: |
          docker build -f infra/docker/prefect/Dockerfile --build-arg TARGET=worker \
            -t $WORKER_IMAGE:${{ github.sha }} -t $WORKER_IMAGE:latest .
          docker push $WORKER_IMAGE:${{ github.sha }}
          docker push $WORKER_IMAGE:latest
      - name: Deploy server
        run: |
          sed "s|PROJECT_ID|${{ secrets.GCP_PROJECT }}|g; s|prefect-server:latest|prefect-server:${{ github.sha }}|g" infra/cloudrun/prefect-server.service.yaml | \
            gcloud run services replace - --region=$REGION --project=${{ secrets.GCP_PROJECT }}
      - name: Deploy worker job
        run: |
          sed "s|PROJECT_ID|${{ secrets.GCP_PROJECT }}|g; s|prefect-worker:latest|prefect-worker:${{ github.sha }}|g" infra/cloudrun/prefect-worker.job.yaml | \
            gcloud run jobs replace - --region=$REGION --project=${{ secrets.GCP_PROJECT }}
      - if: success()
        uses: slackapi/slack-github-action@v1.27.0
        with: { payload: '{"text":":white_check_mark: prefect server + worker deployed @ ${{ github.sha }}"}' }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
      - if: failure()
        uses: slackapi/slack-github-action@v1.27.0
        with: { payload: '{"text":":x: prefect build/deploy failed — ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}"}' }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/build-prefect.yml
git commit -m "ci: build-prefect — server + worker images, deploy both"
```

### Task 7.5: `.github/workflows/nano-smoke.yml`

**Files:**
- Create: `.github/workflows/nano-smoke.yml`

- [ ] **Step 1: Write workflow**

```yaml
name: nano-smoke

on:
  pull_request:
  schedule:
    - cron: "0 5 * * *"  # 05:00 UTC daily
  workflow_dispatch:

jobs:
  smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install
        run: |
          pip install -e . -r requirements-orchestration.txt pytest
      - name: Build stub state dict
        run: python scripts/build_stub_state_dict.py
      - name: Prepare env
        run: cp infra/.env.example infra/.env
      - name: nano-up
        run: make nano-up
      - name: nano-smoke
        run: make nano-smoke
      - name: nano-down
        if: always()
        run: make nano-down
      - name: Slack on fail
        if: failure()
        uses: slackapi/slack-github-action@v1.27.0
        with:
          payload: '{"text":":x: nano-smoke failed — ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}"}'
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/nano-smoke.yml
git commit -m "ci: nano-smoke nightly + PR — docker-compose pipeline end-to-end"
```

### Task 7.6: `.github/workflows/build-train.yml` (manual only)

**Files:**
- Create: `.github/workflows/build-train.yml`, `infra/docker/train/Dockerfile`

- [ ] **Step 1: Write training Dockerfile (scaffolded; not built in v1 unless dispatched)**

`infra/docker/train/Dockerfile`:
```dockerfile
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
```

- [ ] **Step 2: Write manual workflow**

```yaml
name: build-train

on:
  workflow_dispatch:

permissions:
  contents: read
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    env:
      IMAGE: us-central1-docker.pkg.dev/${{ secrets.GCP_PROJECT }}/deepvogue/train
    steps:
      - uses: actions/checkout@v4
      - id: auth
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.GCP_WIF_PROVIDER }}
          service_account: ${{ secrets.GCP_DEPLOYER_SA }}
      - uses: google-github-actions/setup-gcloud@v2
      - run: gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
      - run: docker build -f infra/docker/train/Dockerfile -t $IMAGE:${{ github.sha }} -t $IMAGE:latest .
      - run: docker push $IMAGE:${{ github.sha }} && docker push $IMAGE:latest
      - if: failure()
        uses: slackapi/slack-github-action@v1.27.0
        with: { payload: '{"text":":x: build-train failed"}' }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

- [ ] **Step 3: Commit**

```bash
git add infra/docker/train/Dockerfile .github/workflows/build-train.yml
git commit -m "ci: build-train (manual) — Dockerfile scaffolded for v2 RunPod backend"
```

---

## Phase 8 — Documentation (30 min)

### Task 8.1: CLAUDE.md — add "MLOps stack" section

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add the section**

Insert after the "Serving & bot" section:

```markdown
## MLOps stack

Local-nano stack (docker-compose) mirrors the prod-GCP stack one-for-one. Same code, same `models.yaml`, same flows — only the artifact backend (`DV_ARTIFACT_BACKEND=s3` → MinIO vs `gcs` → GCS) and endpoint URLs change.

| Env var | Purpose |
|---|---|
| `DV_ARTIFACT_BACKEND` | `s3` (MinIO local), `gcs` (prod), `memory` (tests), `file` (default) |
| `DV_S3_ENDPOINT_URL` | MinIO URL when backend=s3 |
| `DV_MODELS_YAML` | Full URI to `models.yaml` (e.g. `gs://deepvogue-models/models.yaml`) |
| `DV_MODELS_ROOT` | Resolution root for relative `pkl:` entries (back-compat) |
| `DV_PUBLISH_TARGET` | Where `make publish` uploads (e.g. `gs://deepvogue-models`) |
| `MLFLOW_TRACKING_URI` | MLflow server URL (Cloud Run in prod) |
| `MLFLOW_TRACKING_TOKEN` | IAP id_token for programmatic access from Colab |
| `PREFECT_API_URL` | Prefect server API URL |
| `SLACK_WEBHOOK_URL` | Optional; if unset, Slack helpers no-op |

GCS bucket layout (prod): `gs://deepvogue-{models,datasets,walks,mlflow,queue}`. Artifact Registry: `us-central1-docker.pkg.dev/<project>/deepvogue/{inference,mlflow,prefect-server,prefect-worker,train}`.

New Makefile targets:
- `make nano-up` / `nano-down` / `nano-logs` — local docker-compose stack
- `make nano-smoke` — full pipeline against the local stack
- `make publish MODEL_ID=...` — Drive → GCS + `models.yaml` append + Slack
- `make gcp-setup` — runs `infra/gcp/setup.sh` + setup-sql.sh + setup-iam.sh
- `make deploy-inference` / `deploy-mlflow` / `deploy-prefect` — Cloud Run deploy

Prefect flows live in `deepVogue/orchestration/flows.py`. Each task dispatches via `get_backend(backend)`; v1 supports `backend="local"` (real `prepare`/`publish`, mock `train`/`project`/`walk`/`eval`). `colab` and `runpod` backends are scaffolded but raise `NotImplementedError` in v1 — manual Colab + `make publish` is the v1 training path.

Colab logs to remote MLflow via `deepVogue/tracking/mlflow_helpers.py::log_training_run`. IAP auth: notebook cell sets `MLFLOW_TRACKING_TOKEN` from `gcloud auth print-identity-token --audiences=<iap-oauth-client-id>`.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude.md): MLOps stack section — env vars, make targets, flow topology"
```

### Task 8.2: README.md — nano-mode quickstart

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Append section**

```markdown
## Local nano-mode (MLOps stack)

Bring up an end-to-end stack on your laptop in two commands:

```bash
cp infra/.env.example infra/.env
make nano-up        # postgres + minio + mlflow + prefect + fastapi
make nano-smoke     # run pipeline_stills end-to-end (mocked train/walk on Mac)
make nano-down
```

Endpoints: MLflow http://localhost:5000 · Prefect http://localhost:4200 · FastAPI http://localhost:8080 · MinIO http://localhost:9001

## GCP deployment

```bash
export GCP_PROJECT=your-project GITHUB_REPO=owner/deepvogue
make gcp-setup                      # APIs, buckets, AR repo, Cloud SQL, runtime SAs, WIF
make deploy-mlflow deploy-prefect deploy-inference
```

After the first GCP deployment, register your Workload Identity Federation outputs (printed at the end of `setup-iam.sh`) as GitHub repo secrets: `GCP_PROJECT`, `GCP_WIF_PROVIDER`, `GCP_DEPLOYER_SA`, `SLACK_WEBHOOK_URL`.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs(readme): local nano-mode + GCP deployment quickstart"
```

### Task 8.3: `../../DOCS.md` — root catalog update

**Files:**
- Modify: `/Users/juan-garassino/Code/005-products/DOCS.md`

- [ ] **Step 1: Locate deepVogue entry**

```bash
grep -n "deepVogue" ../../DOCS.md
```

- [ ] **Step 2: Update the entry**

Surgically replace the existing one-liner description with text that mentions:
- "MLflow + Prefect + Cloud Run inference deployment"
- "Local docker-compose mirror for fast iteration"

Keep size compatible with the rest of the catalog.

- [ ] **Step 3: Commit**

```bash
git add ../../DOCS.md
git commit -m "docs(catalog): deepVogue entry — MLOps stack mention"
```

---

## Self-Review (engineer should re-run after completing all tasks)

Run the full test suite + lint:
```bash
pytest -v
black --check deepVogue tests
```
Expected: all green.

Run the nano smoke end-to-end:
```bash
make nano-up && make nano-smoke && make nano-down
```

Verify the new repo structure matches the spec File Structure table.

Open MLflow UI at http://localhost:5000 during a nano run — confirm experiments appear, params logged.

---

## v2 backlog (NOT implemented in this plan)

- `backends/colab.py` — GCS-based training_request queue, Prefect Suspended/Resumed states, polling cell in `notebooks/deepVogue_colab.ipynb`
- `backends/runpod.py` — RunPod GraphQL API: spin up training pod with `train:latest` image, `gsutil cp` outputs, terminate
- Frontend web UI
- Auto-publish on Drive snapshot rotation
- Cloud Monitoring alert policies → Slack
- BigQuery billing export + Looker Studio cost dashboard
