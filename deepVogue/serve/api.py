"""FastAPI inference server — generate / walk / list / static films."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, FileResponse, JSONResponse

from . import loader
from .registry import Registry
from .schemas import GenerateRequest, WalkRequest


app = FastAPI(title="deepVogue inference", version="0.1.0")
_registry = Registry()


@app.on_event("startup")
def warm_models() -> None:
    """Pre-load the first two registered models so first request isn't cold."""
    try:
        entries = _registry.list()[:2]
    except Exception as e:
        print(f"[serve] registry not loadable yet ({e}); skipping warm-up")
        return
    for entry in entries:
        try:
            loader.load(entry)
            print(f"[serve] warm: {entry.id} ← {entry.pkl}")
        except Exception as e:
            print(f"[serve] warm failed for {entry.id}: {e}")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/status")
def status(model: str, last_n: int = 5) -> JSONResponse:
    """Tail metric-fid50k_full.jsonl for the registered model.

    Returns the last `last_n` FID rows and the highest-kimg snapshot path.
    `model` must be a registered id; the FID log is discovered as
    `<pkl-parent>/metric-fid50k_full.jsonl` via fsspec so it works for
    `s3://`, `gs://`, `memory://`, or plain local paths.
    """
    import json
    import fsspec
    from deepVogue._paths import latest_snapshot

    try:
        entry = _registry.get(model)
    except KeyError:
        raise HTTPException(404, f"unknown model: {model}")
    pkl_uri = entry.pkl_resolved or entry.pkl
    jsonl_uri = f"{pkl_uri.rsplit('/', 1)[0]}/metric-fid50k_full.jsonl"
    rows = []
    try:
        with fsspec.open(jsonl_uri, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except (FileNotFoundError, OSError):
        # No FID log yet (or unreachable backend) — return empty rows.
        rows = []
    latest = (
        latest_snapshot(entry.id)
        if entry.dataset_kind in ("stills", "frames")
        else None
    )
    return JSONResponse(
        {
            "model": model,
            "pkl": pkl_uri,
            "latest_snapshot": str(latest) if latest else None,
            "fid_rows": rows[-last_n:],
        }
    )


@app.get("/models")
def list_models() -> JSONResponse:
    return JSONResponse([m.model_dump() for m in _registry.list()])


@app.post("/generate")
def generate(req: GenerateRequest):
    try:
        entry = _registry.get(req.model)
    except KeyError:
        raise HTTPException(404, f"unknown model: {req.model}")
    png = loader.generate(
        entry,
        seed=req.seed,
        trunc=req.trunc,
        factor_idx=req.factor_idx,
        factor_amp=req.factor_amp,
    )
    return Response(content=png, media_type="image/png")


@app.post("/walk")
def walk(req: WalkRequest):
    try:
        entry = _registry.get(req.model)
    except KeyError:
        raise HTTPException(404, f"unknown model: {req.model}")
    if len(req.seeds) < 2:
        raise HTTPException(400, "walk needs ≥2 seeds")
    mp4 = loader.walk(
        entry,
        seeds=req.seeds,
        steps=req.steps,
        fps=req.fps,
        mode=req.mode,
        trunc=req.trunc,
    )
    return Response(content=mp4, media_type="video/mp4")


def _resolve_film_path(model_id: str, walk_id: str) -> Path:
    """Compute the on-disk path for a precomputed walk mp4.

    Priority:
      1. `entry.walks_dir` from the registry (absolute override).
      2. `$DV_WALKS_DIR/<model_id>/` — the **unscoped** env var so the films
         lookup is independent of whatever `DV_DATASET_NAME` the FastAPI
         process was launched with. `model_id` is the only key.
    """
    walks_root = None
    try:
        entry = _registry.get(model_id)
        if entry.walks_dir:
            walks_root = Path(entry.walks_dir)
    except KeyError:
        entry = None
    if walks_root is None:
        env = os.environ.get("DV_WALKS_DIR")
        walks_root = (
            (Path(env) / model_id) if env else (Path("/tmp/deepVogue/walks") / model_id)
        )
    candidate = walks_root / walk_id
    if candidate.suffix != ".mp4":
        candidate = candidate.with_suffix(".mp4")
    return candidate


@app.get("/films/{model_id}/{walk_id}")
def films(model_id: str, walk_id: str):
    """Serve a precomputed walk mp4. See `_resolve_film_path` for the rules."""
    candidate = _resolve_film_path(model_id, walk_id)
    if not candidate.exists():
        raise HTTPException(404, f"film not found: {candidate}")
    return FileResponse(str(candidate), media_type="video/mp4", filename=candidate.name)
