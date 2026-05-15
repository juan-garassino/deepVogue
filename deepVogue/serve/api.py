"""FastAPI inference server — generate / walk / list / static films."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

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
        entry, seed=req.seed, trunc=req.trunc,
        factor_idx=req.factor_idx, factor_amp=req.factor_amp,
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
        entry, seeds=req.seeds, steps=req.steps, fps=req.fps,
        mode=req.mode, trunc=req.trunc,
    )
    return Response(content=mp4, media_type="video/mp4")


@app.get("/films/{model_id}/{walk_id}")
def films(model_id: str, walk_id: str):
    """Serve a precomputed walk mp4 from `<DV_WALKS_DIR>/<model_id>/<walk_id>.mp4`."""
    from deepVogue._paths import resolve
    p = resolve()
    candidate = (p.walks_dir / model_id / walk_id) if not walk_id.endswith(".mp4") else (
        p.walks_dir / model_id / walk_id
    )
    if candidate.suffix != ".mp4":
        candidate = candidate.with_suffix(".mp4")
    if not candidate.exists():
        raise HTTPException(404, f"film not found: {candidate}")
    return FileResponse(str(candidate), media_type="video/mp4", filename=candidate.name)
