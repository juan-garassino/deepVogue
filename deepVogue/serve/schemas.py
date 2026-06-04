"""Pydantic request/response schemas for the FastAPI server."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ModelEntry(BaseModel):
    id: str
    backbone: str = "sg3-t"
    pkl: str
    dataset_kind: str = "stills"  # 'stills' | 'frames'
    default_trunc: float = 0.7
    factors: Optional[str] = None
    anchors_dir: Optional[str] = None
    walks_dir: Optional[str] = None  # override; default: <unscoped DV_WALKS_DIR>/<id>
    pkl_resolved: Optional[str] = None  # derived: absolute URI after resolve_uri()

    def model_post_init(self, __context) -> None:
        """Compute `pkl_resolved` from `pkl` using DV_MODELS_ROOT-aware resolver."""
        if self.pkl_resolved is None:
            from deepVogue.clients import resolve_uri

            self.pkl_resolved = resolve_uri(self.pkl)

    def __getitem__(self, key: str):
        """Dict-style read access for registry consumers (`entry["pkl_resolved"]`)."""
        try:
            return getattr(self, key)
        except AttributeError as e:
            raise KeyError(key) from e


class GenerateRequest(BaseModel):
    model: str
    seed: int = 0
    trunc: Optional[float] = None
    factor_idx: Optional[int] = None
    factor_amp: float = 0.0


class WalkRequest(BaseModel):
    model: str
    seeds: List[int] = Field(
        default_factory=list, description="Start/intermediate/end seeds; ≥2."
    )
    steps: int = 24
    fps: int = 24
    mode: str = "cubic"  # linear|slerp|bezier|cubic
    trunc: Optional[float] = None
