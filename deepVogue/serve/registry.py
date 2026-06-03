"""Hot-reloading model registry backed by ``models.yaml`` on Drive."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

from .schemas import ModelEntry


def _default_yaml_path() -> Path:
    """`$DV_MODELS_YAML` or `<DV_DRIVE_SYNC parent>/models.yaml` or `<DV_RUN_DIR parent>/models.yaml`."""
    if os.environ.get("DV_MODELS_YAML"):
        return Path(os.environ["DV_MODELS_YAML"])
    from deepVogue._paths import resolve

    p = resolve()
    base = p.drive_sync.parent if p.drive_sync else p.run_dir.parent
    return Path(base) / "models.yaml"


class Registry:
    def __init__(self, yaml_path: Optional[Path] = None):
        self.path: Path = yaml_path or _default_yaml_path()
        self._mtime: float = -1
        self._models: Dict[str, ModelEntry] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ load
    def _load_if_stale(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            with self._lock:
                self._models = {}
                self._mtime = -1
            return
        if mtime == self._mtime:
            return
        try:
            import yaml  # PyYAML
        except ImportError as e:
            raise RuntimeError("PyYAML required: pip install pyyaml") from e
        raw = yaml.safe_load(self.path.read_text()) or []
        if isinstance(raw, dict) and "models" in raw:
            raw = raw["models"]
        if not isinstance(raw, list):
            raise RuntimeError(
                f"{self.path}: expected list (or {{models: [...]}}) at top level"
            )
        models = {}
        for item in raw:
            entry = ModelEntry(**item)
            models[entry.id] = entry
        with self._lock:
            self._models = models
            self._mtime = mtime

    # ------------------------------------------------------------------ api
    def list(self) -> List[ModelEntry]:
        self._load_if_stale()
        with self._lock:
            return list(self._models.values())

    def get(self, model_id: str) -> ModelEntry:
        self._load_if_stale()
        with self._lock:
            if model_id not in self._models:
                raise KeyError(model_id)
            return self._models[model_id]

    # ---------------------------------------------------------------- write
    def append_entry(self, entry: ModelEntry) -> None:
        """Insert or update ``entry`` in ``models.yaml``.

        Atomic-ish: load → upsert → safe_dump back. Same-id entries are replaced.
        """
        try:
            import yaml
        except ImportError as e:
            raise RuntimeError("PyYAML required: pip install pyyaml") from e
        self._load_if_stale()
        with self._lock:
            current = dict(self._models)
        current[entry.id] = entry
        items = [m.model_dump(exclude_none=True) for m in current.values()]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(yaml.safe_dump(items, sort_keys=False, indent=2))
        # force next read to pick up the new mtime
        with self._lock:
            self._mtime = -1
