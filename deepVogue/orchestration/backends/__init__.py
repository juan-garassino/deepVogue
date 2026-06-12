from __future__ import annotations
from typing import Literal, Protocol

BackendName = Literal["local", "colab", "runpod", "vertex"]


class BackendOp(Protocol):
    def prepare(self, **kw): ...
    def train(self, **kw): ...
    def publish(self, **kw): ...
    def project(self, **kw): ...
    def walk(self, **kw): ...
    def eval(self, **kw): ...


def get_backend(name: str) -> BackendOp:
    # Lazy per-name imports: local.py pulls torch/PIL at module level, which a
    # thin launcher env (make runpod-train / vertex-train) must not need.
    name = name.lower()
    if name == "local":
        from . import local
        return local
    if name == "colab":
        from . import colab
        return colab
    if name == "runpod":
        from . import runpod
        return runpod
    if name == "vertex":
        from . import vertex
        return vertex
    raise ValueError(f"unknown backend: {name!r}")
