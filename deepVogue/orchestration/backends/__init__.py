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
    from . import local, colab, runpod, vertex

    name = name.lower()
    if name == "local":
        return local
    if name == "colab":
        return colab
    if name == "runpod":
        return runpod
    if name == "vertex":
        return vertex
    raise ValueError(f"unknown backend: {name!r}")
