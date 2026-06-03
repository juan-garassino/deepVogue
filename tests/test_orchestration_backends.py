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
        b.train(
            dataset_name="x", cfg="stylegan3-t", kimg=10, gamma=2.0, batch=32, res=64
        )


def test_runpod_backend_raises_notimplemented_in_v1():
    b = get_backend("runpod")
    with pytest.raises(NotImplementedError):
        b.train(
            dataset_name="x", cfg="stylegan3-t", kimg=10, gamma=2.0, batch=32, res=64
        )
