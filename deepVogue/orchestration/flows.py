"""Prefect flows for the deepVogue pipeline. All tasks delegate to a backend module."""

from __future__ import annotations

import logging
from typing import Any

from prefect import flow, task

from deepVogue.clients import artifact_uri
from deepVogue.orchestration.backends import get_backend
from deepVogue.notifications import slack

log = logging.getLogger(__name__)


@task
def _prepare_task(*, backend: str, **kw):
    return get_backend(backend).prepare(**kw)


@task
def _train_task(*, backend: str, **kw):
    return get_backend(backend).train(**kw)


@task
def _publish_task(*, backend: str, **kw):
    return get_backend(backend).publish(**kw)


@task
def _project_task(*, backend: str, **kw):
    return get_backend(backend).project(**kw)


@task
def _walk_task(*, backend: str, **kw):
    return get_backend(backend).walk(**kw)


@task
def _eval_task(*, backend: str, **kw):
    return get_backend(backend).eval(**kw)


@flow(name="prepare_flow")
def prepare_flow(
    *,
    source_uri: str,
    dataset_name: str,
    res: int,
    target_uri: str,
    kind: str = "stills",
    fps: int | None = None,
    backend: str = "local",
) -> dict[str, Any]:
    out = _prepare_task(
        backend=backend,
        source_uri=source_uri,
        dataset_name=dataset_name,
        res=res,
        kind=kind,
        target_uri=target_uri,
        fps=fps,
    )
    slack.notify_success(
        "flow",
        f"prepare {dataset_name} done",
        {"dataset_uri": out["dataset_uri"], "n": str(out.get("n_images", 0))},
    )
    return out


@flow(name="train_flow")
def train_flow(
    *,
    dataset_name: str,
    cfg: str,
    kimg: int,
    gamma: float,
    batch: int,
    res: int = 256,
    target_uri: str | None = None,
    backend: str = "local",
    resume_from: str | None = None,
) -> dict[str, Any]:
    out = _train_task(
        backend=backend,
        dataset_name=dataset_name,
        cfg=cfg,
        kimg=kimg,
        gamma=gamma,
        batch=batch,
        res=res,
        target_uri=target_uri,
        resume_from=resume_from,
    )
    slack.notify_success(
        "flow",
        f"train {dataset_name} done",
        {"pkl": out["pkl"], "fid": str(out.get("fid"))},
    )
    return out


@flow(name="publish_flow")
def publish_flow(
    *,
    model_id: str,
    src_dir: str,
    backbone: str = "sg3-t",
    dataset_kind: str = "stills",
    default_trunc: float = 0.7,
    backend: str = "local",
) -> dict[str, Any]:
    return _publish_task(
        backend=backend,
        model_id=model_id,
        src_dir=src_dir,
        backbone=backbone,
        dataset_kind=dataset_kind,
        default_trunc=default_trunc,
    )


@flow(name="project_flow")
def project_flow(
    *,
    model_id: str,
    frames_uri: str,
    target_uri: str,
    stride: int = 4,
    steps: int = 500,
    backend: str = "local",
) -> dict[str, Any]:
    return _project_task(
        backend=backend,
        model_id=model_id,
        frames_uri=frames_uri,
        stride=stride,
        steps=steps,
        target_uri=target_uri,
    )


@flow(name="walk_flow")
def walk_flow(
    *,
    model_id: str,
    target_uri: str,
    steps: int = 60,
    fps: int = 24,
    seeds: list[int] | None = None,
    anchors_uri: str | None = None,
    mode: str = "cubic",
    backend: str = "local",
) -> dict[str, Any]:
    out = _walk_task(
        backend=backend,
        model_id=model_id,
        target_uri=target_uri,
        steps=steps,
        fps=fps,
        seeds=seeds,
        anchors_uri=anchors_uri,
        mode=mode,
    )
    slack.notify_success("flow", f"walk {model_id} done", {"walk_uri": out["walk_uri"]})
    return out


@flow(name="eval_flow")
def eval_flow(
    *, model_id: str, dataset_uri: str, backend: str = "local"
) -> dict[str, Any]:
    return _eval_task(backend=backend, model_id=model_id, dataset_uri=dataset_uri)


@flow(name="pipeline_stills")
def pipeline_stills(
    *,
    source_uri: str,
    dataset_name: str,
    model_id: str,
    res: int = 256,
    kimg: int = 5000,
    gamma: float = 2.0,
    batch: int = 32,
    walk_steps: int = 60,
    walk_fps: int = 24,
    backend: str = "local",
) -> dict[str, Any]:
    prep = prepare_flow(
        source_uri=source_uri,
        dataset_name=dataset_name,
        res=res,
        target_uri=artifact_uri("deepvogue-datasets"),
        backend=backend,
    )
    train = train_flow(
        dataset_name=dataset_name,
        cfg="stylegan3-t",
        kimg=kimg,
        gamma=gamma,
        batch=batch,
        res=res,
        target_uri=artifact_uri("deepvogue-models", dataset_name),
        backend=backend,
    )
    # publish requires a local directory; nano flow skips actual publish and just records URIs
    walk = walk_flow(
        model_id=model_id,
        target_uri=artifact_uri("deepvogue-walks"),
        steps=walk_steps,
        fps=walk_fps,
        backend=backend,
    )
    return {"prepare": prep, "train": train, "walk": walk, **walk}


@flow(name="pipeline_frames")
def pipeline_frames(
    *,
    source_uri: str,
    dataset_name: str,
    model_id: str,
    res: int = 256,
    fps: int = 1,
    kimg: int = 5000,
    gamma: float = 2.0,
    batch: int = 32,
    walk_steps: int = 60,
    walk_fps: int = 24,
    backend: str = "local",
) -> dict[str, Any]:
    prep = prepare_flow(
        source_uri=source_uri,
        dataset_name=dataset_name,
        res=res,
        kind="frames",
        fps=fps,
        target_uri=artifact_uri("deepvogue-datasets"),
        backend=backend,
    )
    train = train_flow(
        dataset_name=dataset_name,
        cfg="stylegan3-t",
        kimg=kimg,
        gamma=gamma,
        batch=batch,
        res=res,
        target_uri=artifact_uri("deepvogue-models", dataset_name),
        backend=backend,
    )
    project = project_flow(
        model_id=model_id,
        frames_uri=prep["dataset_uri"],
        target_uri=artifact_uri("deepvogue-anchors"),
        backend=backend,
    )
    walk = walk_flow(
        model_id=model_id,
        anchors_uri=project["anchors_uri"],
        target_uri=artifact_uri("deepvogue-walks"),
        steps=walk_steps,
        fps=walk_fps,
        backend=backend,
    )
    ev = eval_flow(model_id=model_id, dataset_uri=prep["dataset_uri"], backend=backend)
    return {
        "prepare": prep,
        "train": train,
        "project": project,
        "walk": walk,
        "eval": ev,
    }
