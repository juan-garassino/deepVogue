"""Colab backend — v2 stub.

Planned v2 design (not implemented):

  * The orchestrator (this module's ``train()``) writes a job JSON to
    ``gs://deepvogue-queue/pending/<job-id>.json`` with ``DV_*`` env values.
  * A long-running Colab notebook polls ``pending/``; on claim it moves the
    job to ``claimed/<job-id>.json`` (atomic rename via fsspec) and runs the
    training in-notebook against ``DV_DATA_DIR`` / ``DV_RUN_DIR``.
  * On completion the notebook writes ``done/<job-id>.json`` with the pkl URI
    and FID; the orchestrator polls ``done/`` to return.

For v1 the supported Colab workflow is **manual**: run the cells in
``notebooks/deepVogue_colab.ipynb`` directly, then ``make publish
MODEL_ID=<id>`` from the notebook to land the snapshot in ``models.yaml``.
RunPod (``backends/runpod.py``) is the autonomous-job path for v1.
"""

from __future__ import annotations

_MANUAL_HINT = (
    "colab.{op} — v2; for v1 run cells in notebooks/deepVogue_colab.ipynb, then "
    "`make publish MODEL_ID=<id>`. RunPod backend handles autonomous jobs."
)


def prepare(**kw):  # v2: handoff queue via gs://deepvogue-queue/{pending,claimed,done}/
    raise NotImplementedError(_MANUAL_HINT.format(op="prepare"))


def train(**kw):
    raise NotImplementedError(_MANUAL_HINT.format(op="train"))


def publish(**kw):
    raise NotImplementedError(_MANUAL_HINT.format(op="publish"))


def project(**kw):
    raise NotImplementedError(_MANUAL_HINT.format(op="project"))


def walk(**kw):
    raise NotImplementedError(_MANUAL_HINT.format(op="walk"))


def eval(**kw):
    raise NotImplementedError(_MANUAL_HINT.format(op="eval"))
