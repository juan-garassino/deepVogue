"""scripts/submit_train.py — thin-launcher contract + dispatch."""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "submit_train.py"


def _run(args, env_extra=None):
    import os
    env = {k: v for k, v in os.environ.items() if not k.startswith("DV_")}
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, env=env, cwd=str(REPO),
    )


def test_missing_run_uri_emits_error_json():
    res = _run(["--backend=runpod", "--model-id=t"])
    assert res.returncode == 2
    assert json.loads(res.stdout.splitlines()[-1]) == {
        "event": "error", "message": "DV_RUN_URI is required",
    }


def test_missing_backend_env_fails_cleanly():
    res = _run(["--backend=vertex", "--model-id=t"], {"DV_RUN_URI": "gs://x/r"})
    assert res.returncode == 1
    events = [json.loads(line) for line in res.stdout.splitlines()]
    assert events[0]["event"] == "submitting" and events[0]["backend"] == "vertex"
    assert events[-1]["event"] == "error"


def test_invalid_backend_rejected_by_argparse():
    res = _run(["--backend=colab", "--model-id=t"])
    assert res.returncode == 2
    assert "invalid choice" in res.stderr
