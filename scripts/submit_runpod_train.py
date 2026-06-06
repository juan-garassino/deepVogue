#!/usr/bin/env python
"""Submit a RunPod training job for deepVogue.

Reads gs:// URIs and RunPod creds from the environment (see infra/.env.example);
prints a JSON line per lifecycle event to stdout so callers can pipe / parse.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepVogue.orchestration.backends.runpod import train as runpod_train  # noqa: E402


def _emit(event: str, **fields):
    sys.stdout.write(json.dumps({"event": event, **fields}) + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="submit_runpod_train")
    p.add_argument("--model-id", required=True)
    p.add_argument("--cfg", default=os.environ.get("DV_CFG", "stylegan3-t"))
    p.add_argument("--kimg", type=int, default=int(os.environ.get("DV_KIMG", "5000")))
    p.add_argument("--gamma", type=float, default=float(os.environ.get("DV_GAMMA", "2")))
    p.add_argument("--batch", type=int, default=int(os.environ.get("DV_BATCH", "32")))
    p.add_argument("--res", type=int, default=int(os.environ.get("DV_RES", "256")))
    p.add_argument("--resume-from", default=os.environ.get("DV_RESUME_FROM"))
    args = p.parse_args(argv)

    run_uri = os.environ.get("DV_RUN_URI")
    if not run_uri:
        _emit("error", message="DV_RUN_URI is required")
        return 2

    _emit(
        "submitting",
        model_id=args.model_id,
        cfg=args.cfg,
        kimg=args.kimg,
        res=args.res,
    )
    try:
        out = runpod_train(
            dataset_name=args.model_id,
            cfg=args.cfg,
            kimg=args.kimg,
            gamma=args.gamma,
            batch=args.batch,
            res=args.res,
            target_uri=run_uri,
            resume_from=args.resume_from,
        )
    except Exception as e:
        _emit("error", message=str(e))
        return 1
    _emit("done", **{k: v for k, v in out.items() if not isinstance(v, bytes)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
