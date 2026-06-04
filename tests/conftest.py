"""Pytest bootstrap.

Forces a clean PREFECT_HOME for tests so the user's global profiles.toml
(which may contain settings from a newer Prefect minor version) does not
break in-process flow runs in CI / local pytest invocations.
"""

from __future__ import annotations

import os
import tempfile

_PREFECT_HOME = os.environ.get("PREFECT_HOME_OVERRIDE") or os.path.join(
    tempfile.gettempdir(), "prefect-deepvogue-tests"
)
os.environ.setdefault("PREFECT_HOME", _PREFECT_HOME)
os.environ.setdefault("PREFECT_TELEMETRY_ENABLED", "false")
os.makedirs(_PREFECT_HOME, exist_ok=True)
