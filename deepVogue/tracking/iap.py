"""IAP id-token fetch + refresher for talking to IAP-fronted Cloud Run.

The GCE metadata server mints id-tokens scoped to a given audience
(the OAuth client id of the IAP-protected service). MLflow's Python
client reads ``MLFLOW_TRACKING_TOKEN`` and sends it as
``Authorization: Bearer <token>``, so refreshing this env var on a
background thread is enough to keep an IAP-fronted MLflow reachable
from the inference container.

Silent no-op when ``IAP_OAUTH_CLIENT_ID`` is unset, so local dev and
the nano stack keep working unchanged.
"""

from __future__ import annotations

import logging
import os
import threading
import time

import requests

log = logging.getLogger(__name__)

METADATA_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/"
    "service-accounts/default/identity"
)
METADATA_HEADERS = {"Metadata-Flavor": "Google"}
DEFAULT_REFRESH_S = 1800  # tokens are valid for 3600s; refresh halfway through


def fetch_iap_id_token(audience: str, timeout_s: float = 5.0) -> str:
    """GET an id-token from the GCE metadata server, bound to ``audience``."""
    r = requests.get(
        METADATA_URL,
        params={"audience": audience, "format": "full"},
        headers=METADATA_HEADERS,
        timeout=timeout_s,
    )
    r.raise_for_status()
    return r.text.strip()


def _refresh_loop(audience: str, interval_s: int, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            token = fetch_iap_id_token(audience)
            os.environ["MLFLOW_TRACKING_TOKEN"] = token
            log.info("iap: refreshed MLFLOW_TRACKING_TOKEN (audience=%s)", audience)
        except Exception as e:
            log.warning("iap: token refresh failed: %s", e)
        stop.wait(interval_s)


def start_iap_token_refresher(
    audience: str | None = None,
    *,
    interval_s: int = DEFAULT_REFRESH_S,
) -> threading.Thread | None:
    """Start a daemon thread that keeps MLFLOW_TRACKING_TOKEN fresh.

    Reads ``IAP_OAUTH_CLIENT_ID`` from env when ``audience`` is None.
    Returns ``None`` when no audience is configured (silent no-op).
    """
    audience = audience or os.environ.get("IAP_OAUTH_CLIENT_ID")
    if not audience:
        return None
    stop = threading.Event()
    t = threading.Thread(
        target=_refresh_loop,
        args=(audience, interval_s, stop),
        name="iap-token-refresher",
        daemon=True,
    )
    t._stop_event = stop  # type: ignore[attr-defined]
    t.start()
    log.info("iap: token refresher started (interval=%ds)", interval_s)
    return t
