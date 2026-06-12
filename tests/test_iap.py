"""Tests for the IAP id-token helper."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from deepVogue.tracking import iap


@pytest.fixture(autouse=True)
def _clear_token(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
    monkeypatch.delenv("IAP_OAUTH_CLIENT_ID", raising=False)


def _make_response(text="canned-token"):
    r = MagicMock()
    r.text = text
    r.raise_for_status.return_value = None
    return r


def test_fetch_iap_id_token_hits_metadata_server():
    with patch(
        "deepVogue.tracking.iap.requests.get", return_value=_make_response()
    ) as g:
        token = iap.fetch_iap_id_token("iap-client-abc")
    assert token == "canned-token"
    g.assert_called_once()
    args, kwargs = g.call_args
    assert args[0] == iap.METADATA_URL
    assert kwargs["headers"] == iap.METADATA_HEADERS
    assert kwargs["params"]["audience"] == "iap-client-abc"


def test_start_refresher_noop_without_env(monkeypatch):
    assert iap.start_iap_token_refresher() is None


def test_start_refresher_uses_env_audience(monkeypatch):
    monkeypatch.setenv("IAP_OAUTH_CLIENT_ID", "iap-client-xyz")
    fired = threading.Event()

    def fake_fetch(audience, timeout_s=5.0):
        assert audience == "iap-client-xyz"
        fired.set()
        return "tok-from-env"

    with patch("deepVogue.tracking.iap.fetch_iap_id_token", side_effect=fake_fetch):
        t = iap.start_iap_token_refresher(interval_s=1)
        assert t is not None
        assert fired.wait(timeout=2.0)
        # Stop the daemon thread so it doesn't keep polling during the next test.
        t._stop_event.set()
        t.join(timeout=2.0)
    import os

    assert os.environ.get("MLFLOW_TRACKING_TOKEN") == "tok-from-env"


def test_refresher_survives_transient_failures(monkeypatch):
    monkeypatch.setenv("IAP_OAUTH_CLIENT_ID", "iap-client-xyz")
    calls = []
    fired = threading.Event()

    def flaky(audience, timeout_s=5.0):
        calls.append(audience)
        if len(calls) == 1:
            raise RuntimeError("metadata transient")
        fired.set()
        return "tok-recovered"

    with patch("deepVogue.tracking.iap.fetch_iap_id_token", side_effect=flaky):
        t = iap.start_iap_token_refresher(interval_s=1)
        assert fired.wait(timeout=3.0)
        t._stop_event.set()
        t.join(timeout=2.0)
    import os

    assert os.environ.get("MLFLOW_TRACKING_TOKEN") == "tok-recovered"
