from unittest.mock import patch, MagicMock
import pytest
from deepVogue.notifications import slack


def test_no_op_without_webhook(monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    # should not raise, should not call requests
    with patch("deepVogue.notifications.slack.requests") as r:
        slack.notify_success("ci", "build done", {"sha": "abc"})
        r.post.assert_not_called()


def test_notify_success_posts(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/X")
    resp = MagicMock(status_code=200)
    with patch("deepVogue.notifications.slack.requests") as r:
        r.post.return_value = resp
        slack.notify_success("ci", "build done", {"sha": "abc"})
        r.post.assert_called_once()
        body = r.post.call_args.kwargs["json"]
        assert "blocks" in body
        text = str(body)
        assert "ci" in text
        assert "build done" in text
        assert "abc" in text


def test_notify_failure_includes_exception_summary(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/X")
    with patch("deepVogue.notifications.slack.requests") as r:
        r.post.return_value = MagicMock(status_code=200)
        try:
            raise RuntimeError("boom")
        except RuntimeError as e:
            slack.notify_failure("ci", "build failed", exc=e)
        text = str(r.post.call_args.kwargs["json"])
        assert "RuntimeError" in text
        assert "boom" in text


def test_swallows_non_2xx(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/X")
    with patch("deepVogue.notifications.slack.requests") as r:
        r.post.return_value = MagicMock(status_code=500, text="err")
        # must not raise
        slack.notify_success("ci", "ok")
