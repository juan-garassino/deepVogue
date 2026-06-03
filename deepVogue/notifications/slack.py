"""Slack webhook notifier. Silent no-op if SLACK_WEBHOOK_URL unset."""
from __future__ import annotations

import logging
import os
import traceback
from typing import Mapping

import requests

log = logging.getLogger(__name__)


def _post(blocks: list) -> None:
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        return
    try:
        resp = requests.post(url, json={"blocks": blocks}, timeout=5)
        if resp.status_code // 100 != 2:
            log.warning("slack non-2xx: %s %s", resp.status_code, resp.text[:200])
    except requests.RequestException as e:
        log.warning("slack post failed: %s", e)


def _fields(meta: Mapping[str, str] | None) -> list:
    if not meta:
        return []
    return [
        {"type": "mrkdwn", "text": f"*{k}*\n{v}"} for k, v in list(meta.items())[:10]
    ]


def notify_event(channel_tag: str, level: str, headline: str, meta: Mapping[str, str] | None = None) -> None:
    emoji = {"success": ":white_check_mark:", "failure": ":x:", "info": ":information_source:"}.get(level, ":bell:")
    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": f"{emoji} *[{channel_tag}]* {headline}"}}
    ]
    fields = _fields(meta)
    if fields:
        blocks.append({"type": "section", "fields": fields})
    _post(blocks)


def notify_success(channel_tag: str, headline: str, meta: Mapping[str, str] | None = None) -> None:
    notify_event(channel_tag, "success", headline, meta)


def notify_failure(channel_tag: str, headline: str, meta: Mapping[str, str] | None = None, exc: BaseException | None = None) -> None:
    meta = dict(meta or {})
    if exc is not None:
        meta.setdefault("error", f"{type(exc).__name__}: {exc}")
        meta.setdefault("trace", "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-500:])
    notify_event(channel_tag, "failure", headline, meta)
