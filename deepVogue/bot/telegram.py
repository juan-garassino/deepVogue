"""Telegram bot — talks to the FastAPI inference server over localhost.

Long-polling (no public URL needed for the bot). The FastAPI server should be
running at ``$DV_FASTAPI_URL`` (default ``http://127.0.0.1:8080``).

Env:
  DV_TG_TOKEN       Telegram bot token (BotFather).
  DV_TG_ALLOWLIST   Comma-separated Telegram user IDs allowed to use the bot.
  DV_FASTAPI_URL    Where the bot reaches the inference server.

Commands:
  /start
  /models                                       — list registry
  /gen <model> [seed=N] [trunc=0.7]             — single image
  /walk <model> seeds=A,B,C [steps=24] [fps=24] [mode=cubic] [trunc=0.7]
  /film <model> <walk_id>                       — fetch precomputed mp4
  /factor <model> seed=N idx=K [deg=0.4] [trunc=0.7]
"""

from __future__ import annotations

import asyncio
import os
import shlex
from typing import Dict, List, Set, Tuple


def _allowlist() -> Set[int]:
    raw = os.environ.get("DV_TG_ALLOWLIST", "")
    return {int(x) for x in raw.split(",") if x.strip()}


def _api_url() -> str:
    return os.environ.get("DV_FASTAPI_URL", "http://127.0.0.1:8080").rstrip("/")


def _parse_kv(tokens: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Split positional args from key=value pairs."""
    positional: List[str] = []
    kv: Dict[str, str] = {}
    for tok in tokens:
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k.strip()] = v.strip()
        else:
            positional.append(tok)
    return positional, kv


# ---------------------------------------------------------------------------
# command handlers — bound to python-telegram-bot v20 Application below
# ---------------------------------------------------------------------------


async def _is_allowed(update) -> bool:
    al = _allowlist()
    if not al:
        return True  # empty allowlist = open to all (Colab convenience)
    return update.effective_user and update.effective_user.id in al


async def cmd_start(update, ctx):
    if not await _is_allowed(update):
        return await update.message.reply_text("not authorized")
    await update.message.reply_text(
        "deepVogue inference bot.\n"
        "/models — list registered checkpoints\n"
        "/gen <model> seed=N trunc=0.7\n"
        "/walk <model> seeds=A,B steps=24 fps=24 mode=cubic\n"
        "/film <model> <walk_id>\n"
        "/factor <model> seed=N idx=K deg=0.4\n"
        "/status <model>  — last FID rows + kimg"
    )


async def cmd_models(update, ctx):
    if not await _is_allowed(update):
        return
    import httpx

    async with httpx.AsyncClient(timeout=30) as cli:
        r = await cli.get(f"{_api_url()}/models")
    if r.status_code != 200:
        return await update.message.reply_text(f"server: HTTP {r.status_code}")
    models = r.json()
    if not models:
        return await update.message.reply_text("no models registered")
    msg = "\n".join(
        f"• {m['id']}  ({m['backbone']}, {m['dataset_kind']}, ψ={m['default_trunc']})"
        for m in models
    )
    await update.message.reply_text(msg)


async def cmd_gen(update, ctx):
    if not await _is_allowed(update):
        return
    pos, kv = _parse_kv(ctx.args)
    if not pos:
        return await update.message.reply_text("usage: /gen <model> seed=N trunc=0.7")
    payload = {
        "model": pos[0],
        "seed": int(kv.get("seed", 0)),
    }
    if "trunc" in kv:
        payload["trunc"] = float(kv["trunc"])
    import httpx

    async with httpx.AsyncClient(timeout=120) as cli:
        r = await cli.post(f"{_api_url()}/generate", json=payload)
    if r.status_code != 200:
        return await update.message.reply_text(
            f"server: HTTP {r.status_code} {r.text[:200]}"
        )
    await update.message.reply_photo(photo=r.content)


async def cmd_walk(update, ctx):
    if not await _is_allowed(update):
        return
    pos, kv = _parse_kv(ctx.args)
    if not pos or "seeds" not in kv:
        return await update.message.reply_text(
            "usage: /walk <model> seeds=A,B,C steps=24 fps=24 mode=cubic"
        )
    seeds = [int(s) for s in kv["seeds"].split(",") if s]
    payload = {
        "model": pos[0],
        "seeds": seeds,
        "steps": int(kv.get("steps", 24)),
        "fps": int(kv.get("fps", 24)),
        "mode": kv.get("mode", "cubic"),
    }
    if "trunc" in kv:
        payload["trunc"] = float(kv["trunc"])
    await update.message.reply_text(
        f"rendering walk ({len(seeds)} anchors × {payload['steps']} steps)…"
    )
    import httpx

    async with httpx.AsyncClient(timeout=600) as cli:
        r = await cli.post(f"{_api_url()}/walk", json=payload)
    if r.status_code != 200:
        return await update.message.reply_text(
            f"server: HTTP {r.status_code} {r.text[:200]}"
        )
    await update.message.reply_video(video=r.content)


async def cmd_film(update, ctx):
    if not await _is_allowed(update):
        return
    pos, _ = _parse_kv(ctx.args)
    if len(pos) < 2:
        return await update.message.reply_text("usage: /film <model> <walk_id>")
    model, walk_id = pos[0], pos[1]
    import httpx

    async with httpx.AsyncClient(timeout=600) as cli:
        r = await cli.get(f"{_api_url()}/films/{model}/{walk_id}")
    if r.status_code != 200:
        return await update.message.reply_text(
            f"server: HTTP {r.status_code} {r.text[:200]}"
        )
    await update.message.reply_video(video=r.content)


async def cmd_status(update, ctx):
    if not await _is_allowed(update):
        return
    pos, kv = _parse_kv(ctx.args)
    if not pos:
        return await update.message.reply_text("usage: /status <model> [last_n=5]")
    last_n = int(kv.get("last_n", 5))
    import httpx

    async with httpx.AsyncClient(timeout=30) as cli:
        r = await cli.get(
            f"{_api_url()}/status", params={"model": pos[0], "last_n": last_n}
        )
    if r.status_code != 200:
        return await update.message.reply_text(
            f"server: HTTP {r.status_code} {r.text[:200]}"
        )
    data = r.json()
    lines = [f"model: {data['model']}"]
    if data.get("latest_snapshot"):
        lines.append(f"latest: {data['latest_snapshot'].rsplit('/', 1)[-1]}")
    if data.get("fid_rows"):
        for row in data["fid_rows"]:
            try:
                fid = row["results"]["fid50k_full"]
                snap = row.get("snapshot_pkl", "?")
                lines.append(f"  {snap}: fid={fid:.3f}")
            except (KeyError, TypeError):
                continue
    else:
        lines.append("(no FID rows yet)")
    await update.message.reply_text("\n".join(lines))


async def cmd_factor(update, ctx):
    if not await _is_allowed(update):
        return
    pos, kv = _parse_kv(ctx.args)
    if not pos or "idx" not in kv:
        return await update.message.reply_text(
            "usage: /factor <model> seed=N idx=K deg=0.4"
        )
    payload = {
        "model": pos[0],
        "seed": int(kv.get("seed", 0)),
        "factor_idx": int(kv["idx"]),
        "factor_amp": float(kv.get("deg", 0.4)),
    }
    if "trunc" in kv:
        payload["trunc"] = float(kv["trunc"])
    import httpx

    async with httpx.AsyncClient(timeout=120) as cli:
        r = await cli.post(f"{_api_url()}/generate", json=payload)
    if r.status_code != 200:
        return await update.message.reply_text(
            f"server: HTTP {r.status_code} {r.text[:200]}"
        )
    await update.message.reply_photo(photo=r.content)


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    token = os.environ.get("DV_TG_TOKEN")
    if not token:
        raise SystemExit("set DV_TG_TOKEN with your BotFather token")
    try:
        from telegram.ext import Application, CommandHandler
    except ImportError as e:
        raise SystemExit("pip install -r requirements-bot.txt") from e

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("models", cmd_models))
    app.add_handler(CommandHandler("gen", cmd_gen))
    app.add_handler(CommandHandler("walk", cmd_walk))
    app.add_handler(CommandHandler("film", cmd_film))
    app.add_handler(CommandHandler("factor", cmd_factor))
    app.add_handler(CommandHandler("status", cmd_status))
    print(f"[bot] long-polling; api={_api_url()} allowlist={_allowlist() or 'OPEN'}")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
