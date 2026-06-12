"""Smoke test: deepvogue-prepare CLI registers both subcommands and validates input."""

from click.testing import CliRunner
from deepVogue.dataset_tool.prepare import cli


def test_subcommands_registered():
    assert set(cli.commands) == {"stills", "frames"}


def test_stills_missing_source(tmp_path, monkeypatch):
    monkeypatch.setenv("DV_DATA_DIR", str(tmp_path / "nope"))
    monkeypatch.setenv("DV_DATASET_DIR", str(tmp_path / "out"))
    res = CliRunner().invoke(cli, ["stills", "--resolution", "64"])
    assert res.exit_code != 0
    assert "source not found" in res.output


def test_frames_no_videos(tmp_path, monkeypatch):
    src = tmp_path / "data"
    src.mkdir()
    monkeypatch.setenv("DV_DATA_DIR", str(src))
    monkeypatch.setenv("DV_DATASET_DIR", str(tmp_path / "out"))
    res = CliRunner().invoke(cli, ["frames", "--resolution", "64", "--fps", "1"])
    assert res.exit_code != 0
    assert "no videos under" in res.output
