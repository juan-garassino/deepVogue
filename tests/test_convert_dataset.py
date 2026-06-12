"""Regression: `python -m deepVogue.dataset_tool` must stay runnable.

The convert CLI used to live in a top-level dataset_tool.py shadowed by the
dataset_tool/ package, so `-m` invocation (what prepare.py spawns) was broken.
"""

import subprocess
import sys
import zipfile

import numpy as np
import PIL.Image
from click.testing import CliRunner

from deepVogue.dataset_tool.convert import convert_dataset


def test_module_is_runnable():
    res = subprocess.run(
        [sys.executable, "-m", "deepVogue.dataset_tool", "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "--source" in res.stdout


def test_convert_emits_stylegan_zip(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(3):
        arr = (np.random.rand(80, 80, 3) * 255).astype("uint8")
        PIL.Image.fromarray(arr).save(src / f"{i}.png")
    dest = tmp_path / "dataset.zip"

    res = CliRunner().invoke(
        convert_dataset,
        [
            "--source",
            str(src),
            "--dest",
            str(dest),
            "--width",
            "64",
            "--height",
            "64",
            "--transform",
            "center-crop",
        ],
    )
    assert res.exit_code == 0, res.output

    names = zipfile.ZipFile(dest).namelist()
    assert "dataset.json" in names
    assert "00000/img00000000.png" in names
