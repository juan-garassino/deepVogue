"""Background mirror of run_dir → Drive.

Ported from
  /Users/juan-garassino/Code/005-products/010-more-than-words/
  living_tales/trainer/tools/train_all_cases.py::_save_case_to_external

Long Colab training runs disconnect; this thread copies new files in run_dir to
drive_sync every N seconds so a kernel death never costs more than one tick.
"""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path
from typing import Optional


class DriveSync:
    def __init__(self, src: Path, dst: Optional[Path], interval: float = 60.0):
        self.src = Path(src)
        self.dst = Path(dst) if dst else None
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _mirror_once(self) -> None:
        if self.dst is None or not self.src.exists():
            return
        self.dst.mkdir(parents=True, exist_ok=True)
        for src_f in self.src.rglob("*"):
            if not src_f.is_file():
                continue
            rel = src_f.relative_to(self.src)
            dst_f = self.dst / rel
            try:
                if dst_f.exists() and dst_f.stat().st_mtime >= src_f.stat().st_mtime:
                    continue
                dst_f.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src_f), str(dst_f))
            except (OSError, shutil.Error) as e:
                print(f"[drive_sync] skip {rel}: {e}")

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._mirror_once()

    def start(self) -> "DriveSync":
        if self.dst is None:
            print("[drive_sync] DV_DRIVE_SYNC unset, sync disabled")
            return self
        self._mirror_once()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="drive-sync")
        self._thread.start()
        print(f"[drive_sync] mirroring {self.src} -> {self.dst} every {self.interval}s")
        return self

    def stop(self, *, final_flush: bool = True) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if final_flush:
            self._mirror_once()
