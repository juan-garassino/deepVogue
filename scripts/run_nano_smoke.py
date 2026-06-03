"""Wait for the nano stack to be ready, then run the smoke test."""

import os
import sys
import time
import urllib.request


def wait(url: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return
        except Exception as e:
            last = e
            time.sleep(1)
    raise RuntimeError(f"timeout waiting for {url}: {last}")


def main():
    for url in [
        "http://localhost:5000",
        "http://localhost:4200/api/health",
        "http://localhost:8080/health",
        "http://localhost:9000/minio/health/live",
    ]:
        print(f"waiting on {url}...", flush=True)
        wait(url)
    print("all services up; running smoke", flush=True)
    os.environ["DV_NANO_SMOKE"] = "1"
    rc = os.system("pytest tests/test_nano_smoke.py -v")
    sys.exit(0 if rc == 0 else 1)


if __name__ == "__main__":
    main()
