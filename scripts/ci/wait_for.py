"""Wait until the named platform services are ready (used by the platform workflow).

    python scripts/ci/wait_for.py minio-init nifi-init trino clickhouse mlflow [--timeout 900]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

PLATFORM = Path(__file__).resolve().parents[2] / "platform"

HTTP = {
    "trino": ("http://localhost:8080/v1/info", lambda r: r.ok and not r.json().get("starting", True)),
    "clickhouse": ("http://localhost:8123/ping", lambda r: r.ok),
    "mlflow": ("http://localhost:5000/health", lambda r: r.ok),
    "api": ("http://localhost:8000/health", lambda r: r.ok),
    "grafana": ("http://localhost:3000/api/health", lambda r: r.ok),
    "prometheus": ("http://localhost:9090/-/ready", lambda r: r.ok),
    "superset": ("http://localhost:8088/health", lambda r: r.ok),
    "minio": ("http://localhost:9000/minio/health/ready", lambda r: r.ok),
}
JOBS = {"minio-init", "nifi-init"}            # one-shot containers: must exit 0


def job_state(service: str) -> tuple[str, int | None]:
    out = subprocess.run(["docker", "compose", "ps", "-a", "--format", "json", service], cwd=PLATFORM,
                         capture_output=True, text=True).stdout.strip()
    if not out:
        return "missing", None
    rows = [json.loads(line) for line in out.splitlines()] if not out.startswith("[") else json.loads(out)
    row = rows[0]
    return row.get("State", ""), row.get("ExitCode")


def ready(service: str) -> bool | None:
    """True = ready, False = not yet, None = failed for good."""
    if service in JOBS:
        state, code = job_state(service)
        if state == "exited":
            return True if code == 0 else None
        return False
    url, ok = HTTP[service]
    try:
        return bool(ok(requests.get(url, timeout=5)))
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("services", nargs="+")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    pending, t0 = list(a.services), time.time()
    while pending:
        for s in list(pending):
            r = ready(s)
            if r is None:
                print(f"::error title={s} failed::{s} exited with an error; see its log below")
                subprocess.run(["docker", "compose", "logs", "--no-color", "--tail", "80", s], cwd=PLATFORM)
                return 1
            if r:
                print(f"ready: {s} after {time.time() - t0:.0f}s", flush=True)
                pending.remove(s)
        if pending and time.time() - t0 > a.timeout:
            print(f"::error title=timeout::not ready after {a.timeout}s: {pending}")
            for s in pending:
                subprocess.run(["docker", "compose", "logs", "--no-color", "--tail", "60", s], cwd=PLATFORM)
            return 1
        time.sleep(5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
