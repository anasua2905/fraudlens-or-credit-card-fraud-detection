"""Collect the results of a `platform` workflow run into reports/platform_ci.json.

The run publishes its smoke-test, OLAP-benchmark and governance results as notice
annotations; this reads them back through the GitHub checks API (needs `gh`).

    python scripts/ci/collect_platform_results.py [RUN_ID]      # default: latest run on final-a4
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = "anasua2905/fraudlens-or-credit-card-fraud-detection"
OUT = Path(__file__).resolve().parents[2] / "reports" / "platform_ci.json"


def gh(path: str):
    return json.loads(subprocess.run(["gh", "api", path], capture_output=True, text=True, check=True).stdout)


def main() -> int:
    if len(sys.argv) > 1:
        run = gh(f"repos/{REPO}/actions/runs/{sys.argv[1]}")
    else:
        runs = gh(f"repos/{REPO}/actions/workflows/platform.yml/runs?branch=final-a4&per_page=1")["workflow_runs"]
        run = runs[0]
    out = {"run_id": run["id"], "run_url": run["html_url"], "commit": run["head_sha"][:7],
           "date": run["updated_at"], "conclusion": run["conclusion"], "jobs": {}}
    for job in gh(f"repos/{REPO}/actions/runs/{run['id']}/jobs")["jobs"]:
        out["jobs"][job["name"]] = job["conclusion"]
        for a in gh(f"repos/{REPO}/check-runs/{job['id']}/annotations"):
            if a["annotation_level"] != "notice":
                continue
            try:
                payload = json.loads(a["message"])
            except ValueError:
                continue
            key = {"platform smoke result": "smoke", "olap benchmark": "olap",
                   "governance result": "governance"}.get(a["title"])
            if key:
                out[key] = payload
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k in ("run_url", "conclusion", "jobs")}, indent=1))
    return 0 if out["conclusion"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
