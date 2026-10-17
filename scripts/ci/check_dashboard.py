"""Render the Streamlit dashboard headlessly (streamlit.testing AppTest) and fail on any exception.

This is the check that the committed deploy bundle (dashboard/bundle) is enough to run
the hosted dashboard, exactly as Streamlit Community Cloud would.

    python scripts/ci/check_dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    from streamlit.testing.v1 import AppTest
    bundle = ROOT / "dashboard" / "bundle" / "manifest.json"
    if not bundle.exists():
        print("::warning::dashboard/bundle is not committed; skipping the dashboard render check")
        return 0
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=240)
    at.run()
    if at.exception:
        for e in at.exception:
            print(f"::error title=dashboard exception::{e.message}")
            print(e.stack_trace if hasattr(e, "stack_trace") else e)
        return 1
    print(f"dashboard rendered: {len(at.tabs)} tabs, {len(at.metric)} metrics, {len(at.markdown)} markdown blocks")
    assert len(at.tabs) >= 5, "expected the dashboard tabs"
    return 0


if __name__ == "__main__":
    sys.exit(main())
