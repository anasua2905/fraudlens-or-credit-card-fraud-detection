"""Build (and optionally execute) the eight lifecycle notebooks in notebooks/.

One notebook per stage of the data science lifecycle. They read the outputs of
`python -m fraudlens.run_all` and demonstrate each stage with live code, so run
the pipeline first. The source of the notebooks is the specs in
scripts/notebook_specs/; this script turns them into .ipynb files.

    python scripts/build_notebooks.py                  # write notebooks/*.ipynb
    python scripts/build_notebooks.py --execute        # ... and run them (outputs embedded)
    python scripts/build_notebooks.py --execute --only 04 06

FRAUDLENS_ROOT points the notebooks at another project root (used by CI on sample data).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "scripts" / "notebook_specs"
OUT = ROOT / "notebooks"

SETUP = '''import os, sys, json, warnings
from pathlib import Path
ROOT = Path(os.environ.get("FRAUDLENS_ROOT") or (Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd())).resolve()
CODE = Path(os.environ.get("FRAUDLENS_CODE", ROOT)).resolve()
sys.path.insert(0, str(CODE / "src"))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display

from fraudlens import viz
from fraudlens.utils import load_config, read_json, read_table

viz.setup()
import plotly.io as pio
pio.renderers.default = "plotly_mimetype+notebook_connected"   # small notebooks: plotly.js from the CDN
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 170)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
cfg = load_config(root=ROOT)
P = cfg["paths"]
print(f"Project root: {ROOT.name} | data present: {(P['silver'] / 'transactions.parquet').exists()}")'''


def load_spec(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(spec_mod) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    cells = []
    for kind, text in spec_mod.CELLS:
        text = text.strip("\n")
        if kind == "md":
            cells.append(nbf.v4.new_markdown_cell(text))
        elif kind == "setup":
            cells.append(nbf.v4.new_code_cell(SETUP + ("\n" + text if text else "")))
        else:
            cells.append(nbf.v4.new_code_cell(text))
    nb.cells = cells
    return nb


def execute(path: Path, timeout: int, kernel: str = "python3") -> float:
    from nbclient import NotebookClient
    nb = nbf.read(path, as_version=4)
    t0 = time.time()
    try:
        NotebookClient(nb, timeout=timeout, kernel_name=kernel,
                       resources={"metadata": {"path": str(path.parent)}}).execute()
    finally:                      # keep partial outputs too: they show where a failure happened
        nbf.write(nb, path)
    return time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--only", nargs="*", help="notebook number prefixes, e.g. 04 06")
    ap.add_argument("--timeout", type=int, default=3600, help="per-cell timeout (s)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--kernel", default="python3", help="Jupyter kernel to execute with (e.g. fraudlens)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    failed = []
    for spec_path in sorted(SPECS.glob("nb*.py")):
        number = spec_path.stem[2:4]
        if a.only and number not in a.only:
            continue
        mod = load_spec(spec_path)
        path = out / mod.FILENAME
        nbf.write(build(mod), path)
        print(f"wrote {path.relative_to(out.parent) if out.parent in path.parents else path}", flush=True)
        if a.execute:
            try:
                secs = execute(path, a.timeout, a.kernel)
                print(f"  executed in {secs:.0f}s", flush=True)
            except Exception as exc:
                failed.append(mod.FILENAME)
                print(f"  FAILED: {type(exc).__name__}: {str(exc)[-3000:]}", flush=True)
    if failed:
        print("failed notebooks:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
