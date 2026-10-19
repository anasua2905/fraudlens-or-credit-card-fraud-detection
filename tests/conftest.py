"""One shared, quick run of the whole lifecycle on synthetic sample data.

The sample has the exact Sparkov schema but is NOT the real dataset; it exists so
every stage can be exercised in a few minutes (locally and in GitHub Actions).
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import make_sample_data  # noqa: E402
from fraudlens.run_all import run_all  # noqa: E402
from fraudlens.utils import load_config  # noqa: E402


@pytest.fixture(scope="session")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("fraudlens")
    shutil.copytree(ROOT / "config", root / "config")
    sample = make_sample_data.main(str(root / "sample"), n_cards=150)
    cfg = load_config(root=root)
    cfg["eda"].update(min_state_txns=10, sample_rows=20_000)
    cfg["model"].update(selection_bootstrap=100)
    cfg["model"]["tuning"].update(n_iter=2, cv_folds=2)
    cfg["evaluation"].update(bootstrap=50, permutation_importance_rows=2000, shap_rows=300)
    timings = run_all(cfg, stop="bundle", source="local", local_dir=str(sample), strict=False)
    return {"cfg": cfg, "root": root, "sample": Path(sample), "timings": timings}
