"""Run the whole FraudLens lifecycle on synthetic sample data, end to end on the platform.

Used by the `platform` GitHub Actions workflow (and handy locally): it generates
Sparkov-shaped sample CSVs, then runs every stage from acquisition to the
ClickHouse benchmark. With the Docker platform up this exercises NiFi ingest,
MinIO, the MLflow server, the Hive Metastore crawler, Trino and ClickHouse.

    python scripts/ci/run_sample_pipeline.py --ingest nifi --cards 300

The sample is NOT the real dataset; results on it say nothing about model quality.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import ci_annotate  # noqa: E402

ci_annotate.install("sample pipeline")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import make_sample_data  # noqa: E402
from fraudlens.run_all import STAGES, run_all  # noqa: E402
from fraudlens.utils import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cards", type=int, default=300)
    ap.add_argument("--ingest", choices=["direct", "nifi"], default=None)
    ap.add_argument("--stop", choices=STAGES, default=None)
    a = ap.parse_args()

    sample = make_sample_data.main(str(ROOT / "data" / "sample"), n_cards=a.cards)
    cfg = load_config()
    if a.ingest:
        cfg["platform"]["ingest"] = a.ingest
    # small-sample settings (the same as tests/conftest.py)
    cfg["eda"].update(min_state_txns=10, sample_rows=20_000)
    cfg["model"].update(selection_bootstrap=100)
    cfg["model"]["tuning"].update(n_iter=2, cv_folds=2)
    cfg["evaluation"].update(bootstrap=50, permutation_importance_rows=2000, shap_rows=300)
    run_all(cfg, stop=a.stop, source="local", local_dir=str(sample), strict=False)


if __name__ == "__main__":
    ci_annotate.run(main, "sample pipeline")
