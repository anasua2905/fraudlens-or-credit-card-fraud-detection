"""End-to-end data pipeline: bronze -> silver -> gold.

Usage (from the project root):
    python -m fraud_pipeline.pipeline                       # download from Kaggle
    python -m fraud_pipeline.pipeline --source local --local-dir ~/Downloads/sparkov
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from . import features as F
from .clean import clean
from .ingest import ingest
from .split import temporal_split
from .utils import get_logger, load_config, write_json, write_table
from .validate import assert_valid, validate_raw

log = get_logger()

DASHBOARD_COLUMNS = [
    "transaction_id", "ts", "card_id", "merchant", "category", "amount",
    "gender", "age", "job", "city", "state", "city_pop",
    "lat", "lon", "merch_lat", "merch_lon", "distance_km",
    "hour", "day_of_week", "is_fraud", "split",
]


def run(cfg: dict, source: str = "kaggle", local_dir: str | None = None) -> dict:
    t0 = time.perf_counter()
    paths, fmt = cfg["paths"], cfg["storage"]["format"]
    run_report: dict = {}

    # 1. Bronze
    run_report["manifest"] = ingest(cfg, source=source, local_dir=local_dir)

    # 2. Validate + 3. Clean
    silver_parts, validation, cleaning = [], [], []
    for split_name, filename in cfg["dataset"]["files"].items():
        raw = pd.read_csv(paths["raw"] / filename)
        report = validate_raw(raw, filename)
        validation.append(report)
        log.info("Validation %s: %s", filename, report["status"])
        assert_valid(report)
        part, stats = clean(raw, split_name, cfg)
        silver_parts.append(part)
        cleaning.append(stats)
    write_json(validation, paths["reports"] / "validation_report.json")
    run_report["cleaning"] = cleaning

    silver = pd.concat(silver_parts, ignore_index=True)
    for c in ["category", "gender", "state", "job", "city", "merchant", "source_split"]:
        silver[c] = silver[c].astype("category")
    dup_across = int(silver["transaction_id"].duplicated().sum())
    if dup_across:
        log.warning("%s transaction ids appear in both files; keeping first", dup_across)
        silver = silver.drop_duplicates("transaction_id")
    run_report["silver_path"] = write_table(silver, paths["silver"] / "transactions", fmt)

    # 4. Features
    log.info("Engineering features on %s rows", f"{len(silver):,}")
    gold = F.build_features(silver, cfg)

    # 5. Split
    gold, split_summary = temporal_split(gold, cfg["split"]["validation_weeks"])
    run_report["split"] = split_summary

    # Write gold tables
    windows = cfg["features"]["velocity_windows"]
    model_cols = F.ID_COLUMNS + F.numeric_features(windows) + F.CATEGORICAL_FEATURES + [F.LABEL]
    outputs = {}
    for name in ["train", "val", "test"]:
        part = gold.loc[gold["split"] == name, model_cols]
        outputs[name] = write_table(part, paths["gold"] / f"features_{name}", fmt)
    outputs["dashboard"] = write_table(gold[DASHBOARD_COLUMNS], paths["gold"] / "dashboard_transactions", fmt)
    run_report["gold_outputs"] = outputs
    run_report["feature_columns"] = {
        "numeric": F.numeric_features(windows),
        "categorical": F.CATEGORICAL_FEATURES,
        "label": F.LABEL,
    }
    run_report["feature_null_rates_pct"] = {
        c: round(100 * float(v), 3)
        for c, v in gold[F.numeric_features(windows)].isna().mean().items() if v > 0
    }
    run_report["runtime_seconds"] = round(time.perf_counter() - t0, 1)
    write_json(run_report, paths["reports"] / "pipeline_run.json")
    log.info("Pipeline finished in %.1fs", run_report["runtime_seconds"])
    return run_report


def main() -> None:
    ap = argparse.ArgumentParser(description="Sparkov fraud data pipeline")
    ap.add_argument("--config", default=None, help="path to pipeline.yaml")
    ap.add_argument("--source", choices=["kaggle", "local"], default="kaggle")
    ap.add_argument("--local-dir", default=None, help="folder with fraudTrain.csv and fraudTest.csv")
    args = ap.parse_args()
    run(load_config(args.config), source=args.source, local_dir=args.local_dir)


if __name__ == "__main__":
    main()
