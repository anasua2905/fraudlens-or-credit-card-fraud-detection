"""Lifecycle stage 3 - Data preparation: validate, clean, transform, split.

bronze (raw CSVs) -> data contract -> silver (typed, de-duplicated, PII removed)
-> features (leakage-safe card history and velocity) -> chronological split -> gold.

Also builds the online feature-store snapshot used for real-time scoring.

    python -m fraudlens.prepare
"""
from __future__ import annotations

import time

import pandas as pd

from . import features as F
from .clean import clean
from .online import CardStore
from .split import temporal_split
from .utils import get_logger, load_config, read_json, rel, write_json, write_table
from .validate import assert_valid, validate_raw

log = get_logger()

DASHBOARD_COLUMNS = [
    "transaction_id", "ts", "card_id", "merchant", "category", "amount",
    "gender", "age", "job", "city", "state", "city_pop",
    "lat", "lon", "merch_lat", "merch_lon", "distance_km",
    "hour", "day_of_week", "is_fraud", "split",
]
CATEGORICAL_SILVER = ["category", "gender", "state", "job", "city", "merchant", "source_split"]


def run_prepare(cfg: dict) -> dict:
    t0 = time.perf_counter()
    paths, fmt = cfg["paths"], cfg["storage"]["format"]
    report: dict = {}
    manifest_path = paths["raw"] / "manifest.json"
    if manifest_path.exists():
        report["manifest"] = read_json(manifest_path)

    # Validate + clean each source file
    parts, validation, cleaning = [], [], []
    for split_name, filename in cfg["dataset"]["files"].items():
        raw = pd.read_csv(paths["raw"] / filename)
        rep = validate_raw(raw, filename)
        validation.append(rep)
        log.info("Validation %s: %s", filename, rep["status"])
        assert_valid(rep)
        part, stats = clean(raw, split_name, cfg)
        parts.append(part)
        cleaning.append(stats)
        del raw
    write_json(validation, paths["reports"] / "validation_report.json")
    report["cleaning"] = cleaning

    silver = pd.concat(parts, ignore_index=True)
    del parts
    for c in CATEGORICAL_SILVER:
        silver[c] = silver[c].astype("category")
    dup = int(silver["transaction_id"].duplicated().sum())
    if dup:
        log.warning("%s transaction ids appear in both files; keeping the first", dup)
        silver = silver.drop_duplicates("transaction_id")
    report["silver_path"] = rel(write_table(silver, paths["silver"] / "transactions", fmt), cfg)

    # Online feature store snapshot (state of every card after the full stream)
    store = CardStore.from_history(silver, cfg["features"]["velocity_windows"], cfg["features"]["night_hours"])
    for name, frame in zip(("cards", "categories", "window"), store.to_frames()):
        write_table(frame, paths["artifacts"] / f"card_state_{name}", fmt)
    report["card_state_cards"] = len(store.cards)

    # Features + split
    log.info("Engineering features on %s rows", f"{len(silver):,}")
    gold = F.build_features(silver, cfg)
    gold, split_summary = temporal_split(gold, cfg["split"]["validation_weeks"])
    report["split"] = split_summary

    windows = cfg["features"]["velocity_windows"]
    model_cols = F.ID_COLUMNS + F.numeric_features(windows) + F.CATEGORICAL_FEATURES + [F.LABEL]
    outputs = {}
    for name in ["train", "val", "test"]:
        outputs[name] = rel(write_table(gold.loc[gold["split"] == name, model_cols],
                                        paths["gold"] / f"features_{name}", fmt), cfg)
    outputs["dashboard"] = rel(write_table(gold[DASHBOARD_COLUMNS], paths["gold"] / "dashboard_transactions", fmt), cfg)
    report["gold_outputs"] = outputs
    report["feature_columns"] = {"numeric": F.numeric_features(windows), "categorical": F.CATEGORICAL_FEATURES,
                                 "label": F.LABEL}
    report["feature_null_rates_pct"] = {
        c: round(100 * float(v), 3) for c, v in gold[F.numeric_features(windows)].isna().mean().items() if v > 0}
    report["runtime_seconds"] = round(time.perf_counter() - t0, 1)
    write_json(report, paths["reports"] / "pipeline_run.json")
    log.info("Preparation finished in %.1fs", report["runtime_seconds"])
    return report


if __name__ == "__main__":
    run_prepare(load_config())
