"""Score the monitoring periods with the champion model for the dashboard.

Writes
  data/gold/scored_transactions  — validation + test transactions with the
                                   champion's score, alert flag and the evidence
                                   an analyst needs to judge each alert
  reports/dashboard_meta.json    — champion, thresholds and cost assumptions

The dashboard reads only these files, so it always shows the model and
threshold that the training run selected.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .datasets import load_features
from .train import score_split
from .utils import get_logger, load_config, read_table, write_json, write_table

log = get_logger()

EVIDENCE = ["amt_to_card_mean", "amt_zscore_card", "txn_count_24h", "amt_sum_24h",
            "secs_since_last_txn", "is_night"]
DESCRIPTIVE = ["transaction_id", "ts", "card_id", "merchant", "category", "amount", "gender", "age",
               "job", "city", "state", "lat", "lon", "merch_lat", "merch_lon", "hour", "day_of_week",
               "is_fraud", "split"]
SPLITS = ["val", "test"]


def run_scoring(cfg: dict) -> dict:
    results = json.loads((cfg["paths"]["reports"] / "model_results.json").read_text())
    champion = results["champion"]
    strategy = results["threshold_strategy"]
    thresholds = results["results"][champion]["thresholds_from_validation"]
    threshold = thresholds[strategy]

    dash = read_table(cfg["paths"]["gold"] / "dashboard_transactions")
    parts = []
    for split in SPLITS:
        scored = score_split(cfg, champion, split)
        evidence = load_features(cfg, split)[["transaction_id", *EVIDENCE]]
        base = dash.loc[dash["split"].astype(str) == split, DESCRIPTIVE]
        part = (base.merge(scored, on="transaction_id", how="inner", validate="one_to_one")
                    .merge(evidence, on="transaction_id", how="left", validate="one_to_one"))
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"])
    df["alert"] = (df["score"] >= threshold).astype("int8")
    df["split"] = df["split"].astype(str)
    df = df.sort_values("ts").reset_index(drop=True)

    out = write_table(df, cfg["paths"]["gold"] / "scored_transactions", cfg["storage"]["format"])
    tcfg = cfg["model"]["threshold"]
    meta = {
        "champion": champion,
        "selection_reason": results.get("selection", {}).get("reason"),
        "threshold_strategy": strategy,
        "threshold": threshold,
        "thresholds_from_validation": thresholds,
        "investigation_cost": tcfg["investigation_cost"],
        "recovery_rate": tcfg["recovery_rate"],
        "alerts_per_day_budget": tcfg["alerts_per_day"],
        "rows": {s: int((df["split"] == s).sum()) for s in SPLITS},
        "alerts": {s: int(df.loc[df["split"] == s, "alert"].sum()) for s in SPLITS},
        "scored_file": out,
    }
    write_json(meta, cfg["paths"]["reports"] / "dashboard_meta.json")
    log.info("Scored %s transactions with %s (threshold %.4f, %s alerts on test)",
             f"{len(df):,}", champion, threshold, f"{meta['alerts']['test']:,}")
    return meta


if __name__ == "__main__":
    run_scoring(load_config())
