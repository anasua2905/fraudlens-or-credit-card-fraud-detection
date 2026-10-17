"""Batch scoring (stage 7): score the validation and test periods with the champion.

Writes data/gold/scored_transactions (one row per transaction with score,
alert flag, outcome, evidence features and SHAP reasons) and
reports/dashboard_meta.json. The dashboard, the lake publication and the
monitoring stage all read these files, so they always show the model and
threshold that training selected.

    python -m fraudlens.batch_score
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .datasets import load_features
from .scoring import FraudModel
from .utils import get_logger, load_config, read_json, read_table, rel, table_exists, write_json, write_table

log = get_logger()

EVIDENCE = ["amt_to_card_mean", "amt_zscore_card", "txn_count_1h", "txn_count_24h", "amt_sum_24h",
            "secs_since_last_txn", "is_night", "category_share_card"]
DESCRIPTIVE = ["transaction_id", "ts", "card_id", "merchant", "category", "amount", "gender", "age",
               "job", "city", "state", "lat", "lon", "merch_lat", "merch_lon", "hour", "day_of_week",
               "is_fraud", "split"]
SPLITS = ["val", "test"]


def outcome(alert: np.ndarray, fraud: np.ndarray) -> np.ndarray:
    return np.select([alert & fraud, alert & ~fraud, ~alert & fraud], ["caught", "false_alert", "missed"], "cleared")


def _category_risk(cfg: dict) -> dict[str, float]:
    """Training-period fraud rate (%) per category, from the EDA summary (or the features)."""
    path = cfg["paths"]["reports"] / "eda_summary.json"
    if path.exists():
        return {k: float(v["fraud_rate_pct"]) for k, v in read_json(path).get("category", {}).items()}
    train = load_features(cfg, "train")
    return (100 * train.groupby("category", observed=True)["is_fraud"].mean()).round(3).to_dict()


def run_scoring(cfg: dict) -> dict:
    P = cfg["paths"]
    fm = FraudModel.load(P["artifacts"] / "fraudlens_model.joblib")
    dash = read_table(P["gold"] / "dashboard_transactions")
    dash["split"] = dash["split"].astype(str)
    parts = []
    for split in SPLITS:
        feats = load_features(cfg, split)
        X = fm.transform(feats)
        score = fm.predict(feats)
        ev = X.reindex(columns=[c for c in EVIDENCE if c in X.columns]).copy()
        for c in EVIDENCE:
            if c not in ev and c in feats:
                ev[c] = feats[c].to_numpy()
        ev.insert(0, "transaction_id", feats["transaction_id"].to_numpy())
        ev["score"] = score.astype("float32")
        base = dash.loc[dash["split"] == split, DESCRIPTIVE]
        parts.append(base.merge(ev, on="transaction_id", how="inner", validate="one_to_one"))
    df = pd.concat(parts, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"])
    df["alert"] = (df["score"] >= fm.threshold).astype("int8")
    df["outcome"] = outcome(df["alert"].to_numpy() == 1, df["is_fraud"].to_numpy() == 1)
    if table_exists(P["gold"] / "alert_explanations"):
        df = df.merge(read_table(P["gold"] / "alert_explanations"), on="transaction_id", how="left")
    df = df.sort_values("ts").reset_index(drop=True)
    out = write_table(df, P["gold"] / "scored_transactions", cfg["storage"]["format"])

    results = read_json(P["reports"] / "model_results.json")
    tcfg = cfg["model"]["threshold"]
    meta = {
        "champion": fm.champion,
        "selection_reason": results.get("selection", {}).get("reason"),
        "threshold_strategy": fm.threshold_strategy,
        "threshold": fm.threshold,
        "thresholds_from_validation": results["results"][fm.champion]["thresholds_from_validation"],
        "investigation_cost": tcfg["investigation_cost"],
        "recovery_rate": tcfg["recovery_rate"],
        "alerts_per_day_budget": tcfg["alerts_per_day"],
        "rows": {s: int((df["split"] == s).sum()) for s in SPLITS},
        "alerts": {s: int(df.loc[df["split"] == s, "alert"].sum()) for s in SPLITS},
        "scored_file": rel(out, cfg),
        "velocity_windows": cfg["features"]["velocity_windows"],
        "night_hours": cfg["features"]["night_hours"],
        "category_risk": _category_risk(cfg),
        "test_metrics": {k: results["results"][fm.champion]["test"][k]
                         for k in ("pr_auc", "roc_auc", "precision", "recall", "alerts_per_day", "net_benefit")},
        "base_rate_test": float(df.loc[df["split"] == "test", "is_fraud"].mean()),
    }
    write_json(meta, P["reports"] / "dashboard_meta.json")
    log.info("Scored %s transactions with %s (threshold %.4f, %s test alerts)",
             f"{len(df):,}", fm.champion, fm.threshold, f"{meta['alerts']['test']:,}")
    return meta


if __name__ == "__main__":
    run_scoring(load_config())
