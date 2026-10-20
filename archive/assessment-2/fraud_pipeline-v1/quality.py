"""Data cleaning audit on the silver layer.

Step 1 already removed structural problems (types, duplicates, invalid rows,
PII). This audit looks for subtler issues and records a decision for each.
Structural checks use all transactions; checks that need the fraud label use
the TRAINING split only, so no test labels influence cleaning decisions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .datasets import load_analysis_frame, load_silver
from .utils import get_logger, write_json

log = get_logger()

OUTLIER_COLUMNS = ["amount", "city_pop", "age", "distance_km"]

CLEANING_DECISIONS = [
    {
        "issue": "Extreme transaction amounts (IQR outliers)",
        "decision": "Keep all rows. Use log-scaling and train-fitted clipping for scale-sensitive models only.",
        "reason": "Fraud is concentrated among high amounts; removing outliers would remove the signal.",
    },
    {
        "issue": "Near-duplicate transactions (same card, merchant and amount within a short window)",
        "decision": "Keep and report. They have distinct transaction ids, so they are separate events.",
        "reason": "Rapid repeats are a known fraud pattern; the velocity features capture them.",
    },
    {
        "issue": "Missing history features on a card's first transactions",
        "decision": "Keep NaN in the tree view; median-impute with a missing-indicator in the linear view.",
        "reason": "The absence of history is informative and must stay visible to the model.",
    },
    {
        "issue": "Skewed and heavy-tailed numeric features",
        "decision": "No change to stored data; handled in preprocessing (log1p, clipping, scaling).",
        "reason": "Cleaning keeps values faithful; model-specific transforms belong in preprocessing.",
    },
    {
        "issue": "High-cardinality text columns (merchant, job, city)",
        "decision": "Excluded from the model matrices for now.",
        "reason": "They need label-based encoding, which must be fitted inside the modelling step to avoid leakage.",
    },
]


def iqr_outliers(s: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


def structural_profile(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 1),
        "columns_detail": {
            c: {"dtype": str(df[c].dtype), "missing": int(df[c].isna().sum()), "unique": int(df[c].nunique())}
            for c in df.columns
        },
    }


def near_duplicates(df: pd.DataFrame, seconds: int) -> pd.Series:
    """True for a transaction repeating the previous one on the same card+merchant+amount within `seconds`."""
    d = df.sort_values(["card_id", "merchant", "amount", "ts"])
    same = (
        d["card_id"].eq(d["card_id"].shift())
        & d["merchant"].astype(str).eq(d["merchant"].astype(str).shift())
        & d["amount"].eq(d["amount"].shift())
    )
    close = d["ts"].diff().dt.total_seconds().le(seconds)
    return (same & close).reindex(df.index)


def run_quality_audit(cfg: dict) -> dict:
    silver = load_silver(cfg)
    train = load_analysis_frame(cfg, "train")
    base_rate = float(train["is_fraud"].mean())

    audit: dict = {"structural_profile": structural_profile(silver)}

    # Consistency checks (all data, no labels)
    merch_cats = silver.groupby("merchant", observed=True)["category"].nunique()
    card_profile = silver.groupby("card_id", observed=True)[["gender", "dob", "state"]].nunique()
    age_at_txn = (silver["ts"] - silver["dob"]).dt.days / 365.25
    audit["consistency"] = {
        "merchants": int(len(merch_cats)),
        "merchants_with_multiple_categories": int((merch_cats > 1).sum()),
        "cards": int(len(card_profile)),
        "cards_with_conflicting_profile": int((card_profile > 1).any(axis=1).sum()),
        "age_min": round(float(age_at_txn.min()), 1),
        "age_max": round(float(age_at_txn.max()), 1),
        "ages_below_16": int((age_at_txn < 16).sum()),
        "ages_above_100": int((age_at_txn > 100).sum()),
    }

    # Outliers (training split, with fraud rate inside vs outside)
    outliers = {}
    for c in OUTLIER_COLUMNS:
        mask = iqr_outliers(train[c])
        outliers[c] = {
            "outlier_rows": int(mask.sum()),
            "outlier_pct": round(100 * float(mask.mean()), 2),
            "fraud_rate_in_outliers_pct": round(100 * float(train.loc[mask, "is_fraud"].mean()), 3) if mask.any() else None,
            "fraud_rate_overall_pct": round(100 * base_rate, 3),
            "share_of_all_fraud_pct": round(100 * float(train.loc[mask, "is_fraud"].sum() / train["is_fraud"].sum()), 2),
        }
    audit["outliers_train"] = outliers

    # Near-duplicates (training split)
    nd = near_duplicates(train, cfg["eda"]["near_duplicate_seconds"])
    audit["near_duplicates_train"] = {
        "window_seconds": cfg["eda"]["near_duplicate_seconds"],
        "rows": int(nd.sum()),
        "fraud_rate_pct": round(100 * float(train.loc[nd, "is_fraud"].mean()), 3) if nd.any() else None,
    }

    # Missing values in engineered features (training split)
    miss = train.isna().mean()
    audit["missing_train_pct"] = {c: round(100 * float(v), 3) for c, v in miss.items() if v > 0}

    audit["decisions"] = CLEANING_DECISIONS
    write_json(audit, cfg["paths"]["reports"] / "cleaning_audit.json")
    log.info("Cleaning audit written (%s rows audited)", f"{len(silver):,}")
    return audit


if __name__ == "__main__":
    from .utils import load_config
    run_quality_audit(load_config())
