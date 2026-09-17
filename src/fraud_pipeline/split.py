"""Time-based train / validation / test split.

Random splits leak future behaviour into training and overstate performance.
Here the model is always evaluated on transactions that happen AFTER the ones
it learned from, which matches how a bank deploys a fraud model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def temporal_split(df: pd.DataFrame, validation_weeks: int) -> tuple[pd.DataFrame, dict]:
    is_test = df["source_split"].astype(str).eq("test")
    train_pool = df.loc[~is_test, "ts"]
    cutoff = train_pool.max() - pd.Timedelta(weeks=validation_weeks)

    df["split"] = np.where(is_test, "test", np.where(df["ts"] > cutoff, "val", "train"))
    df["split"] = pd.Categorical(df["split"], categories=["train", "val", "test"], ordered=True)

    summary = {"validation_cutoff": cutoff, "splits": {}}
    for name, part in df.groupby("split", observed=True):
        summary["splits"][name] = {
            "rows": len(part),
            "start": part["ts"].min(),
            "end": part["ts"].max(),
            "fraud_count": int(part["is_fraud"].sum()),
            "fraud_rate_pct": round(100 * float(part["is_fraud"].mean()), 4),
            "cards": int(part["card_id"].nunique()),
        }

    s = summary["splits"]
    if s["train"]["end"] > s["val"]["start"]:
        raise AssertionError("train and validation overlap in time")
    # fraudTest.csv should start where fraudTrain.csv ends; flag (don't crash) if not.
    summary["test_starts_after_val"] = bool(s["val"]["end"] <= s["test"]["start"])
    return df, summary
