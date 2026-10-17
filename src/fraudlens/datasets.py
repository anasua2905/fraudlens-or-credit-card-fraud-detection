"""Loaders for the tables produced by the Step 1 pipeline."""
from __future__ import annotations

import pandas as pd

from .utils import read_table

DESCRIPTIVE_COLUMNS = ["transaction_id", "merchant", "job", "city", "state", "city_pop"]


def _ensure_datetime(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df and not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c])
    return df


def load_silver(cfg: dict) -> pd.DataFrame:
    df = read_table(cfg["paths"]["silver"] / "transactions")
    return _ensure_datetime(df, ["ts", "dob"])


def load_features(cfg: dict, split: str) -> pd.DataFrame:
    """Model-ready feature table for one split ('train', 'val' or 'test')."""
    df = read_table(cfg["paths"]["gold"] / f"features_{split}")
    return _ensure_datetime(df, ["ts"])


def load_analysis_frame(cfg: dict, split: str = "train") -> pd.DataFrame:
    """Features joined with descriptive columns (merchant, state, ...) for EDA."""
    feats = load_features(cfg, split)
    dash = read_table(cfg["paths"]["gold"] / "dashboard_transactions")
    dash = dash.loc[dash["split"].astype(str) == split, DESCRIPTIVE_COLUMNS]
    df = feats.merge(dash, on="transaction_id", how="left", validate="one_to_one")
    for c in ["category", "gender", "merchant", "job", "city", "state"]:
        df[c] = df[c].astype("category")
    return df
