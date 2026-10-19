"""Gold layer: leakage-safe features for fraud detection.

Rules that keep the features honest:
  * No feature uses the label (`is_fraud`). Label-based encodings (e.g. merchant
    fraud rate) belong in the modelling step and are fit on training data only.
  * Every per-card history feature looks strictly backwards: a transaction's
    features are computed only from that card's EARLIER transactions.
  * Features are computed on the full, time-ordered stream (train + test).
    That is safe because no labels are used, and it mirrors production,
    where a card's history carries over into the scoring period.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088

NUMERIC_FEATURES_BASE = [
    "amount", "log_amount", "hour", "day_of_week", "is_weekend", "is_night",
    "age", "log_city_pop", "distance_km",
    "card_prior_txn_count", "secs_since_last_txn", "amt_to_card_mean",
    "amt_zscore_card", "is_new_merchant_for_card", "card_prior_category_count",
]
CATEGORICAL_FEATURES = ["category", "gender"]
ID_COLUMNS = ["transaction_id", "ts", "card_id"]
LABEL = "is_fraud"


def velocity_feature_names(windows: dict[str, int]) -> list[str]:
    return [f"{kind}_{w}" for w in windows for kind in ("txn_count", "amt_sum")]


def numeric_features(windows: dict[str, int]) -> list[str]:
    return NUMERIC_FEATURES_BASE + velocity_feature_names(windows)


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _epoch_seconds(ts: pd.Series) -> np.ndarray:
    return ((ts - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy(np.int64)


def add_transaction_features(df: pd.DataFrame, night_hours: list[int]) -> pd.DataFrame:
    df["log_amount"] = np.log1p(df["amount"])
    df["hour"] = df["ts"].dt.hour.astype("int8")
    df["day_of_week"] = df["ts"].dt.dayofweek.astype("int8")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["is_night"] = df["hour"].isin(night_hours).astype("int8")
    df["age"] = ((df["ts"] - df["dob"]).dt.days / 365.25).round(1)
    df["log_city_pop"] = np.log1p(df["city_pop"])
    df["distance_km"] = haversine_km(df["lat"], df["lon"], df["merch_lat"], df["merch_lon"])
    return df


def add_card_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expects df sorted by (card_id, ts). Uses only prior transactions."""
    g = df.groupby("card_id", sort=False, observed=True)
    n_prior = g.cumcount()
    df["card_prior_txn_count"] = n_prior.astype("int32")

    prior_sum = g["amount"].cumsum() - df["amount"]
    prior_sq = (df["amount"] ** 2).groupby(df["card_id"], observed=True).cumsum() - df["amount"] ** 2
    n = n_prior.replace(0, np.nan)
    mean = prior_sum / n
    var = (prior_sq / n - mean ** 2).clip(lower=0)
    std = np.sqrt(var).where(n_prior >= 2)
    df["amt_to_card_mean"] = df["amount"] / mean
    df["amt_zscore_card"] = ((df["amount"] - mean) / std.replace(0, np.nan))

    df["secs_since_last_txn"] = g["ts"].diff().dt.total_seconds()
    df["is_new_merchant_for_card"] = (
        df.groupby(["card_id", "merchant"], sort=False, observed=True).cumcount().eq(0).astype("int8"))
    df["card_prior_category_count"] = (
        df.groupby(["card_id", "category"], sort=False, observed=True).cumcount().astype("int32"))
    return df


def add_velocity_features(df: pd.DataFrame, windows: dict[str, int]) -> pd.DataFrame:
    """Count and sum of a card's PRIOR transactions within each look-back window.

    Vectorised with a composite sort key (card code, epoch seconds) and
    binary search, so it scales to millions of rows in seconds.
    Expects df sorted by (card_id, ts).
    """
    codes = pd.factorize(df["card_id"])[0].astype(np.int64)  # contiguous blocks
    t = _epoch_seconds(df["ts"])
    offset = int(t.max() - t.min()) + max(windows.values()) + 1
    key = codes * offset + (t - t.min())
    if not np.all(np.diff(key) >= 0):
        raise ValueError("DataFrame must be sorted by card_id then ts")

    amt = df["amount"].to_numpy(np.float64)
    prefix = np.concatenate(([0.0], np.cumsum(amt)))
    idx = np.arange(len(df))
    for name, seconds in windows.items():
        left = np.searchsorted(key, key - seconds, side="left")
        df[f"txn_count_{name}"] = (idx - left).astype("int32")
        df[f"amt_sum_{name}"] = prefix[idx] - prefix[left]
    return df


def build_features(silver: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    fcfg = cfg["features"]
    df = silver.sort_values(["card_id", "ts", "transaction_id"]).reset_index(drop=True)
    df = add_transaction_features(df, fcfg["night_hours"])
    df = add_card_history_features(df)
    df = add_velocity_features(df, fcfg["velocity_windows"])
    return df.sort_values(["ts", "transaction_id"]).reset_index(drop=True)
