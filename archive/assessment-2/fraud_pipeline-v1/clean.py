"""Silver layer: typed, de-duplicated, PII-reduced transactions."""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

from .utils import get_logger

log = get_logger()

RENAME = {
    "trans_date_trans_time": "ts",
    "trans_num": "transaction_id",
    "amt": "amount",
    "long": "lon",
    "merch_long": "merch_lon",
}

SILVER_COLUMNS = [
    "transaction_id", "ts", "card_id", "merchant", "category", "amount",
    "gender", "dob", "job", "city", "state", "zip", "city_pop",
    "lat", "lon", "merch_lat", "merch_lon", "is_fraud", "source_split",
]


def hash_card_numbers(cc_num: pd.Series, salt: str) -> pd.Series:
    """Salted SHA-256, truncated to 16 hex chars; hashed once per unique card."""
    uniques = cc_num.astype("int64").unique()
    mapping = {u: hashlib.sha256(f"{salt}{u}".encode()).hexdigest()[:16] for u in uniques}
    return cc_num.astype("int64").map(mapping)


def clean(df: pd.DataFrame, source_split: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    stats = {"source_split": source_split, "rows_in": len(df)}
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

    # Types
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])
    df["is_fraud"] = df["is_fraud"].astype("int8")

    # Exact duplicates on the transaction id
    before = len(df)
    df = df.drop_duplicates(subset="trans_num", keep="first")
    stats["duplicates_removed"] = before - len(df)

    # Invalid rows (validation already reports these; cleaning removes them)
    invalid = (
        (df["amt"] <= 0)
        | ~df["lat"].between(-90, 90) | ~df["long"].between(-180, 180)
        | ~df["merch_lat"].between(-90, 90) | ~df["merch_long"].between(-180, 180)
    )
    stats["invalid_rows_removed"] = int(invalid.sum())
    df = df.loc[~invalid]

    # Sparkov prefixes EVERY merchant name with "fraud_" (a generator artifact,
    # not a label). Strip it so nobody mistakes it for a leaking signal.
    df["merchant"] = df["merchant"].str.removeprefix("fraud_")

    # Privacy: pseudonymise the card number, drop direct identifiers
    salt = os.environ.get(cfg["privacy"]["hash_salt_env"], "sparkov-default-salt")
    df["card_id"] = hash_card_numbers(df["cc_num"], salt)
    df = df.drop(columns=["cc_num", "unix_time", *cfg["privacy"]["drop_columns"]])

    df = df.rename(columns=RENAME)
    df["source_split"] = source_split

    # Compact dtypes
    for c in ["category", "gender", "state", "job", "city", "merchant", "source_split"]:
        df[c] = df[c].astype("category")
    for c in ["lat", "lon", "merch_lat", "merch_lon"]:
        df[c] = df[c].astype(np.float64)
    df["amount"] = df["amount"].astype(np.float64)

    df = df[SILVER_COLUMNS].sort_values(["ts", "transaction_id"]).reset_index(drop=True)
    stats["rows_out"] = len(df)
    log.info("Cleaned %s: %s -> %s rows", source_split, f"{stats['rows_in']:,}", f"{stats['rows_out']:,}")
    return df, stats
