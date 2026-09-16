"""Generate small CSVs with the exact Sparkov schema, for tests and smoke runs.

This is NOT the real dataset. It only mimics the column layout and a few
quirks (leftover index column, "fraud_" merchant prefix, offset unix_time)
so the pipeline can be exercised without downloading ~500 MB.

    python scripts/make_sample_data.py --out data/sample --cards 60
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = ["entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
              "health_fitness", "home", "kids_pets", "misc_net", "misc_pos",
              "personal_care", "shopping_net", "shopping_pos", "travel"]
STATES = ["NY", "CA", "TX", "PA", "OH", "MI", "WA"]
UNIX_OFFSET = 7 * 365 * 86400  # mimic the offset seen in public copies


def _cards(rng, n):
    return pd.DataFrame({
        "cc_num": rng.integers(10**15, 10**16, n),
        "first": [f"First{i}" for i in range(n)],
        "last": [f"Last{i}" for i in range(n)],
        "gender": rng.choice(["M", "F"], n),
        "street": [f"{i} Main St" for i in range(n)],
        "city": [f"City{i % 20}" for i in range(n)],
        "state": rng.choice(STATES, n),
        "zip": rng.integers(10000, 99999, n),
        "lat": rng.uniform(30, 45, n),
        "long": rng.uniform(-120, -75, n),
        "city_pop": rng.integers(200, 2_000_000, n),
        "job": rng.choice(["Engineer", "Teacher", "Nurse", "Designer"], n),
        "dob": pd.to_datetime(rng.integers(pd.Timestamp("1950-01-01").value // 10**9,
                                           pd.Timestamp("2000-01-01").value // 10**9, n), unit="s").strftime("%Y-%m-%d"),
    })


def generate(start: str, end: str, cards: pd.DataFrame, rng, fraud_prob=0.01) -> pd.DataFrame:
    t0, t1 = pd.Timestamp(start).value // 10**9, pd.Timestamp(end).value // 10**9
    rows = []
    for _, c in cards.iterrows():
        n = rng.integers(40, 120)
        ts = np.sort(rng.integers(t0, t1, n))
        amt = np.round(rng.lognormal(3.5, 1.0, n), 2)
        fraud = rng.random(n) < fraud_prob
        # fraud bursts: a few large, late-night, nearby-in-time transactions
        if fraud.any():
            base = ts[fraud][0]
            k = int(fraud.sum())
            ts[fraud] = base + np.sort(rng.integers(0, 7200, k))
            amt[fraud] = np.round(rng.uniform(300, 1200, k), 2)
        dist = np.where(fraud, rng.uniform(1.0, 3.0, n), rng.uniform(0.0, 1.0, n))
        df = pd.DataFrame({
            "ts": ts, "amt": amt, "is_fraud": fraud.astype(int),
            "merchant": [f"fraud_Merchant {m}" for m in rng.integers(0, 80, n)],
            "category": rng.choice(CATEGORIES, n),
            "merch_lat": c["lat"] + dist * rng.choice([-1, 1], n),
            "merch_long": c["long"] + dist * rng.choice([-1, 1], n),
        })
        for col in cards.columns:
            df[col] = c[col]
        rows.append(df)
    out = pd.concat(rows).sort_values("ts").reset_index(drop=True)
    out["trans_date_trans_time"] = pd.to_datetime(out["ts"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
    out["unix_time"] = out["ts"] - UNIX_OFFSET
    out["trans_num"] = [f"{rng.integers(0, 2**63):032x}" for _ in range(len(out))]
    cols = ["trans_date_trans_time", "cc_num", "merchant", "category", "amt", "first", "last",
            "gender", "street", "city", "state", "zip", "lat", "long", "city_pop", "job", "dob",
            "trans_num", "unix_time", "merch_lat", "merch_long", "is_fraud"]
    out = out[cols]
    out.insert(0, "Unnamed: 0", range(len(out)))
    return out


def main(out_dir: str = "data/sample", n_cards: int = 60, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cards = _cards(rng, n_cards)
    generate("2019-01-01", "2019-06-01", cards, rng).to_csv(out / "fraudTrain.csv", index=False)
    generate("2019-06-01", "2019-08-01", cards, rng).to_csv(out / "fraudTest.csv", index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sample")
    ap.add_argument("--cards", type=int, default=60)
    a = ap.parse_args()
    print("Wrote sample to", main(a.out, a.cards))
