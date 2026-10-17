"""Online feature store: the same features as `features.py`, one transaction at a time.

The batch pipeline computes card-history and velocity features with vectorised
group operations. A deployed scorer sees one transaction at a time, so it needs
each card's running state instead: how many earlier transactions, their sum and
sum of squares, when the last one happened, counts per category, and the
transactions inside the longest look-back window.

`CardStore.from_history` builds that state from historical transactions, and
`CardStore.features` + `CardStore.update` reproduce the batch features exactly
(see tests/test_online.py, which replays cards through both paths).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import haversine_km

EPOCH = pd.Timestamp("1970-01-01")


def _epoch(ts) -> int:
    return int((pd.Timestamp(ts) - EPOCH) // pd.Timedelta(seconds=1))


@dataclass
class CardState:
    n: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    last_t: int | None = None
    category_counts: dict = field(default_factory=dict)
    merchants: set = field(default_factory=set)
    window: deque = field(default_factory=deque)      # (epoch_seconds, amount), oldest first
    profile: dict = field(default_factory=dict)       # gender, dob, city_pop, lat, lon, state


class CardStore:
    PROFILE_COLUMNS = ["gender", "dob", "city_pop", "lat", "lon", "state"]

    def __init__(self, windows: dict[str, int], night_hours: list[int]):
        self.windows = dict(windows)
        self.max_window = max(self.windows.values())
        self.night_hours = set(night_hours)
        self.cards: dict[str, CardState] = {}

    # ------------------------------------------------------------ building
    @classmethod
    def from_history(cls, silver: pd.DataFrame, windows: dict[str, int], night_hours: list[int]) -> "CardStore":
        """State of every card after all transactions in `silver` (vectorised)."""
        store = cls(windows, night_hours)
        df = silver.sort_values(["card_id", "ts", "transaction_id"])
        t = ((df["ts"] - EPOCH) // pd.Timedelta(seconds=1)).astype("int64")
        df = df.assign(_t=t.to_numpy(), _sq=df["amount"].to_numpy() ** 2,
                       card_id=df["card_id"].astype(str), merchant=df["merchant"].astype(str),
                       category=df["category"].astype(str))
        agg = df.groupby("card_id", sort=False, observed=True).agg(
            n=("amount", "size"), total=("amount", "sum"), total_sq=("_sq", "sum"), last_t=("_t", "max"))
        last = df.groupby("card_id", sort=False, observed=True).tail(1).set_index("card_id")
        cats = df.groupby(["card_id", "category"], sort=False, observed=True).size()
        merch = df.groupby("card_id", sort=False, observed=True)["merchant"].agg(lambda s: set(map(str, s)))
        recent = df[df["_t"] >= df["card_id"].map(agg["last_t"]).astype("int64") - store.max_window]
        win = recent.groupby("card_id", sort=False, observed=True)[["_t", "amount"]].apply(
            lambda g: list(zip(g["_t"].astype(int), g["amount"].astype(float))))
        for card, row in agg.iterrows():
            card = str(card)
            st = CardState(n=int(row["n"]), total=float(row["total"]), total_sq=float(row["total_sq"]),
                           last_t=int(row["last_t"]))
            st.window = deque(win.get(card, []))
            st.merchants = set(merch.get(card, set()))
            lr = last.loc[card]
            st.profile = {c: (lr[c] if c in lr.index else None) for c in cls.PROFILE_COLUMNS}
            store.cards[card] = st
        for (card, cat), cnt in cats.items():
            store.cards[str(card)].category_counts[str(cat)] = int(cnt)
        return store

    # ------------------------------------------------------------ scoring
    def profile(self, card_id: str) -> dict:
        st = self.cards.get(str(card_id))
        return dict(st.profile) if st else {}

    def features(self, txn: dict) -> dict:
        """Features for one transaction from the card's PRIOR state (state is not changed).

        `txn` needs: card_id, ts, amount, category, merchant. Profile fields (gender, dob,
        city_pop, lat, lon) default to the card's stored profile; merch_lat/merch_lon are optional.
        """
        card = str(txn["card_id"])
        st = self.cards.get(card, CardState())
        prof = {**st.profile, **{k: v for k, v in txn.items() if k in self.PROFILE_COLUMNS and v is not None}}
        ts = pd.Timestamp(txn["ts"])
        t = _epoch(ts)
        amount = float(txn["amount"])
        cat = str(txn["category"])
        f: dict = {"amount": amount, "log_amount": math.log1p(amount)}
        f["hour"] = ts.hour
        f["day_of_week"] = ts.dayofweek
        f["is_weekend"] = int(ts.dayofweek >= 5)
        f["is_night"] = int(ts.hour in self.night_hours)
        dob = prof.get("dob")
        f["age"] = round((ts - pd.Timestamp(dob)).days / 365.25, 1) if dob is not None else np.nan
        cp = prof.get("city_pop")
        f["log_city_pop"] = math.log1p(float(cp)) if cp is not None else np.nan
        if all(prof.get(k) is not None for k in ("lat", "lon")) and txn.get("merch_lat") is not None:
            f["distance_km"] = float(haversine_km(prof["lat"], prof["lon"], txn["merch_lat"], txn["merch_lon"]))
        else:
            f["distance_km"] = np.nan
        n = st.n
        f["card_prior_txn_count"] = n
        mean = st.total / n if n > 0 else np.nan
        f["amt_to_card_mean"] = amount / mean if n > 0 and mean != 0 else np.nan
        if n >= 2:
            var = max(st.total_sq / n - mean ** 2, 0.0)
            std = math.sqrt(var)
            f["amt_zscore_card"] = (amount - mean) / std if std > 0 else np.nan
        else:
            f["amt_zscore_card"] = np.nan
        f["secs_since_last_txn"] = float(t - st.last_t) if st.last_t is not None else np.nan
        f["is_new_merchant_for_card"] = int(str(txn.get("merchant")) not in st.merchants)
        f["card_prior_category_count"] = int(st.category_counts.get(cat, 0))
        for name, secs in self.windows.items():
            inside = [a for (tt, a) in st.window if tt >= t - secs]
            f[f"txn_count_{name}"] = len(inside)
            f[f"amt_sum_{name}"] = float(sum(inside))
        f["category"] = cat
        f["gender"] = prof.get("gender")
        return f

    def update(self, txn: dict) -> None:
        """Add a transaction to its card's state (call after scoring it)."""
        card = str(txn["card_id"])
        st = self.cards.setdefault(card, CardState())
        t = _epoch(txn["ts"])
        amount = float(txn["amount"])
        st.n += 1
        st.total += amount
        st.total_sq += amount ** 2
        st.last_t = t if st.last_t is None else max(st.last_t, t)
        cat = str(txn["category"])
        st.category_counts[cat] = st.category_counts.get(cat, 0) + 1
        st.merchants.add(str(txn.get("merchant")))
        st.window.append((t, amount))
        while st.window and st.window[0][0] < t - self.max_window:
            st.window.popleft()
        for k in self.PROFILE_COLUMNS:
            if txn.get(k) is not None:
                st.profile[k] = txn[k]

    # ------------------------------------------------------------ persistence
    def to_frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cards, cats, win = [], [], []
        for card, st in self.cards.items():
            cards.append({"card_id": card, "n": st.n, "total": st.total, "total_sq": st.total_sq,
                          "last_t": st.last_t, "merchants": "|".join(sorted(st.merchants)), **st.profile})
            cats += [{"card_id": card, "category": c, "count": k} for c, k in st.category_counts.items()]
            win += [{"card_id": card, "t": t, "amount": a} for t, a in st.window]
        return pd.DataFrame(cards), pd.DataFrame(cats), pd.DataFrame(win)

    @classmethod
    def from_frames(cls, cards: pd.DataFrame, cats: pd.DataFrame, win: pd.DataFrame,
                    windows: dict[str, int], night_hours: list[int]) -> "CardStore":
        store = cls(windows, night_hours)
        for r in cards.to_dict("records"):
            st = CardState(n=int(r["n"]), total=float(r["total"]), total_sq=float(r["total_sq"]),
                           last_t=None if pd.isna(r["last_t"]) else int(r["last_t"]))
            st.merchants = set(filter(None, str(r.get("merchants") or "").split("|")))
            st.profile = {c: (None if pd.isna(r.get(c)) else r.get(c)) for c in cls.PROFILE_COLUMNS}
            store.cards[str(r["card_id"])] = st
        for r in cats.to_dict("records"):
            store.cards[str(r["card_id"])].category_counts[str(r["category"])] = int(r["count"])
        for card, g in win.sort_values(["card_id", "t"]).groupby("card_id", sort=False):
            store.cards[str(card)].window = deque(zip(g["t"].astype(int), g["amount"].astype(float)))
        return store
