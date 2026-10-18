"""Scoring (lifecycle stage 7): one object that turns features into fraud scores.

`FraudModel` packages everything needed to score: the fitted preprocessor(s),
the champion model (plus the Isolation Forest if the champion is the hybrid),
the alert threshold chosen on validation, and metadata. The batch pipeline,
the FastAPI service and the Streamlit dashboard all score through it, so they
cannot drift apart.

`FraudScorer` adds the online feature store (`online.CardStore`) so a single
raw transaction can be scored in real time, with plain-language reasons.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import features as F
from . import models as M
from .online import CardStore
from .preprocess import MATRIX_DTYPE, add_stationary_features


@dataclass
class FraudModel:
    champion: str
    model: object
    preprocessor: object                 # fitted ColumnTransformer for the champion's view
    threshold: float
    threshold_strategy: str
    view: str
    iso_model: object | None = None      # hybrid only
    linear_preprocessor: object | None = None  # hybrid only (Isolation Forest input)
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- core
    def transform(self, feats: pd.DataFrame) -> pd.DataFrame:
        df = feats.copy()
        if "category_share_card" not in df:
            df = add_stationary_features(df)
        for c in F.CATEGORICAL_FEATURES:
            df[c] = df[c].astype(str)
        X = self.preprocessor.transform(df)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.preprocessor.get_feature_names_out(), index=df.index)
        # Same representation as the stored training matrices (see preprocess.MATRIX_DTYPE).
        X = X.astype(MATRIX_DTYPE)
        if self.view == "linear":
            # Training matrices of the linear view never contain NaN; a brand-new card can lack
            # customer fields (age, city size), so fall back to the standardised mean (0).
            X = X.fillna(0.0)
        for c in M.CATEGORICAL_IN_TREE_VIEW:
            if self.view == "tree" and c in X.columns:
                X[c] = X[c].fillna(-1).astype("int32")
        if self.champion == "hybrid_gb":
            Xl = self.linear_preprocessor.transform(df)
            if not isinstance(Xl, pd.DataFrame):
                Xl = pd.DataFrame(Xl, columns=self.linear_preprocessor.get_feature_names_out(), index=df.index)
            X = X.assign(anomaly_score=M.anomaly_score(self.iso_model, Xl.astype(MATRIX_DTYPE)))
        return X

    def predict(self, feats: pd.DataFrame) -> np.ndarray:
        return M.positive_scores(self.model, self.transform(feats))

    def decide(self, scores: np.ndarray) -> np.ndarray:
        return (np.asarray(scores) >= self.threshold).astype(np.int8)

    # ---------------------------------------------------------------- io
    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        return path

    @staticmethod
    def load(path: Path) -> "FraudModel":
        return joblib.load(path)


REASON_RULES = (
    ("amt_to_card_mean", lambda v: v >= 2, lambda v, r: f"Amount is {v:.1f}x this card's usual spend"),
    ("is_night", lambda v: v == 1, lambda v, r: f"Night-time transaction ({int(r['hour']):02d}:00)"),
    ("txn_count_24h", lambda v: v >= 3, lambda v, r: f"{int(v)} earlier transactions in 24h "
                                                     f"totalling ${r.get('amt_sum_24h', 0):,.0f}"),
    ("secs_since_last_txn", lambda v: v < 3600, lambda v, r: f"Only {v / 60:.0f} min since the card's previous transaction"),
    ("amt_zscore_card", lambda v: v >= 3, lambda v, r: f"Amount is {v:.1f} standard deviations above the card's mean"),
)


def rule_reasons(row: dict, category_risk: dict | None = None) -> list[str]:
    """Plain-language evidence from the engineered features (no model internals)."""
    out = []
    for col, test, text in REASON_RULES:
        v = row.get(col)
        if v is not None and not (isinstance(v, float) and np.isnan(v)) and test(v):
            out.append(text(v, row))
    if category_risk:
        top = sorted(category_risk, key=category_risk.get, reverse=True)[:3]
        if row.get("category") in top:
            out.append(f"High-risk category ({row['category']}: {category_risk[row['category']]:.2f}% "
                       "fraud rate in training)")
    return out or ["No single strong indicator; flagged by the combined pattern"]


class FraudScorer:
    """Real-time scoring of raw transactions: online features -> model -> decision + reasons."""

    def __init__(self, model: FraudModel, store: CardStore, category_risk: dict | None = None,
                 explainer=None):
        self.model = model
        self.store = store
        self.category_risk = category_risk or {}
        self.explainer = explainer

    def score_transaction(self, txn: dict, update_state: bool = True) -> dict:
        txn = dict(txn)
        txn.setdefault("ts", dt.datetime.now().replace(microsecond=0).isoformat())
        feats = self.store.features(txn)
        frame = pd.DataFrame([feats])
        score = float(self.model.predict(frame)[0])
        alert = bool(score >= self.model.threshold)
        out = {
            "card_id": str(txn["card_id"]),
            "ts": str(pd.Timestamp(txn["ts"])),
            "amount": float(txn["amount"]),
            "category": str(txn["category"]),
            "score": round(score, 6),
            "alert": alert,
            "threshold": round(self.model.threshold, 6),
            "model": self.model.champion,
            "known_card": str(txn["card_id"]) in self.store.cards,
            "reasons": rule_reasons(feats, self.category_risk) if alert else [],
            "features": {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in feats.items()},
        }
        if self.explainer is not None:
            try:
                contrib = self.explainer.top_contributions(self.model.transform(frame), k=5)[0]
                for item in contrib:          # show category names, not ordinal codes
                    if item["feature"] in F.CATEGORICAL_FEATURES:
                        item["value"] = feats.get(item["feature"])
                out["contributions"] = contrib
            except Exception:  # explanation is a convenience, never a reason to fail scoring
                out["contributions"] = None
        if update_state:
            self.store.update(txn)
        return out
