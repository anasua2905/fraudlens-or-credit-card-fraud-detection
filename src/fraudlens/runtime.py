"""Load everything needed to score in real time, from the artifacts folder or a deploy bundle.

Used by the FastAPI service and the Streamlit dashboard so both serve exactly
the same model, threshold and card-state snapshot.

A *source* directory contains either the pipeline layout
(artifacts/fraudlens_model.joblib, artifacts/card_state_*.parquet, reports/...)
or the compact bundle layout written by `fraudlens.bundle`
(model.joblib, card_state_*.parquet, reports/...).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .online import CardStore
from .scoring import FraudModel, FraudScorer
from .utils import get_logger, read_json, read_table

log = get_logger()

DEFAULT_WINDOWS = {"1h": 3600, "24h": 86400, "7d": 604800}
DEFAULT_NIGHT = [22, 23, 0, 1, 2, 3]


@dataclass
class Runtime:
    model: FraudModel
    store: CardStore
    scorer: FraudScorer
    meta: dict
    source: Path


def _find(base: Path, *candidates: str) -> Path | None:
    for c in candidates:
        p = base / c
        if p.exists() or p.with_suffix(".parquet").exists() or p.with_suffix(".csv").exists():
            return p
    return None


def category_risk_from(base: Path) -> dict[str, float]:
    meta = _find(base, "reports/dashboard_meta.json")
    if meta is not None:
        m = read_json(meta)
        if m.get("category_risk"):
            return m["category_risk"]
    eda = _find(base, "reports/eda_summary.json")
    if eda is None:
        return {}
    return {k: float(v["fraud_rate_pct"]) for k, v in read_json(eda).get("category", {}).items()}


def load_runtime(base: Path, explain: bool = True) -> Runtime:
    base = Path(base)
    model_path = _find(base, "model.joblib", "artifacts/fraudlens_model.joblib")
    if model_path is None:
        raise FileNotFoundError(f"No model found under {base} (run the pipeline or build the bundle)")
    fm = FraudModel.load(model_path)
    state_dir = base if _find(base, "card_state_cards") else base / "artifacts"
    meta_path = _find(base, "reports/dashboard_meta.json")
    meta = read_json(meta_path) if meta_path else {}
    windows = meta.get("velocity_windows", DEFAULT_WINDOWS)
    night = meta.get("night_hours", DEFAULT_NIGHT)
    store = CardStore.from_frames(*(read_table(state_dir / f"card_state_{n}") for n in ("cards", "categories", "window")),
                                  windows=windows, night_hours=night)
    explainer = None
    if explain:
        try:
            from .explain import Explainer
            bg = _find(base, "shap_background", "artifacts/shap_background")
            explainer = Explainer(fm, read_table(bg) if bg is not None else None, permutations=1)
        except Exception as exc:  # shap missing, or a non-tree champion without background data
            log.info("Live SHAP explanations disabled (%s)", type(exc).__name__)
    scorer = FraudScorer(fm, store, category_risk_from(base), explainer)
    log.info("Runtime loaded from %s: %s, threshold %.4f, %s cards", base, fm.champion, fm.threshold,
             f"{len(store.cards):,}")
    return Runtime(fm, store, scorer, meta, base)


def example_transaction(store: CardStore, card_id: str | None = None, minutes_after_last: int = 30) -> dict:
    """A plausible next transaction for a known card (for demos and API docs)."""
    card_id = card_id or next(iter(store.cards))
    st = store.cards[card_id]
    ts = pd.Timestamp(st.last_t + minutes_after_last * 60, unit="s") if st.last_t else pd.Timestamp.now()
    return {"card_id": card_id, "ts": ts.isoformat(), "amount": 875.0, "category": "shopping_net",
            "merchant": "Example Store"}
