"""Lifecycle stage 7 - Deployment: the FraudLens scoring API (FastAPI).

Endpoints
  GET  /health          liveness + model identity
  GET  /model           champion, threshold, selection reason, test metrics
  POST /score           score one transaction in real time (online card features)
  POST /score/batch     score up to 1,000 transactions in time order
  POST /feedback        analyst verdict on an alert (fraud / legitimate) -> retraining data
  GET  /metrics         Prometheus metrics (scraped by Prometheus, charted in Grafana)

Run locally:   uvicorn fraudlens.api:app --port 8000
               (FRAUDLENS_MODEL_DIR selects the artifacts or bundle folder)
Interactive docs at http://localhost:8000/docs
"""
from __future__ import annotations

import csv
import datetime as dt
import os
import threading
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from .runtime import example_transaction, load_runtime
from .utils import PROJECT_ROOT, read_json

MODEL_DIR = Path(os.environ.get("FRAUDLENS_MODEL_DIR", PROJECT_ROOT))
FEEDBACK_FILE = Path(os.environ.get("FRAUDLENS_FEEDBACK_FILE", PROJECT_ROOT / "data" / "feedback" / "feedback.csv"))

RT = load_runtime(MODEL_DIR)
LOCK = threading.Lock()          # the card store is updated in place

# ------------------------------------------------------------------ metrics
REGISTRY = CollectorRegistry()
REQUESTS = Counter("fraudlens_requests_total", "API requests", ["endpoint", "status"], registry=REGISTRY)
LATENCY = Histogram("fraudlens_request_seconds", "Request latency", ["endpoint"], registry=REGISTRY,
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
SCORED = Counter("fraudlens_transactions_scored_total", "Transactions scored", registry=REGISTRY)
ALERTS = Counter("fraudlens_alerts_total", "Transactions flagged for review", registry=REGISTRY)
SCORES = Histogram("fraudlens_score", "Distribution of fraud scores", registry=REGISTRY,
                   buckets=(0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0))
AMOUNT = Counter("fraudlens_amount_scored_dollars_total", "Dollar value scored", registry=REGISTRY)
FEEDBACK = Counter("fraudlens_feedback_total", "Analyst feedback received", ["label", "was_alert"], registry=REGISTRY)
UNKNOWN_CARD = Counter("fraudlens_unknown_card_total", "Transactions from cards with no history", registry=REGISTRY)
THRESHOLD = Gauge("fraudlens_model_threshold", "Alert threshold in use", registry=REGISTRY)
MODEL_INFO = Gauge("fraudlens_model_info", "Model identity (value is always 1)", ["champion", "strategy"],
                   registry=REGISTRY)
MON = {name: Gauge(f"fraudlens_monitoring_{name}", doc, registry=REGISTRY) for name, doc in {
    "drift_share": "Share of monitored features drifting in the latest month (Evidently)",
    "recall": "Recall in the latest monitored month",
    "precision": "Precision in the latest monitored month",
    "alerts_per_day": "Alerts per day in the latest monitored month",
    "retrain_recommended": "1 if the monitoring policy recommends retraining",
}.items()}
THRESHOLD.set(RT.model.threshold)
MODEL_INFO.labels(RT.model.champion, RT.model.threshold_strategy).set(1)


def refresh_monitoring_gauges() -> None:
    for candidate in (MODEL_DIR / "reports" / "monitoring" / "summary.json",
                      MODEL_DIR / "reports" / "monitoring_summary.json"):
        if candidate.exists():
            s = read_json(candidate)
            latest = s.get("latest", {})
            share = latest.get("evidently_drift_share")
            MON["drift_share"].set(share if share is not None else latest.get("psi_major_share") or 0)
            MON["recall"].set(latest.get("recall") or 0)
            MON["precision"].set(latest.get("precision") or 0)
            MON["alerts_per_day"].set(latest.get("alerts_per_day") or 0)
            MON["retrain_recommended"].set(1 if s.get("retrain_recommended") else 0)
            return


refresh_monitoring_gauges()

# ------------------------------------------------------------------ schemas
_EX = example_transaction(RT.store)


class Transaction(BaseModel):
    card_id: str = Field(..., description="Pseudonymised card id (salted hash), as in the silver layer")
    amount: float = Field(..., gt=0, description="Transaction amount in dollars")
    category: str = Field(..., description="Merchant category, e.g. shopping_net, grocery_pos")
    merchant: str | None = Field(None, description="Merchant name (used for card history)")
    ts: str | None = Field(None, description="ISO timestamp; defaults to now")
    transaction_id: str | None = None
    gender: str | None = None
    dob: str | None = None
    city_pop: int | None = None
    lat: float | None = None
    lon: float | None = None
    merch_lat: float | None = None
    merch_lon: float | None = None

    model_config = {"json_schema_extra": {"examples": [_EX]}}


class ScoreResult(BaseModel):
    transaction_id: str | None
    card_id: str
    ts: str
    amount: float
    category: str
    score: float
    alert: bool
    threshold: float
    model: str
    known_card: bool
    reasons: list[str]
    contributions: list[dict] | None = None
    features: dict | None = None


class Feedback(BaseModel):
    transaction_id: str
    label: Literal["fraud", "legitimate"]
    score: float | None = None
    was_alert: bool | None = None
    analyst: str | None = None
    note: str | None = None


# ------------------------------------------------------------------ app
app = FastAPI(title="FraudLens scoring API", version=RT.model.meta.get("version", "2.0.0"),
              description="Real-time credit card fraud scoring with the FraudLens champion model. "
                          "Features are computed from each card's history in the online feature store.")


@app.middleware("http")
async def _metrics(request: Request, call_next):
    t0 = time.perf_counter()
    endpoint = request.url.path
    try:
        response = await call_next(request)
    except Exception:
        REQUESTS.labels(endpoint, "500").inc()
        raise
    if endpoint != "/metrics":
        REQUESTS.labels(endpoint, str(response.status_code)).inc()
        LATENCY.labels(endpoint).observe(time.perf_counter() - t0)
    return response


def _score(txn: Transaction, update_state: bool, include_features: bool) -> dict:
    payload = {k: v for k, v in txn.model_dump().items() if v is not None}
    with LOCK:
        known = payload["card_id"] in RT.store.cards
        out = RT.scorer.score_transaction(payload, update_state=update_state)
    out["transaction_id"] = txn.transaction_id
    if not include_features:
        out.pop("features", None)
    SCORED.inc()
    AMOUNT.inc(txn.amount)
    SCORES.observe(out["score"])
    if out["alert"]:
        ALERTS.inc()
    if not known:
        UNKNOWN_CARD.inc()
    return out


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": RT.model.champion, "threshold": RT.model.threshold,
            "cards_in_store": len(RT.store.cards), "time": dt.datetime.now().isoformat(timespec="seconds")}


@app.get("/model")
def model_info() -> dict:
    return {"champion": RT.model.champion, "threshold": RT.model.threshold,
            "threshold_strategy": RT.model.threshold_strategy, "view": RT.model.view, **RT.model.meta,
            "source": str(RT.source)}


@app.post("/score", response_model=ScoreResult)
def score(txn: Transaction, update_state: bool = True, include_features: bool = False) -> dict:
    """Score one transaction. With update_state=true the card's history is updated afterwards."""
    return _score(txn, update_state, include_features)


@app.post("/score/batch", response_model=list[ScoreResult])
def score_batch(txns: list[Transaction], update_state: bool = True) -> list[dict]:
    if len(txns) > 1000:
        raise HTTPException(413, "At most 1,000 transactions per batch")
    ordered = sorted(txns, key=lambda t: t.ts or "")      # card history must be built in time order
    return [_score(t, update_state, False) for t in ordered]


@app.post("/feedback")
def feedback(fb: Feedback) -> dict:
    """Record an analyst's verdict. These labelled outcomes feed monitoring and the next retraining."""
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    new = not FEEDBACK_FILE.exists()
    row = {"received_at": dt.datetime.now().isoformat(timespec="seconds"), **fb.model_dump()}
    with LOCK, open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    FEEDBACK.labels(fb.label, str(fb.was_alert).lower()).inc()
    return {"status": "recorded", "file": FEEDBACK_FILE.name}


@app.get("/metrics")
def metrics() -> Response:
    refresh_monitoring_gauges()
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
