"""Lifecycle stage 7 - Deployment: the FraudLens scoring API (FastAPI).

Endpoints
  GET  /health          liveness + model identity
  GET  /model           champion, threshold, selection reason, test metrics
  POST /score           score one transaction in real time (online card features)
  POST /score/batch     score up to 1,000 transactions in time order
  GET  /alerts          the review queue: open cases created by alerts (analyst)
  GET  /alerts/{id}     one case: score, threshold, reasons, SHAP contributions, model inputs
  POST /alerts/{id}/decision   analyst verdict (fraud / legitimate + note): closes the case
  POST /feedback        verdict by transaction id (chargebacks, other systems)
  GET  /monitoring/feedback    what the decisions say: live precision, backlog, time to decision
  GET  /metrics         Prometheus metrics (scraped by Prometheus, charted in Grafana)

Every scored transaction is stored as a case (`fraudlens.casework`): alerts open a case in the
review queue, other transactions are closed as approved. Decisions are labels: they drive the
live-precision check in monitoring and become training data for the next model.

Access control (`fraudlens.security`): with FRAUDLENS_API_KEYS set, every call except /health
and /metrics needs an `X-API-Key` header whose role allows it (scorer, analyst, admin).

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

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from . import security
from .casework import get_store
from .runtime import example_transaction, load_runtime
from .utils import PROJECT_ROOT, read_json

MODEL_DIR = Path(os.environ.get("FRAUDLENS_MODEL_DIR", PROJECT_ROOT))
FEEDBACK_FILE = Path(os.environ.get("FRAUDLENS_FEEDBACK_FILE", PROJECT_ROOT / "data" / "feedback" / "feedback.csv"))

RT = load_runtime(MODEL_DIR)
LOCK = threading.Lock()          # the card store is updated in place
CASES = get_store(os.environ.get("FRAUDLENS_CASE_DB"))
KEYS = security.parse_keys(os.environ.get("FRAUDLENS_API_KEYS"))
PRECISION_FLOOR = float(os.environ.get("FRAUDLENS_PRECISION_FLOOR", "0.40"))

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
DECISIONS = Counter("fraudlens_decisions_total", "Analyst decisions on cases", ["label"], registry=REGISTRY)
OPEN_ALERTS = Gauge("fraudlens_open_alerts", "Alerts waiting for an analyst decision", registry=REGISTRY)
LIVE_PRECISION = Gauge("fraudlens_live_precision", "Share of reviewed alerts confirmed as fraud", registry=REGISTRY)
REVIEWED = Gauge("fraudlens_reviewed_alerts", "Alerts with an analyst decision", registry=REGISTRY)
DENIED = Counter("fraudlens_access_denied_total", "Requests refused by access control", ["endpoint"],
                 registry=REGISTRY)
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


def refresh_feedback_gauges() -> dict:
    fb = CASES.feedback_summary(precision_floor=PRECISION_FLOOR)
    OPEN_ALERTS.set(fb["open_alerts"])
    REVIEWED.set(fb["reviewed_alerts"])
    LIVE_PRECISION.set(fb["live_precision"] if fb["live_precision"] is not None else float("nan"))
    return fb


refresh_monitoring_gauges()
refresh_feedback_gauges()


# ------------------------------------------------------------------ access control
def require(*roles: str):
    """Dependency: the caller's API key must carry one of `roles` (admin always may)."""
    def check(request: Request, x_api_key: str | None = Header(None, description="API key (role-based)")) -> dict:
        if not KEYS:
            return {"role": "admin", "key": None, "mode": "development (access control off)"}
        role = security.role_for_key(KEYS, x_api_key)
        if role is None:
            DENIED.labels(request.url.path).inc()
            raise HTTPException(401, "Missing or unknown API key (send it in the X-API-Key header)")
        if role != "admin" and role not in roles:
            DENIED.labels(request.url.path).inc()
            raise HTTPException(403, f"Role '{role}' may not call {request.url.path}")
        return {"role": role, "key": security.key_id(x_api_key)}
    return check


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
    clear_below_amount: float | None = Field(None, description="Alerts only: a lower amount at which the same "
                                             "transaction would have been approved (None: not amount-driven)")
    case_id: str | None = Field(None, description="Case created for this transaction (record=true)")
    routed_to: str | None = Field(None, description="'review queue' for alerts, 'approved' otherwise")


class Decision(BaseModel):
    label: Literal["fraud", "legitimate"]
    analyst: str = Field(..., min_length=1, description="Who decided (recorded in the audit log)")
    note: str | None = None


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


def _score(txn: Transaction, update_state: bool, include_features: bool, record: bool = False,
           who: dict | None = None) -> dict:
    payload = {k: v for k, v in txn.model_dump().items() if v is not None}
    with LOCK:
        known = payload["card_id"] in RT.store.cards
        out = RT.scorer.score_transaction(payload, update_state=update_state)
    out["transaction_id"] = txn.transaction_id
    if record:
        out["case_id"] = CASES.record_scored(out, payload, source="api", actor=(who or {}).get("key"),
                                             role=(who or {}).get("role"))
        out["routed_to"] = "review queue" if out["alert"] else "approved"
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
            "cards_in_store": len(RT.store.cards), "time": dt.datetime.now().isoformat(timespec="seconds"),
            "access_control": "api-keys" if KEYS else "off (development mode: set FRAUDLENS_API_KEYS)"}


@app.get("/model")
def model_info(who: dict = Depends(require("scorer", "analyst"))) -> dict:
    return {"champion": RT.model.champion, "threshold": RT.model.threshold,
            "threshold_strategy": RT.model.threshold_strategy, "view": RT.model.view, **RT.model.meta,
            "source": str(RT.source)}


@app.post("/score", response_model=ScoreResult)
def score(txn: Transaction, update_state: bool = True, include_features: bool = False, record: bool = True,
          who: dict = Depends(require("scorer"))) -> dict:
    """Score one transaction: online features from the card's history -> model -> decision.

    With update_state=true the card's history is updated afterwards. With record=true (default) the
    transaction is stored as a case: an alert opens a case in the review queue (GET /alerts)."""
    return _score(txn, update_state, include_features, record, who)


@app.post("/score/batch", response_model=list[ScoreResult])
def score_batch(txns: list[Transaction], update_state: bool = True, record: bool = True,
                who: dict = Depends(require("scorer"))) -> list[dict]:
    if len(txns) > 1000:
        raise HTTPException(413, "At most 1,000 transactions per batch")
    ordered = sorted(txns, key=lambda t: t.ts or "")      # card history must be built in time order
    return [_score(t, update_state, False, record, who) for t in ordered]


@app.get("/alerts")
def alerts(status: Literal["open", "closed", "approved"] | None = "open", limit: int = 50,
           who: dict = Depends(require("analyst"))) -> list[dict]:
    """The review queue: cases created by alerts, highest score first."""
    out = CASES.cases(status, min(max(limit, 1), 500))
    for c in out:
        c.pop("features", None)
    return out


@app.get("/alerts/{case_id}")
def alert_case(case_id: str, who: dict = Depends(require("analyst"))) -> dict:
    """One case with everything an analyst needs: score vs threshold, reasons, contributions, inputs."""
    try:
        return CASES.get(case_id)
    except KeyError:
        raise HTTPException(404, f"No case {case_id}") from None


@app.post("/alerts/{case_id}/decision")
def decide(case_id: str, d: Decision, who: dict = Depends(require("analyst"))) -> dict:
    """Record the analyst's verdict; the case closes and the label feeds monitoring and retraining."""
    try:
        case = CASES.decide(case_id, d.label, d.analyst, d.note, role=who["role"])
    except KeyError:
        raise HTTPException(404, f"No case {case_id}") from None
    DECISIONS.labels(d.label).inc()
    FEEDBACK.labels(d.label, str(case["alert"]).lower()).inc()
    refresh_feedback_gauges()
    return {"status": "recorded", "case": {k: case[k] for k in ("case_id", "status", "decision", "analyst",
                                                                 "decided_at", "score", "alert")}}


@app.get("/monitoring/feedback")
def feedback_monitoring(who: dict = Depends(require("analyst"))) -> dict:
    """How analyst decisions inform monitoring: live precision (Wilson 95% CI) vs the policy floor,
    backlog, time to decision and the number of new labelled examples for retraining."""
    return refresh_feedback_gauges()


@app.post("/feedback")
def feedback(fb: Feedback, who: dict = Depends(require("analyst"))) -> dict:
    """Record a verdict by transaction id (e.g. a chargeback). These labelled outcomes feed monitoring
    and the next retraining; a matching case is closed with the decision."""
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    new = not FEEDBACK_FILE.exists()
    row = {"received_at": dt.datetime.now().isoformat(timespec="seconds"), **fb.model_dump()}
    with LOCK, open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    FEEDBACK.labels(fb.label, str(fb.was_alert).lower()).inc()
    DECISIONS.labels(fb.label).inc()
    case = CASES.decide_transaction(fb.transaction_id, fb.label, fb.analyst or who.get("key") or "feedback-api",
                                    fb.note, extra={"score": fb.score, "source": "feedback"}, role=who["role"])
    refresh_feedback_gauges()
    return {"status": "recorded", "file": FEEDBACK_FILE.name, "case_id": case["case_id"]}


@app.get("/metrics")
def metrics() -> Response:
    refresh_monitoring_gauges()
    refresh_feedback_gauges()
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
