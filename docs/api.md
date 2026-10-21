# Scoring API reference

FastAPI service (`src/fraudlens/api.py`). Interactive docs at `http://<host>:8000/docs` (Swagger) and `/redoc`.

Start it:

```bash
uvicorn fraudlens.api:app --host 0.0.0.0 --port 8000          # from the repository (uses the pipeline output or the bundle)
docker run -p 8000:8000 ghcr.io/anasua2905/fraudlens-app api   # the published image
```

Environment:

| Variable | Meaning | Default |
|---|---|---|
| `FRAUDLENS_MODEL_DIR` | bundle or project folder | the repository |
| `FRAUDLENS_API_KEYS` | role-based API keys, `scorer:<key>,analyst:<key>,admin:<key>` | not set: access control **off** (development) |
| `FRAUDLENS_CASE_DB` | the case store (SQLite): cases, decisions, audit log | `data/cases/cases.db` |
| `FRAUDLENS_FEEDBACK_FILE` | CSV copy of verdicts sent to `/feedback` | `data/feedback/feedback.csv` |
| `FRAUDLENS_PRECISION_FLOOR` | live-precision floor for the feedback check | `0.40` |

## The path of a new transaction

```
POST /score ──► online feature store (card history) ──► champion model ──► score vs threshold
                                                                            │
                         ┌──────────────────────────────────────────────────┴───────────────┐
                         ▼ alert                                                            ▼ no alert
              case "open" in the review queue                                  case "approved" (logged)
                         │ GET /alerts, GET /alerts/{id}
                         ▼
              POST /alerts/{id}/decision  {label, analyst, note}  ──► case "closed" + audit entry
                         │
                         ▼
              GET /monitoring/feedback: live precision, backlog, time to decision, labelled examples
              /metrics: fraudlens_live_precision, fraudlens_open_alerts → Prometheus alert rules, Grafana
              python -m fraudlens.monitor: adds the block to the monitoring summary, exports labelled examples
```

## Access control

With `FRAUDLENS_API_KEYS` set, every endpoint except `/health` and `/metrics` needs an `X-API-Key` header.
A missing or unknown key gets **401**, a key whose role is not allowed gets **403** (counted in
`fraudlens_access_denied_total`). Generate keys with `python -m fraudlens.security keys`.

| Role | Who | May call |
|---|---|---|
| `scorer` | the payment system | `POST /score`, `POST /score/batch`, `GET /model` |
| `analyst` | fraud analysts, case tools | `GET /alerts`, `GET /alerts/{id}`, `POST /alerts/{id}/decision`, `POST /feedback`, `GET /monitoring/feedback`, `GET /model` |
| `admin` | operators | everything |

Every scoring call and decision is written to the case store's audit table with the caller's role and a
non-secret key id (decisions also carry the analyst's name). `/health` reports whether access control is on.

## Endpoints

| Method | Path | Role | Purpose |
|---|---|---|---|
| GET | `/health` | open | liveness: model, threshold, cards in the feature store, access-control mode |
| GET | `/model` | scorer, analyst | model metadata: champion, threshold and strategy, feature view, version |
| POST | `/score` | scorer | score one transaction and open a case if it alerts |
| POST | `/score/batch` | scorer | score up to 1,000 transactions (processed in time order) |
| GET | `/alerts` | analyst | the review queue (`status=open` by default; `closed`, `approved`) |
| GET | `/alerts/{case_id}` | analyst | one case: score, threshold, reasons, SHAP contributions, counterfactual, model inputs |
| POST | `/alerts/{case_id}/decision` | analyst | record the verdict; closes the case |
| POST | `/feedback` | analyst | verdict by transaction id (e.g. a chargeback on an approved transaction) |
| GET | `/monitoring/feedback` | analyst | what the decisions say: live precision with 95% CI, backlog, time to decision |
| GET | `/metrics` | open | Prometheus metrics |

### POST /score

Query parameters: `update_state` (default `true`: the card's history is updated after scoring, as in
production; use `false` for what-if calls), `include_features` (default `false`), `record` (default `true`:
store the transaction as a case).

Request (only `card_id`, `amount` and `category` are required; `ts` defaults to now):

```json
{
  "card_id": "3f2c9a1b7d4e5f60",
  "amount": 875.0,
  "category": "shopping_net",
  "merchant": "Example Store",
  "ts": "2020-12-31T23:30:00",
  "transaction_id": "pay-000123"
}
```

Response:

```json
{
  "transaction_id": "pay-000123",
  "card_id": "3f2c9a1b7d4e5f60",
  "ts": "2020-12-31 23:30:00",
  "amount": 875.0,
  "category": "shopping_net",
  "score": 0.99987,
  "alert": true,
  "threshold": 0.739727,
  "model": "hist_gb",
  "known_card": true,
  "reasons": ["Amount is 13.2x this card's usual spend", "Night-time transaction (23:00)"],
  "contributions": [{"feature": "amount", "value": 875.0, "shap": 4.21}, "..."],
  "clear_below_amount": 212.4,
  "case_id": "C-3a9f1c27d0",
  "routed_to": "review queue"
}
```

`clear_below_amount` (alerts only) is a counterfactual: a lower amount at which the same transaction, with the
same card history and time, would have been approved (found by bisection between $1 and the amount); `null` means even $1 would alert (the pattern, not the
amount, drives the alert). `known_card` is false for a card with no history (cold start); its history features
are neutral. Invalid input (e.g. a negative amount) returns HTTP 422.

### GET /alerts, GET /alerts/{case_id}

The queue lists open cases, highest score first: `case_id, received_at, ts, card_id, amount, category, score,
threshold, model, alert, status, reasons, contributions, clear_below_amount, source`. The single-case view adds
`features`, the exact model inputs used for the score.

### POST /alerts/{case_id}/decision

```json
{"label": "fraud", "analyst": "jsmith", "note": "card blocked after call with the customer"}
```

`label` is `fraud` or `legitimate`; `analyst` is required. Returns the closed case. Unknown case: 404.

### POST /feedback

```json
{"transaction_id": "pay-000123", "label": "fraud", "score": 0.998, "was_alert": true, "analyst": "jsmith", "note": "chargeback"}
```

Closes the matching case (or creates one for a transaction scored elsewhere) and appends to the feedback CSV.

### GET /monitoring/feedback

```json
{"scored": 1240, "alerts": 31, "reviewed_alerts": 27, "confirmed_fraud": 19, "live_precision": 0.7037,
 "live_precision_ci95": [0.5153, 0.8415], "fraud_reported_on_approved": 1, "open_alerts": 4,
 "oldest_open_hours": 3.2, "median_hours_to_decision": 0.6, "labelled_examples": 28,
 "policy": {"precision_floor": 0.4, "min_decisions": 20}, "verdict": "ok", "reason": "..."}
```

`verdict` is `review_threshold` when the whole 95% interval of live precision lies below the floor (with at
least 20 decisions), `insufficient` below 20 decisions, else `ok`.

### GET /metrics

| Metric | Type | Meaning |
|---|---|---|
| `fraudlens_transactions_scored_total` | counter | transactions scored |
| `fraudlens_alerts_total` | counter | alerts raised |
| `fraudlens_open_alerts` | gauge | alerts waiting for a decision |
| `fraudlens_reviewed_alerts` | gauge | alerts with a decision |
| `fraudlens_live_precision` | gauge | share of reviewed alerts confirmed as fraud |
| `fraudlens_decisions_total{label}` | counter | analyst decisions |
| `fraudlens_access_denied_total{endpoint}` | counter | requests refused by access control |
| `fraudlens_unknown_card_total` | counter | transactions from cards with no history |
| `fraudlens_feedback_total{label, was_alert}` | counter | verdicts received |
| `fraudlens_request_seconds{endpoint}` | histogram | request latency |
| `fraudlens_score` | histogram | score distribution |
| `fraudlens_requests_total{endpoint, status}` | counter | API requests |
| `fraudlens_amount_scored_dollars_total` | counter | dollar value scored |
| `fraudlens_model_threshold` | gauge | threshold in use |
| `fraudlens_model_info{champion, strategy}` | gauge | model identity |
| `fraudlens_monitoring_*` | gauges | latest monitored month: drift share, recall, precision, retrain recommended |

Prometheus alert rules (`platform/prometheus/alerts.yml`) fire on low live precision, a review backlog above
100, bursts of refused requests, a down API, slow scoring, an alert-rate spike and a retraining recommendation.

## Example (Python)

```python
import requests
API, S, A = "http://localhost:8000", {"X-API-Key": "<scorer key>"}, {"X-API-Key": "<analyst key>"}
r = requests.post(f"{API}/score", headers=S,
                  json={"card_id": "3f2c9a1b7d4e5f60", "amount": 875.0, "category": "shopping_net"}).json()
print(r["score"], r["alert"], r["reasons"], r["routed_to"])
if r["alert"]:
    requests.post(f"{API}/alerts/{r['case_id']}/decision", headers=A,
                  json={"label": "fraud", "analyst": "jsmith", "note": "confirmed"})
print(requests.get(f"{API}/monitoring/feedback", headers=A).json()["live_precision"])
```

A valid example transaction for a real card in the feature store is embedded in the OpenAPI schema
(`/openapi.json → components.schemas.Transaction.examples`), which the smoke tests use.
