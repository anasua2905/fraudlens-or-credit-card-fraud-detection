# Scoring API reference

FastAPI service (`src/fraudlens/api.py`). Interactive docs at `http://<host>:8000/docs` (Swagger) and `/redoc`.

Start it:

```bash
uvicorn fraudlens.api:app --host 0.0.0.0 --port 8000          # from the repository (uses the pipeline output or the bundle)
docker run -p 8000:8000 ghcr.io/anasua2905/fraudlens-app api   # the published image
```

Environment: `FRAUDLENS_MODEL_DIR` (bundle or project folder; default: the repository), `FRAUDLENS_FEEDBACK_FILE`
(where analyst verdicts are appended; default `data/feedback/feedback.csv`).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness: model, threshold, number of cards in the feature store |
| GET | `/model` | model metadata: champion, threshold and strategy, feature view, version |
| POST | `/score` | score one transaction |
| POST | `/score/batch` | score up to 1,000 transactions (processed in time order) |
| POST | `/feedback` | record an analyst's verdict on a transaction |
| GET | `/metrics` | Prometheus metrics |

### POST /score

Query parameters: `update_state` (default `true`: the card's history is updated after scoring, as in
production; use `false` for what-if calls), `include_features` (default `false`).

Request (only `card_id`, `amount` and `category` are required; `ts` defaults to now):

```json
{
  "card_id": "3f2c9a1b7d4e5f60",
  "amount": 875.0,
  "category": "shopping_net",
  "merchant": "Example Store",
  "ts": "2020-12-31T23:30:00"
}
```

Response:

```json
{
  "transaction_id": null,
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
  "contributions": [{"feature": "amount", "value": 875.0, "shap": 4.21}, "..."]
}
```

`known_card` is false for a card with no history (cold start); its history features are neutral.
Invalid input (e.g. a negative amount) returns HTTP 422 with the validation error.

### POST /feedback

```json
{"transaction_id": "f5734f41ee6d1da5", "label": "fraud", "score": 0.998, "was_alert": true, "analyst": "jsmith", "note": "confirmed with customer"}
```

`label` is `fraud` or `legitimate`. Returns `{"status": "recorded", "file": "feedback.csv"}`.

### GET /metrics

| Metric | Type | Meaning |
|---|---|---|
| `fraudlens_transactions_scored_total` | counter | transactions scored |
| `fraudlens_alerts_total` | counter | alerts raised |
| `fraudlens_unknown_card_total` | counter | transactions from cards with no history |
| `fraudlens_feedback_total{label, was_alert}` | counter | analyst verdicts |
| `fraudlens_request_seconds{endpoint}` | histogram | request latency |
| `fraudlens_score` | histogram | score distribution |
| `fraudlens_requests_total{endpoint, status}` | counter | API requests |
| `fraudlens_amount_scored_dollars_total` | counter | dollar value scored |
| `fraudlens_model_threshold` | gauge | threshold in use |
| `fraudlens_model_info{champion, strategy}` | gauge | model identity |
| `fraudlens_monitoring_*` | gauges | latest monitored month: drift share, recall, precision, retrain recommended |

## Example (Python)

```python
import requests
r = requests.post("http://localhost:8000/score", params={"update_state": "false"},
                  json={"card_id": "3f2c9a1b7d4e5f60", "amount": 875.0, "category": "shopping_net"})
print(r.json()["score"], r.json()["alert"], r.json()["reasons"])
```

A valid example transaction for a real card in the feature store is embedded in the OpenAPI schema
(`/openapi.json → components.schemas.Transaction.examples`), which the smoke tests use.
