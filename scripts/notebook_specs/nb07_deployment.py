FILENAME = "07_deployment_visualisation.ipynb"

CELLS = [
("md", r"""
# 7 · Deployment & visualisation

| | |
|---|---|
| Lifecycle stage | 7 of 8: *Deployment & visualisation* (dashboard, model API, user interaction and feedback) |
| Tools | FastAPI, Streamlit, Plotly, Docker, MinIO, Hive Metastore, Trino, ClickHouse, Superset, Grafana |
| Code | `api.py`, `runtime.py`, `online.py`, `scoring.py`, `dashboard/app.py`, `bundle.py`, `publish.py`, `crawler.py`, `olap.py` |

## 7.1 How FraudLens is served

```
                     ┌───────────────────── real time ──────────────────────┐
 card authorisation ─► FastAPI /score ─► online feature store (card history) ─► champion model ─► score, alert, reasons
                     │      ▲                                                                      │
 analyst verdict ─────► /feedback ──► feedback.csv ──► monitoring + next retraining                  ▼
                     │                                                         Prometheus /metrics ─► Grafana
                     └────────────────────────────────────────────────────────────────────────────┘
 batch (daily): pipeline ─► gold Parquet in MinIO ─► Hive Metastore (crawler) ─► Trino ─► Superset dashboards
                                                                       └──────► ClickHouse ─► Grafana business dashboard
 people: Streamlit dashboard (alert queue, live scoring, threshold & cost, patterns, monitoring, model, data)
```

Everything the API and the dashboard need is in the **deploy bundle** (`dashboard/bundle`, about 40 MB):
the `FraudModel` (preprocessor + champion + threshold), the online feature-store snapshot, the scored
transactions and the reports.
"""),
("setup", r"""
from fraudlens.runtime import example_transaction, load_runtime
bundle = ROOT / "dashboard" / "bundle"
manifest = read_json(bundle / "manifest.json")
print(f"bundle: {len(manifest['files'])} files, {manifest['total_mb']} MB, champion {manifest['champion']}, built {manifest['built_at']}")
print("pinned requirements:", ", ".join(manifest["requirements"]))"""),
("md", r"""
## 7.2 Real-time scoring with the online feature store

`load_runtime` loads the bundle exactly as the API does. A transaction arrives with only what the
authorisation system knows (card id, amount, category, time); the feature store derives the card-history
and velocity features from the card's previous transactions, the model scores it, and the alert comes with
reasons an analyst can read.
"""),
("code", r"""
rt = load_runtime(bundle)
txn = example_transaction(rt.store)
cases = {
    "large online purchase, 30 min after the card's last transaction": txn,
    "same card, small grocery purchase": {**txn, "amount": 12.5, "category": "grocery_pos"},
    "same card, large purchase at 2 am next day": {**txn, "amount": 950.0,
        "ts": (pd.Timestamp(txn["ts"]) + pd.Timedelta(days=1)).replace(hour=2, minute=0).isoformat()},
    "card never seen before (cold start)": {**txn, "card_id": "new-card-0001"},
}
rows = []
for label, t in cases.items():
    r = rt.scorer.score_transaction(t, update_state=False)
    rows.append({"case": label, "amount": r["amount"], "category": r["category"], "score": r["score"], "alert": r["alert"],
                 "known card": r["known_card"], "reasons": "; ".join(r["reasons"][:2])})
pd.DataFrame(rows)
"""),
("code", r"""
import time
t0 = time.perf_counter()
n = 200
for i in range(n):
    rt.scorer.score_transaction({**txn, "amount": 5 + i}, update_state=False)
ms = (time.perf_counter() - t0) / n * 1000
print(f"average in-process scoring latency incl. feature computation and SHAP reasons: {ms:.1f} ms per transaction")
"""),
("md", r"""
## 7.3 The model API (FastAPI)

Endpoints: `GET /health`, `GET /model`, `POST /score`, `POST /score/batch`, the review queue `GET /alerts`,
`GET /alerts/{id}`, `POST /alerts/{id}/decision`, `POST /feedback` (verdicts by transaction id),
`GET /monitoring/feedback` and `GET /metrics` (Prometheus). Interactive documentation is served at `/docs`.
Here the app is called in-process with FastAPI's test client, which runs exactly the same code as the
deployed container, with **role-based API keys switched on** (`scorer`, `analyst`).
"""),
("code", r"""
import importlib, tempfile
os.environ["FRAUDLENS_MODEL_DIR"] = str(bundle)
tmp = Path(tempfile.mkdtemp())
os.environ["FRAUDLENS_FEEDBACK_FILE"] = str(tmp / "feedback.csv")
os.environ["FRAUDLENS_CASE_DB"] = str(tmp / "cases.db")
os.environ["FRAUDLENS_API_KEYS"] = "scorer:demo-scorer-key,analyst:demo-analyst-key"
S, A = {"X-API-Key": "demo-scorer-key"}, {"X-API-Key": "demo-analyst-key"}
from fastapi.testclient import TestClient
import fraudlens.api as api
api = importlib.reload(api)
client = TestClient(api.app)

print("GET /health ->", client.get("/health").json())
print("POST /score without a key ->", client.post("/score", json=txn).status_code, "(401: access control)")
r = client.post("/score", json=txn, params={"update_state": False}, headers=S).json()
print("POST /score ->", json.dumps({k: r[k] for k in ("score", "alert", "threshold", "model", "reasons", "case_id", "routed_to")}, indent=1))
batch = client.post("/score/batch", json=[txn, {**txn, "amount": 8.0}], params={"update_state": False}, headers=S).json()
print("POST /score/batch ->", [(b["amount"], b["score"], b["alert"]) for b in batch])
print("POST /score with amount=-1 ->", client.post("/score", json={**txn, "amount": -1}, headers=S).status_code, "(rejected by validation)")
"""),
("md", r"""
### 7.3.1 A new transaction, end to end: features → inference → review queue → decision → monitoring

A card-testing burst on a known card: six rising amounts three minutes apart, sent to `POST /score` as the
payment system would. Each response shows the features the online store generated, the score against the
threshold and where the transaction was routed. Alerts wait in the review queue; an analyst opens one,
records a decision, and the decision flows into the feedback monitoring.
"""),
("code", r"""
card = txn["card_id"]
t0 = pd.Timestamp(txn["ts"])
rows = []
for i, amt in enumerate([2.0, 5.0, 25.0, 180.0, 650.0, 1200.0]):
    t = {**txn, "amount": amt, "ts": (t0 + pd.Timedelta(minutes=3 * i)).isoformat(), "transaction_id": f"burst-{i + 1}",
         "category": "shopping_net" if i > 2 else "misc_net", "merchant": "Burst Merchant"}
    out = client.post("/score", json=t, params={"include_features": True}, headers=S).json()
    f = out["features"]
    rows.append({"step": i + 1, "amount": amt, "txns past 1h": f["txn_count_1h"], "spend past 24h": f["amt_sum_24h"],
                 "amount / card mean": round(f["amt_to_card_mean"] or 0, 1), "score": out["score"],
                 "routed to": out["routed_to"], "case": out["case_id"]})
pd.DataFrame(rows)
"""),
("code", r"""
queue = client.get("/alerts", headers=A).json()
if not queue:   # no alert in the burst for this card: send one clear night-time outlier
    client.post("/score", headers=S, json={**txn, "amount": 1500.0, "category": "shopping_net", "transaction_id": "night-1",
                                           "ts": (t0.normalize() + pd.Timedelta(days=1, hours=2)).isoformat()})
    queue = client.get("/alerts", headers=A).json()
print(f"review queue: {len(queue)} open case(s)")
case = client.get(f"/alerts/{queue[0]['case_id']}", headers=A).json()
print(json.dumps({k: case[k] for k in ("case_id", "amount", "score", "threshold", "reasons", "clear_below_amount")}, indent=1))
print("top contributions:", [(c["feature"], c["shap"]) for c in case["contributions"][:3]])
print("scorer key on the queue ->", client.get("/alerts", headers=S).status_code, "(403: wrong role)")
d = client.post(f"/alerts/{case['case_id']}/decision", headers=A,
                json={"label": "fraud", "analyst": "notebook-analyst", "note": "customer confirmed card was stolen"}).json()
print("decision ->", d["case"])
fb = client.get("/monitoring/feedback", headers=A).json()
print("feedback monitoring ->", {k: fb[k] for k in ("reviewed_alerts", "confirmed_fraud", "live_precision",
                                                    "live_precision_ci95", "open_alerts", "labelled_examples", "verdict")})
print(api.CASES.audit_log()[["at", "actor", "role", "action", "case_id"]].head(4).to_string(index=False))
metrics = client.get("/metrics").text
print("\n".join(l for l in metrics.splitlines() if l.startswith(("fraudlens_open_alerts", "fraudlens_reviewed_alerts",
                                                                 "fraudlens_live_precision", "fraudlens_access_denied_total"))))
del os.environ["FRAUDLENS_API_KEYS"]
"""),
("md", r"""
The analyst's decision is a label. With at least 20 decisions, live precision is compared with the 40% policy
floor; `python -m fraudlens.monitor` adds this block to the monitoring summary and exports the decided cases
(with the exact model inputs used) as labelled examples for the next retraining.

## 7.4 The dashboard (Streamlit + Plotly)

`dashboard/app.py` is the analysts' and managers' interface. It runs from the full pipeline output locally,
or from the bundle when hosted (Streamlit Community Cloud or the Docker image). Streamlit's own test harness
renders it headlessly here to prove it starts cleanly from the bundle.
"""),
("code", r"""
from streamlit.testing.v1 import AppTest
at = AppTest.from_file(str(CODE / "dashboard" / "app.py"), default_timeout=300)
at.run()
print("exceptions:", [e.message for e in at.exception] or "none")
print("tabs:", [t.label for t in at.tabs])
print("headline metrics:", [(m.label, m.value) for m in at.metric][:8])
"""),
("md", r"""
| Tab | Who uses it | What it does |
|---|---|---|
| Overview | manager | KPIs for any period and split, daily alerts and outcomes, money caught and missed, estimated net benefit with its assumptions |
| Live transaction | analyst / demo | a new transaction end to end: received → features → score and explanation → routed to review |
| Case review | analyst | the review queue of live alerts; record decisions; how decisions feed monitoring |
| Alert history | analyst | the replayed test period by card, with SHAP reasons |
| Threshold & cost | manager | move the threshold or the cost assumptions and see recall, precision, alerts/day and estimated net benefit change |
| Patterns | analyst | fraud by category, hour, amount band and state |
| Monitoring | data scientist | monthly drift and performance, monthly stability with intervals, live feedback, retraining recommendation |
| Model | governance | model comparison, selection rule, SHAP importance, explanation quality, transferability to real data |
| Data pipeline | data engineering | lineage from raw to gold, data quality and verification results |
| Report | manager | a plain-language summary of the selected period, downloadable |

## 7.5 Serving the gold layer to BI: lake → catalog → Trino / ClickHouse → Superset / Grafana

`publish.py` writes Hive-partitioned Parquet to MinIO, `crawler.py` registers every table in the Hive
Metastore through Trino (the Glue-crawler role), and `olap.py` loads the gold layer into ClickHouse and
answers the same business questions on both engines. Superset dashboards read Trino; the Grafana business
dashboard reads ClickHouse.
"""),
("code", r"""
from fraudlens.crawler import VIEWS
from fraudlens.olap import QUERIES
print("Trino views created by the crawler for BI:")
for name, body in VIEWS.items():
    print(f"  hive.{name}: {' '.join(body.split())[:110]}...")
print("\nBusiness questions answered on both Trino and ClickHouse:")
for name, q in QUERIES.items():
    print(f"  {name}: {q['question']}")
for f in ["lake_publication.json", "data_catalog.json", "olap_benchmark.json"]:
    p = P["reports"] / f
    if p.exists():
        r = read_json(p)
        print(f"\n{f}: status={r.get('status')} {r.get('reason', '')}")
"""),
("md", r"""
On a laptop without Docker the platform stages skip themselves (status `skipped` above); the
`platform` workflow runs them for real on a GitHub runner (NiFi → MinIO → Hive Metastore → Trino →
ClickHouse, Superset, Grafana) and publishes the results as a build artefact.

## 7.6 Deployment targets

| Target | How | Command |
|---|---|---|
| Local, no Docker | API + dashboard from the virtual environment | `uvicorn fraudlens.api:app --port 8000` and `streamlit run dashboard/app.py` |
| Container image | `platform/docker/Dockerfile.app`, published to GHCR by `deploy.yml` | `docker run -p 8000:8000 ghcr.io/anasua2905/fraudlens-app api` |
| Full platform | `platform/docker-compose.yml` profiles `lake ingest olap bi mlops` | `docker compose --profile lake --profile mlops up -d` |
| Hosted dashboard | Streamlit Community Cloud from the GitHub repository (`dashboard/app.py`, uses the bundle) | see `docs/deployment.md` |

**Next:** [08 · Monitoring & improvement](08_monitoring_improvement.ipynb)
"""),
]
