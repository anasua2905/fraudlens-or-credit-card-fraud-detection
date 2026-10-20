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

Endpoints: `GET /health`, `GET /model`, `POST /score`, `POST /score/batch`, `POST /feedback` (analyst
verdicts) and `GET /metrics` (Prometheus). Interactive documentation is served at `/docs`. Here the app is
called in-process with FastAPI's test client, which runs exactly the same code as the deployed container.
"""),
("code", r"""
import importlib, tempfile
os.environ["FRAUDLENS_MODEL_DIR"] = str(bundle)
os.environ["FRAUDLENS_FEEDBACK_FILE"] = str(Path(tempfile.mkdtemp()) / "feedback.csv")
from fastapi.testclient import TestClient
import fraudlens.api as api
api = importlib.reload(api)
client = TestClient(api.app)

print("GET /health ->", client.get("/health").json())
r = client.post("/score", json=txn, params={"update_state": False}).json()
print("POST /score ->", json.dumps({k: r[k] for k in ("score", "alert", "threshold", "model", "reasons")}, indent=1))
batch = client.post("/score/batch", json=[txn, {**txn, "amount": 8.0}], params={"update_state": False}).json()
print("POST /score/batch ->", [(b["amount"], b["score"], b["alert"]) for b in batch])
print("POST /feedback ->", client.post("/feedback", json={"transaction_id": "demo-1", "label": "fraud", "was_alert": True,
                                                          "score": r["score"], "analyst": "notebook"}).json())
print("POST /score with amount=-1 ->", client.post("/score", json={**txn, "amount": -1}).status_code, "(rejected by validation)")
metrics = client.get("/metrics").text
print("\n".join(l for l in metrics.splitlines() if l.startswith(("fraudlens_transactions_scored_total", "fraudlens_alerts_total", "fraudlens_feedback_total"))))
"""),
("md", r"""
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
| Overview | manager | KPIs for any period and split, daily alerts and outcomes, money caught and missed |
| Alert queue | analyst | ranked cards and transactions with SHAP reasons; record a verdict (feedback) |
| Live scoring | analyst / demo | type a transaction, get score, alert and reasons from the same runtime as the API |
| Threshold & cost | manager | move the threshold or the review cost and see recall, precision, alerts/day and net benefit change |
| Patterns | analyst | fraud by category, hour, amount band and state |
| Monitoring | data scientist | monthly drift and performance, retraining recommendation |
| Model | governance | model comparison, selection rule, SHAP importance, test metrics |
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
