FILENAME = "08_monitoring_improvement.ipynb"

CELLS = [
("md", r"""
# 8 · Monitoring & improvement

| | |
|---|---|
| Lifecycle stage | 8 of 8: *Monitoring & improvement* (track performance, detect drift, retrain) |
| Tools | Evidently AI, Prometheus, Grafana, MLflow |
| Code | `monitor.py`, `drift.py`, `tracking.py`, `api.py` (`/metrics`, `/feedback`), `platform/prometheus`, `platform/grafana` |
| Outputs | `reports/monitoring/summary.json`, `monthly.csv`, `feature_drift.csv`, Evidently HTML report per month, MLflow monitoring run |

Monitoring replays the validation and test periods **month by month as if the model were live** and asks:

1. **Data drift**: has the input distribution moved away from the training data? (Evidently `DataDriftPreset`,
   with the project's own PSI as a cross-check)
2. **Performance**: at the fixed threshold, are recall, precision and alert volume still acceptable?
   (in production the labels arrive later, from chargebacks and analyst feedback via `/feedback`)

A **retraining recommendation** is raised when more than 30% of monitored features drift, recall falls below
0.85 or precision below 0.40 (`config/pipeline.yaml → monitoring`).
"""),
("setup", r"""
from fraudlens import monitor
from fraudlens.drift import psi
summ = read_json(P["monitoring"] / "summary.json")
monthly = pd.DataFrame(summ["monthly"])
print(f"drift engine: {summ['drift_engine']}; reference: {summ['reference']}; policy: {summ['policy']}")"""),
("md", r"""
## 8.1 Month by month: drift and performance
"""),
("code", r"""
cols = ["month", "transactions", "frauds", "evidently_drift_share", "psi_major_share", "pr_auc", "recall",
        "precision", "alerts_per_day", "net_benefit", "retrain_recommended", "reasons"]
monthly[cols]
"""),
("code", r"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=("Performance at the fixed threshold", "Share of monitored features drifting"))
fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["recall"], name="recall", mode="lines+markers", line_color=viz.SERIES[0]), 1, 1)
fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["precision"], name="precision", mode="lines+markers", line_color=viz.SERIES[1]), 1, 1)
fig.add_hline(y=summ["policy"]["recall_floor"], line_dash="dash", line_color=viz.MUTED, row=1, col=1)
share = monthly["evidently_drift_share"].fillna(monthly["psi_major_share"])
fig.add_trace(go.Bar(x=monthly["month"], y=share, name="drift share",
                     marker_color=[viz.STATUS["missed"] if r else viz.SERIES[0] for r in monthly["retrain_recommended"]]), 1, 2)
fig.add_hline(y=summ["policy"]["drift_share_alert"], line_dash="dash", line_color=viz.MUTED, row=1, col=2)
fig.update_layout(template=viz.plotly_template(), height=380, yaxis_range=[0, 1.05])
fig.show()
if (P["figures"] / "19_monitoring_monthly.png").exists():
    display(Image(filename=str(P["figures"] / "19_monitoring_monthly.png"), width=800))
"""),
("md", r"""
### 8.1.1 Is the monthly performance stable, or just noisy?

Each month has only 130 to 400 frauds, so a monthly recall is itself uncertain. `fraudlens.stability` adds a
Wilson 95% interval to every month, checks each month against the policy floors, and tests with a chi-square
test whether the months differ more than chance would allow.
"""),
("code", r"""
from fraudlens.stability import run_stability
stab = run_stability(cfg)
m = pd.DataFrame(stab["monthly"]["months"])
fig = go.Figure()
for name, col, color in [("recall", "recall", viz.SERIES[0]), ("precision", "precision", viz.SERIES[1])]:
    ci = np.array(m[f"{col}_ci95"].tolist())
    fig.add_trace(go.Scatter(x=m["month"], y=m[col], name=name, mode="lines+markers", line_color=color,
                             error_y=dict(type="data", symmetric=False, array=ci[:, 1] - m[col], arrayminus=m[col] - ci[:, 0])))
fig.add_hline(y=summ["policy"]["recall_floor"], line_dash="dash", line_color=viz.MUTED)
fig.add_hline(y=summ["policy"]["precision_floor"], line_dash="dot", line_color=viz.MUTED)
fig.update_layout(template=viz.plotly_template(), height=360, yaxis_range=[0.3, 1.02],
                  title="Monthly recall and precision with 95% Wilson intervals (test period)")
fig.show()
pd.DataFrame(stab["monthly"]["checks"])
"""),
("md", r"""
### 8.1.2 How far would these numbers transfer to real transactions?

The data are simulated. Precision depends on how common fraud is and on how many false alarms real
customers trigger, so the test recall and false-positive rate are carried over to other conditions:
"""),
("code", r"""
pv = pd.DataFrame(stab["prevalence"]["rows"])
print(stab["prevalence"]["note"])
pv[["fraud_rate_pct", "is_test_rate", "precision_fpr_x1", "alerts_per_day_fpr_x1", "precision_fpr_x2", "precision_fpr_x5"]]
"""),
("md", r"""
## 8.2 Which features drift?

Per-feature PSI per month (0.1-0.25 = moderate, > 0.25 = major) with Evidently's drift verdicts.
"""),
("code", r"""
fd = pd.read_csv(P["monitoring"] / "feature_drift.csv")
pivot = fd.pivot(index="feature", columns="month", values="psi")
display(pivot.style.background_gradient(cmap="Reds", vmin=0, vmax=0.3).format("{:.3f}"))
ev = fd.pivot(index="feature", columns="month", values="evidently_drifted")
ev
"""),
("md", r"""
## 8.3 Evidently, live

The monthly HTML reports are in `reports/monitoring/drift_<month>.html` (December 2020 is committed to the
repository as an example). Here the same `DataDriftPreset` is run again for the latest month, directly from
the notebook.
"""),
("code", r"""
from fraudlens.datasets import load_features
from fraudlens.preprocess import add_stationary_features
feats = cfg["monitoring"]["features"]
categorical = [c for c in feats if c in ("category", "gender")]
numeric = [c for c in feats if c not in categorical]
ref = monitor._reference(cfg, feats, n=50_000)
cur = add_stationary_features(load_features(cfg, "test"))
cur = cur[cur["ts"].dt.strftime("%Y-%m") == summ["latest_month"]][feats]
for c in categorical:
    ref[c] = ref[c].astype(str); cur[c] = cur[c].astype(str)
out = monitor.evidently_drift(ref, cur, numeric, categorical, html_path=None)
print(f"{summ['latest_month']}: {out['count']} of {len(feats)} monitored features drifted (share {out['share']:.2f})")
pd.DataFrame(out["columns"]).T.assign(psi=lambda t: [round(psi(ref[c], cur[c]), 4) if c in numeric else None for c in t.index])
"""),
("md", r"""
**Reading the result.** The drift flags in August and December are driven by *seasonality* (holiday
spending shifts amounts and times of day), while recall stays well above the 85% floor and precision above
40%. The policy correctly raises a recommendation and a person decides; the next section shows what that
retraining would give.

## 8.4 Improvement: a retrained challenger

Champion/challenger test. The challenger is the **same algorithm and parameters**, retrained on train **plus
the six more recent validation weeks**. It is compared with the champion on the test period; it would be
promoted only if it is at least as good (non-inferior PR-AUC) at the same alert budget.

Here the test period is used for the comparison only to *illustrate* the loop; in production the comparison
runs on the next labelled period, and the reported champion keeps its clean, untouched test evaluation.
"""),
("code", r"""
import time
from sklearn.metrics import average_precision_score
from fraudlens import models as M
from fraudlens import evaluate as E
from fraudlens.scoring import FraudModel
from fraudlens.train import load_view

res = read_json(P["reports"] / "model_results.json")
fm = FraudModel.load(P["artifacts"] / "fraudlens_model.joblib")
d = load_view(cfg, fm.view)
cat = [c for c in M.CATEGORICAL_IN_TREE_VIEW if c in d["train"]["X"].columns]
X_new = pd.concat([d["train"]["X"], d["val"]["X"]], ignore_index=True)
y_new = np.concatenate([d["train"]["y"], d["val"]["y"]])
t0 = time.perf_counter()
challenger = M.fit(M.build(fm.champion, 42, categorical=cat, params=res["tuning"].get(fm.champion, {}).get("best_params")),
                   fm.champion, X_new, y_new, cat)
print(f"challenger trained on {len(X_new):,} rows in {time.perf_counter() - t0:.0f}s")

te = d["test"]
s_champ = M.positive_scores(fm.model, te["X"])
s_chal = M.positive_scores(challenger, te["X"])
budget = int((s_champ >= fm.threshold).sum())          # same number of alerts for a fair comparison
thr_chal = np.sort(s_chal)[-budget]
rows = []
for name, sc, thr in (("champion (train)", s_champ, fm.threshold), ("challenger (train + val)", s_chal, thr_chal)):
    m = E.evaluate(te["y"], sc, te["amount"], thr, te["days"], cfg["model"]["threshold"])
    rows.append({"model": name, "pr_auc": m["pr_auc"], "recall": m["recall"], "precision": m["precision"],
                 "alerts_per_day": m["alerts_per_day"], "net_benefit": m["net_benefit"]})
cmp = pd.DataFrame(rows).set_index("model")
cmp
"""),
("code", r"""
from fraudlens.selection import paired_bootstrap_gap
gap = paired_bootstrap_gap(te["y"], s_chal, s_champ, n_boot=300, seed=0)
print("PR-AUC gap champion vs challenger (positive = challenger better):", gap)
promote = gap["ci95_low"] > -0.01            # challenger non-inferior to the champion within the 0.01 margin
print("Promote the challenger:", promote)
with open(P["monitoring"] / "challenger.json", "w") as fh:
    json.dump({"comparison": cmp.reset_index().to_dict("records"), "pr_auc_gap_challenger_minus_champion": gap,
               "promote": bool(promote), "rule": "non-inferior PR-AUC (95% bootstrap lower bound of the gap > -0.01)"}, fh, indent=1)
"""),
("md", r"""
## 8.5 Operational monitoring: Prometheus and Grafana

The API exports Prometheus metrics at `/metrics`: transactions scored, alerts raised, latency histogram,
score distribution, cards with no history, feedback received, and the monitoring verdict
(`fraudlens_monitoring_retrain_recommended`). Prometheus scrapes it every 15 s and evaluates these alert rules;
Grafana shows two provisioned dashboards (operations from Prometheus, business analytics from ClickHouse).
"""),
("code", r"""
print((CODE / "platform" / "prometheus" / "alerts.yml").read_text())
for f in ["fraudlens.json", "fraudlens_business.json"]:
    g = json.loads((CODE / "platform" / "grafana" / "dashboards" / f).read_text())
    print(f"Grafana '{g['title']}': " + ", ".join(p["title"] for p in g["panels"]))
"""),
("code", r"""
import importlib, tempfile
os.environ["FRAUDLENS_MODEL_DIR"] = str(ROOT / "dashboard" / "bundle")
os.environ["FRAUDLENS_FEEDBACK_FILE"] = str(Path(tempfile.mkdtemp()) / "feedback.csv")
from fastapi.testclient import TestClient
from fraudlens.runtime import example_transaction
import fraudlens.api as api
api = importlib.reload(api)
c = TestClient(api.app)
txn = example_transaction(api.RT.store)
for amt in (5, 40, 120, 650, 980):
    c.post("/score", json={**txn, "amount": float(amt)}, params={"update_state": False})
c.post("/feedback", json={"transaction_id": "t-1", "label": "fraud", "was_alert": True})
print("\n".join(l for l in c.get("/metrics").text.splitlines()
                if l.startswith("fraudlens_") and not l.startswith(("fraudlens_request_seconds_bucket", "fraudlens_score_bucket"))))
"""),
("md", r"""
## 8.6 MLflow: monitoring runs next to training runs

Each monitoring pass is logged to MLflow as its own run, with one step per month, so drift share, recall and
precision can be plotted over time in the MLflow UI next to the training runs and the registered champion.
"""),
("code", r"""
import mlflow
db = ROOT / "mlruns" / "mlflow.db"
if db.exists():
    mlflow.set_tracking_uri(f"sqlite:///{db.as_posix()}")
    runs = mlflow.search_runs(experiment_names=[cfg["platform"]["mlflow_experiment"]],
                              filter_string="tags.stage = 'monitoring'", max_results=5)
    display(runs[[c for c in runs.columns if c in ("run_id", "start_time", "metrics.drift_share", "metrics.recall",
                                                       "metrics.precision", "tags.retrain_recommended")]])
else:
    print("no local MLflow store in this environment")
"""),
("md", r"""
## 8.7 The improvement loop

```
 live traffic ─► /score ─► case ─► review queue ─► analyst decision ─► audit log
                    │                                        │
                    │                                        ├─► live precision vs floor (/monitoring/feedback)
                    │                                        └─► labelled examples ─┐
                    │                                              ▼
                    └─► /metrics ─► Prometheus alert rules    labelled outcomes
                                                                   │
 monthly: python -m fraudlens.monitor ─► drift + performance ─► retrain recommended?
                                                                   │ yes (a person approves)
 python -m fraudlens.run_all --start train ─► challenger ─► compare with champion (8.4)
        └─► promote: MLflow registry alias `champion`, rebuild the bundle, redeploy the image
```

* Analyst decisions are stored with the exact model inputs (`fraudlens.casework`); `python -m fraudlens.monitor`
  adds live precision, backlog and time to decision to the summary and exports the decided cases as labelled
  examples. A whole 95% interval of live precision below the floor raises a threshold/model review.
* `python -m fraudlens.monitor --retrain-if-needed` automates the retraining step when the policy fires.
* The rule for promotion is the same non-inferiority test used for selection, so a retrained model only
  replaces the champion when it is not worse on the latest labelled period.

**This completes the lifecycle.** Start again at [01 · Problem understanding](01_problem_understanding.ipynb)
with the next question: should the threshold adapt to seasonal volume?
"""),
]
