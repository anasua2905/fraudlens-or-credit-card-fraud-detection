# Monitoring runbook

What to watch, what each alert means, and how to retrain and promote a model safely.

## 1. What is monitored

| Signal | Source | Cadence | Threshold (config `monitoring`) |
|---|---|---|---|
| Feature drift (share of monitored features drifting) | Evidently `DataDriftPreset` vs the training reference; PSI cross-check | monthly (`python -m fraudlens.monitor`) | > 0.30 → retraining recommended |
| Recall at the deployed threshold | labelled outcomes (chargebacks, analyst feedback) | monthly | < 0.85 → retraining recommended |
| Precision at the deployed threshold | labelled outcomes | monthly | < 0.40 → retraining recommended |
| Alert rate | API `/metrics` → Prometheus | continuous | > 5% of scored transactions for 10 min (`FraudLensAlertRateSpike`) |
| Scoring latency p95 | API `/metrics` | continuous | > 250 ms for 5 min (`FraudLensSlowScoring`) |
| API availability | Prometheus `up` | continuous | down for 1 min (`FraudLensApiDown`, critical) |
| Monitoring verdict | `fraudlens_monitoring_retrain_recommended` gauge | after each monitoring run | = 1 for 5 min (`FraudLensRetrainRecommended`) |
| Cold-start share | `fraudlens_unknown_card_total` | continuous | review if it rises sharply (new portfolio, data issue) |

Dashboards: Grafana *FraudLens - operations* (Prometheus) and *FraudLens - Business analytics* (ClickHouse);
the Streamlit **Monitoring** tab; MLflow (monitoring runs, one step per month).

## 2. Monthly routine

```bash
python -m fraudlens.run_all --only score monitor        # after new labelled data has been prepared
```

1. Read `reports/monitoring/summary.json` (or the Monitoring tab): latest month, drift share, recall, precision,
   `retrain_recommended` and `reasons`.
2. Open the month's Evidently report (`reports/monitoring/drift_<month>.html`) and check *which* features moved.
3. Decide:
   * **drift only, performance fine** (e.g. Aug and Dec 2020: seasonal spending): note it, keep the champion,
     check again next month; consider retraining if it persists for two months;
   * **performance below a floor**: retrain now (section 3);
   * **data problem** (a feature drifts to impossible values, null spike): fix the pipeline first; do not retrain
     on broken data.

## 3. Retraining and promotion

```bash
python -m fraudlens.monitor --retrain-if-needed          # retrains automatically when the policy fires
# or manually:
python -m fraudlens.run_all --start preprocess            # refit preprocessing and models on the latest data
```

Promotion rule (same as champion selection): the challenger replaces the champion only if its PR-AUC on the
latest labelled period is **non-inferior** (95% paired-bootstrap upper bound of the gap ≤ 0.01) and it meets the
success criteria at the agreed alert budget. Notebook 08 (section 8.4) shows a worked example.

After promotion:

1. MLflow: the new version of `fraudlens-champion` gets the alias `champion` (done by `train.py`).
2. `python -m fraudlens.run_all --only verify bundle` → all verification checks must pass.
3. Commit the new `dashboard/bundle`, push; the `deploy` workflow builds and publishes a new image.
4. Restart the API (`docker compose up -d api`, or redeploy the image); check `/health` shows the new threshold.

Rollback: re-deploy the previous image tag (`ghcr.io/anasua2905/fraudlens-app:sha-<previous>`), or move the MLflow
alias back to the previous version and rebuild the bundle.

## 4. Alert response

| Alert | First checks | Action |
|---|---|---|
| `FraudLensApiDown` | `docker compose ps api`, `docker compose logs api` | restart; if the bundle is missing, rebuild it |
| `FraudLensSlowScoring` | CPU load, batch sizes on `/score/batch` | scale out the API (more replicas) or disable live SHAP contributions |
| `FraudLensAlertRateSpike` | Grafana score distribution; recent deploys; category mix | an attack in progress (escalate to fraud ops) or an input-data problem (check upstream) |
| `FraudLensRetrainRecommended` | `reports/monitoring/summary.json → reasons` | follow sections 2-3 |

## 5. Feedback loop

Analysts' verdicts arrive at `POST /feedback` (and from the dashboard's Alert queue) and are appended to the
feedback file. They are the fastest source of labels, but only for alerted transactions; combine them with
chargeback labels for non-alerted transactions before retraining, otherwise the training data becomes biased
towards what the old model already flagged.
