FILENAME = "01_problem_understanding.ipynb"

CELLS = [
("md", r"""
# 1 · Problem understanding

**FraudLens: an anomaly detection framework for fraudulent credit card transactions**

| | |
|---|---|
| Lifecycle stage | 1 of 8: *Problem understanding* (define the problem, goals and success metrics) |
| Tools | Jupyter, Markdown, pandas |
| Inputs | `config/pipeline.yaml`, the acquisition manifest |
| Outputs | Problem statement, success criteria, cost model, hypotheses that drive the next stages |

## 1.1 The business problem

A card issuer has to decide, for every transaction, within milliseconds, whether to let it through
quietly or raise an alert for a fraud analyst to review. Fraud is **rare** (about 1 in 170 training
transactions) but **expensive**, and every alert costs analyst time and annoys a genuine customer.

The task is therefore not "classify transactions" but **rank transactions by fraud risk and choose
an operating point** that a fraud-operations team can live with:

* catch as much fraud (and as many fraud dollars) as possible,
* keep the alert queue small enough for the team to clear each day,
* explain every alert so an analyst can act on it, and
* keep working as customer behaviour changes (monitor drift, retrain).

## 1.2 Stakeholders and what they need

| Stakeholder | Question they ask | What FraudLens gives them |
|---|---|---|
| Fraud analyst | *Which alerts do I look at first, and why were they raised?* | Ranked alert queue with the top SHAP reasons per alert (dashboard, API) |
| Fraud-operations manager | *How many alerts per day? What does it cost? What do we save?* | Alerts/day, precision, recall, review cost and net benefit at the chosen threshold |
| Risk / model governance | *Is the model fair, stable, documented and reproducible?* | Model card, verification report, drift monitoring, MLflow registry, data catalog with PII tags |
| Platform / data engineering | *Can it run on our open-source stack?* | MinIO lake, NiFi ingest, Hive Metastore, Trino, ClickHouse, Superset, Grafana, Docker |
| Card holders | *Will my genuine purchases be blocked?* | False-alarm rate and false alarms per fraud caught are tracked explicitly |
"""),
("setup", ""),
("md", r"""
## 1.3 Success criteria (defined *before* modelling)

These targets are written in `config/pipeline.yaml` and checked automatically by the verification
stage on the untouched **test period** (the last six months of 2020). They encode the business needs
above: high recall, a manageable queue, and precision high enough that analysts trust alerts.
"""),
("code", r"""
crit = pd.DataFrame([
    {"criterion": "min_test_recall", "target": cfg["success_criteria"]["min_test_recall"],
     "meaning": "share of fraudulent transactions that raise an alert"},
    {"criterion": "min_test_precision", "target": cfg["success_criteria"]["min_test_precision"],
     "meaning": "share of alerts that are real fraud"},
    {"criterion": "max_alerts_per_day", "target": cfg["success_criteria"]["max_alerts_per_day"],
     "meaning": "alerts a small review team can clear in a day"},
    {"criterion": "min_pr_auc_lift", "target": cfg["success_criteria"]["min_pr_auc_lift"],
     "meaning": "PR-AUC divided by the fraud rate (random ranking = 1)"},
])
crit
"""),
("md", r"""
## 1.4 The data in one glance

The project uses the **Sparkov** simulated card-transaction dataset (Kaggle `kartik2112/fraud-detection`,
licence CC0): 1.85 million transactions by 1,000 customers at 800 merchants, 1 Jan 2019 to 31 Dec 2020,
each labelled `is_fraud`. It is simulated, which makes it shareable and free of real personal data,
but it also means patterns are cleaner than in production; we keep that in mind when interpreting results.
"""),
("code", r"""
man = read_json(P["reports"] / "acquisition_manifest.json")
run = read_json(P["reports"] / "pipeline_run.json")
files = pd.DataFrame(man["files"]).T[["file", "rows", "bytes"]]
files["MB"] = (files["bytes"].astype(float) / 1e6).round(1)
display(files.drop(columns="bytes"))
splits = pd.DataFrame(run["split"]["splits"]).T
splits[["rows", "fraud_count", "start", "end"]]
"""),
("md", r"""
## 1.5 Why accuracy is the wrong metric

With a fraud rate below 1%, a "model" that never raises an alert is more than 99% accurate and
completely useless. The project therefore measures:

* **PR-AUC** (average precision): ranking quality on the rare class, compared with its random baseline
  (the fraud rate). Reported as *lift* = PR-AUC / fraud rate.
* **Recall** and **precision** at the operating threshold, **alerts per day**, and
* **Net benefit** = 90% of the fraud dollars caught − $15 per alert reviewed.
"""),
("code", r"""
test = read_table(P["gold"] / "features_test")[["is_fraud", "amount"]]
always_legit_accuracy = 1 - test["is_fraud"].mean()
print(f"Test transactions: {len(test):,}   fraud rate: {test['is_fraud'].mean():.3%}")
print(f"Accuracy of 'never alert': {always_legit_accuracy:.2%}  -> recall 0%, fraud dollars caught $0")
print(f"Random-ranking PR-AUC (the baseline every model is compared with): {test['is_fraud'].mean():.4f}")
"""),
("md", r"""
## 1.6 Cost model and the decision rule

An alert costs about **$15** of analyst time (`investigation_cost`) and, when it is real fraud, recovers
about **90%** of the transaction amount (`recovery_rate`). So an alert on a transaction of amount *a* with
fraud probability *p* is worth raising when

$$0.9 \cdot p \cdot a > 15$$

Small transactions are rarely worth an alert even at high probability; large ones are worth it even at
moderate probability. The modelling stage chooses the threshold that **maximises net benefit on the
validation period** (`threshold.primary: min_cost`) and then applies it unchanged to the test period.
"""),
("code", r"""
tcfg = cfg["model"]["threshold"]
amounts = np.array([10, 50, 100, 250, 500, 1000])
probs = np.array([0.05, 0.1, 0.25, 0.5, 0.9])
value = pd.DataFrame([[tcfg["recovery_rate"] * p * a - tcfg["investigation_cost"] for a in amounts] for p in probs],
                     index=[f"p(fraud)={p:.2f}" for p in probs], columns=[f"${a}" for a in amounts]).round(1)
print("Expected value of raising one alert ($): positive = worth reviewing")
value.style.background_gradient(cmap="RdYlGn", axis=None).format("{:.1f}")
"""),
("md", r"""
## 1.7 Framing: supervised detection plus anomaly detection

Labels exist, so **supervised** gradient-boosted trees are the main candidates. Fraud patterns also change,
and new kinds of fraud have no labels yet, so an **unsupervised anomaly detector** (Isolation Forest,
trained on legitimate transactions only) is evaluated on its own and as an extra feature in a **hybrid**
model. The champion is selected by a pre-registered rule: the *simplest* model that is statistically
non-inferior to the best one on validation PR-AUC.

## 1.8 Hypotheses to test in EDA (stage 4)

| # | Hypothesis | Feature(s) that would capture it |
|---|---|---|
| H1 | Fraudulent transactions are larger than the card's usual spend | `amount`, `amt_to_card_mean`, `amt_zscore_card` |
| H2 | Fraud happens disproportionately at night | `hour`, `is_night` |
| H3 | Fraud comes in bursts on a compromised card | `txn_count_1h/24h/7d`, `amt_sum_1h/24h/7d`, `secs_since_last_txn` |
| H4 | Some merchant categories (online shopping, groceries at POS) are riskier | `category`, `category_share_card` |
| H5 | Customer demographics carry little signal and must not drive decisions | `age`, `gender` (checked for fairness) |

## 1.9 Constraints, risks and ethics

* **No leakage:** every feature uses only the card's *previous* transactions; splits are chronological
  (train → validation → test), never random.
* **Privacy:** names and street addresses are dropped, card numbers are replaced by a salted hash in the
  silver layer; quasi-identifiers are masked for analysts by Trino rules and tagged as PII in OpenMetadata.
* **Fairness:** performance is reported by gender and age band (stage 6).
* **Simulated data:** results are optimistic compared with real card data; the framework, not the exact
  numbers, is what transfers.

## 1.10 How the lifecycle maps onto the project

| Stage | Notebook | Code (`src/fraudlens`) | Open-source tools |
|---|---|---|---|
| 1 Problem understanding | 01 | `config/pipeline.yaml` | Jupyter, Markdown |
| 2 Data acquisition | 02 | `acquire.py`, `platform/nifi` | Kaggle API, requests, BeautifulSoup, Apache NiFi, MinIO |
| 3 Data preparation | 03 | `validate.py`, `clean.py`, `features.py`, `split.py` | pandas, Dask, PyArrow |
| 4 EDA | 04 | `eda.py`, `interactive.py` | pandas, Seaborn, Matplotlib, Plotly |
| 5 Modelling | 05 | `preprocess.py`, `models.py`, `tuning.py`, `train.py`, `selection.py` | scikit-learn, XGBoost, LightGBM, CatBoost, MLflow |
| 6 Evaluation & interpretation | 06 | `evaluate.py`, `explain.py` | sklearn metrics, SHAP, ELI5, Yellowbrick |
| 7 Deployment & visualisation | 07 | `api.py`, `dashboard/app.py`, `publish.py`, `crawler.py`, `olap.py` | FastAPI, Streamlit, Plotly, Trino, ClickHouse, Superset, Docker |
| 8 Monitoring & improvement | 08 | `monitor.py`, `tracking.py` | Evidently AI, Prometheus, Grafana, MLflow |
"""),
("code", r"""
summary = read_json(P["reports"] / "verification_report.json")
print(f"End-to-end verification of the current run: {summary['status']} "
      f"({summary['passed']} checks passed, {summary['failed']} failed)")
pd.DataFrame(summary["success_criteria"])
"""),
("md", r"""
The success criteria are met on the test period; the rest of the notebooks show how each stage gets there.
**Next:** [02 · Data acquisition](02_data_acquisition.ipynb)
"""),
]
