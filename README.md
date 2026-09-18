# Anomaly Detection Framework for Fraudulent Credit Card Transactions

FraudLens: data pipeline, cleaning, preprocessing and EDA for the
[Sparkov Credit Card Transactions Fraud Detection dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
(~1.85M simulated transactions, 2019-01-01 to 2020-12-31, CC0 license).

## Pipeline

| Layer | Folder | Contents |
|---|---|---|
| Bronze | `data/raw/` | Untouched `fraudTrain.csv`, `fraudTest.csv` + `manifest.json` (SHA-256, size, row count) |
| Quality | `reports/validation_report.json` | Data-contract results per file (PASS / WARN / FAIL) |
| Silver | `data/silver/transactions` | Typed, de-duplicated, PII-reduced transactions |
| Gold | `data/gold/features_{train,val,test}` | Model-ready feature tables |
| Gold | `data/gold/dashboard_transactions` | Descriptive table for the monitoring dashboard |
| Run log | `reports/pipeline_run.json` | Row counts, split ranges, fraud rates, null rates, runtime |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Option A: download via the Kaggle API (needs ~/.kaggle/kaggle.json)
python -m fraud_pipeline.pipeline

# Option B: you already downloaded the CSVs
python -m fraud_pipeline.pipeline --source local --local-dir /path/to/folder

# Tests (run on a synthetic sample with the same schema)
pytest -q
```

Optional: `export CARD_HASH_SALT=<secret>` before running to salt card-number hashing.

## Step 3: data cleaning, preprocessing and EDA

Run after the pipeline. Either open the notebook:

```bash
jupyter notebook notebooks/01_cleaning_eda_preprocessing.ipynb
notebooks/02_modelling.ipynb
```

or run the three stages from the terminal (about 2 to 3 minutes on the full dataset):

```bash
python -m fraud_pipeline.quality      # -> reports/cleaning_audit.json
python -m fraud_pipeline.eda          # -> reports/eda_summary.json, reports/figures/*.png
python -m fraud_pipeline.preprocess   # -> reports/preprocessing_report.json, artifacts/, data/processed/
```

| Stage | What it does | Data used |
|---|---|---|
| Cleaning audit | Types, cardinality, consistency, implausible ages, IQR outliers, near-duplicates, missing values; a recorded decision per issue | All rows (label-based checks: train only) |
| EDA | Class balance, amount, category, time, demographics, behavioural signals, card-level bursts, single-feature AUC, correlation, drift (PSI) | Train only (drift uses val/test features, never labels) |
| Preprocessing | `linear` view (log1p, train-quantile clipping, median + missing indicator, sin/cos time, scaling, one-hot) and `tree` view (raw values, NaN kept, ordinal categories) | Fitted on train, applied to val/test |

Preprocessing decisions:

* **Running totals are excluded** (`card_prior_txn_count`, `card_prior_category_count`). They only
  grow over time, so later periods contain values never seen in training. `category_share_card`
  (share of the card's past transactions in this category) replaces the category count.
  The exclusion list is `preprocess.exclude_features` in `config/pipeline.yaml`.
* **Drift decisions use the validation split only.** `reports/preprocessing_report.json` lists
  train-to-validation PSI for every model input; the test split is never used to choose features.
* **Class imbalance is not handled here.** Resampling or class weights belong inside model training
  on the training split. The report gives `suggested_scale_pos_weight` for gradient boosting.
* **High-cardinality columns** (merchant, job, city) are left out until the modelling step, where
  any label-based encoding can be fitted without leakage.

## Model development

Run after preprocessing (about 3 to 6 minutes on the full dataset):

```bash
python -m fraud_pipeline.train     # -> reports/model_results.json, figures 10-13, artifacts/model_*.joblib
```

or open `notebooks/02_modelling.ipynb`, which adds error analysis.

| Model | Type | Role |
|---|---|---|
| `logistic` | Logistic regression, class-weighted | Interpretable baseline |
| `hist_gb` | Histogram gradient boosting | Main supervised model |
| `isolation_forest` | Unsupervised, fitted on legitimate transactions only | Anomaly detector |
| `hybrid_gb` | Gradient boosting + anomaly score as a feature | Combines both approaches |
| `mlp` | Neural network (optional; add to `model.models`) | Deep-learning comparison |

Protocol: models are fitted on train, compared on validation, and the test split is scored **once** with
thresholds already fixed. Imbalance is handled with class weights inside training, never by resampling
validation or test.

Thresholds are chosen on validation by one of three strategies (`model.threshold.primary`):

* `max_f1` — balance precision and recall
* `alert_budget` — match the review team's daily capacity (`alerts_per_day`)
* `min_cost` — maximise recovered fraud minus review cost (`investigation_cost`, `recovery_rate`)

Reported metrics: PR-AUC against its random baseline, precision, recall, alerts per day, false alarms per
catch, and net benefit. Accuracy is not reported, because predicting "legitimate" everywhere scores 99.4%.

## Design decisions

1. **Time-based split, not random.** `fraudTest.csv` continues `fraudTrain.csv` in time.
   The last 6 weeks of `fraudTrain.csv` become the validation set (configurable).
2. **No leakage.** No feature uses `is_fraud`. Per-card history features use only the
   card's *earlier* transactions. Label-based encodings are deferred to the modelling step.
3. **Features computed over the full stream.** Safe because no labels are involved, and it
   mirrors production, where card history carries over into the scoring period.
4. **Privacy hygiene.** `cc_num` becomes a salted 16-char `card_id`; `first`, `last`, `street`
   are dropped. The data is synthetic, but the pipeline behaves as if it were real.
5. **Known dataset quirks handled.** Leftover `Unnamed: 0` index column is dropped; the
   `fraud_` prefix present on *every* merchant name (a generator artifact, not a label) is
   stripped; `unix_time` is ignored in favour of the readable timestamp.
6. **Validation gates the pipeline.** Any FAIL check stops the run.

## Feature dictionary

Features excluded from the models after EDA: the two running totals, plus `distance_km` (no signal) and
`is_new_merchant_for_card` (drifts over time). See `preprocess.exclude_features`.

| Feature | Description |
|---|---|
| `amount`, `log_amount` | Transaction amount and log(1 + amount) |
| `hour`, `day_of_week`, `is_weekend`, `is_night` | Time-of-transaction signals (night = 22:00 to 03:59) |
| `age` | Cardholder age at transaction time |
| `log_city_pop` | log(1 + cardholder city population) |
| `distance_km` | Haversine distance between cardholder and merchant |
| `card_prior_txn_count` | Number of earlier transactions on the card |
| `secs_since_last_txn` | Seconds since the card's previous transaction |
| `amt_to_card_mean` | Amount divided by the card's historical mean amount |
| `amt_zscore_card` | Z-score of amount against the card's history (needs 2+ prior txns) |
| `is_new_merchant_for_card` | 1 if the card has never used this merchant before |
| `card_prior_category_count` | Earlier transactions by the card in this category |
| `txn_count_{1h,24h,7d}` | Number of earlier card transactions in the look-back window |
| `amt_sum_{1h,24h,7d}` | Total amount of earlier card transactions in the window |
| `category`, `gender` | Categorical features (encoded in the modelling step) |

History features are empty (NaN) for a card's first transaction(s). Tree models handle this
natively; linear and neural models need imputation in the modelling step.

## Project layout

```
config/pipeline.yaml            all tunable settings
src/fraud_pipeline/
  ingest.py                     bronze: Kaggle download / local copy + manifest
  validate.py                   data contract checks
  clean.py                      silver: typing, dedupe, PII removal
  features.py                   gold: leakage-safe feature engineering
  split.py                      temporal train / val / test split
  pipeline.py                   orchestrator CLI
  datasets.py                   loaders for silver / gold tables
  quality.py                    Step 3: cleaning audit + decisions
  eda.py                        Step 3: EDA tables and charts
  drift.py                      Population Stability Index
  preprocess.py                 Step 3: linear and tree preprocessors
  models.py                     model definitions
  evaluate.py                   metrics, threshold selection, cost model
  train.py                      training, evaluation, error analysis
notebooks/01_cleaning_eda_preprocessing.ipynb
notebooks/02_modelling.ipynb
scripts/make_sample_data.py     schema-identical synthetic sample for tests
scripts/build_notebook.py       regenerates the notebook
tests/test_pipeline.py          Step 1: end-to-end + leakage tests
tests/test_step3.py             Step 3: audit, EDA and preprocessing tests
tests/test_models.py            Step 4: training, metrics and threshold tests
artifacts/                      fitted preprocessors (.joblib)
```

## Roadmap

- [x] Step 1: Data pipeline and data engineering
- [x] Step 3: Data cleaning, preprocessing and EDA
- [x] Modelling (baselines, gradient boosting, anomaly detectors, threshold tuning)
- [ ] Monitoring dashboard
- [ ] Reporting and decision support
