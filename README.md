# Anomaly Detection Framework for Fraudulent Credit Card Transactions

Step 1 of the project: **data pipeline and data engineering** for the
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
scripts/make_sample_data.py     schema-identical synthetic sample for tests
tests/test_pipeline.py          end-to-end + leakage tests
```

## Roadmap

- [x] Step 1: Data pipeline and data engineering
- [ ] Step 2: EDA and modelling (baselines, gradient boosting, anomaly detectors, threshold tuning)
- [ ] Step 3: Monitoring dashboard
- [ ] Step 4: Reporting and decision support
