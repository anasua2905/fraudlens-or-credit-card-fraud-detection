# Data dictionary

PII classes: **D** = direct identifier (never leaves raw), **Q** = quasi-identifier (masked for analysts by
Trino, tagged *PII.Sensitive* in OpenMetadata), **P** = pseudonymous id (tagged *PII.NonSensitive*).

## Raw (bronze): `fraudTrain.csv`, `fraudTest.csv`

Lake: `s3://fraudlens-raw/sparkov/{train,test}/`; Trino: `hive.raw.sparkov_train`, `hive.raw.sparkov_test` (all VARCHAR).

| Column | Type | Description | PII |
|---|---|---|---|
| `Unnamed: 0` | int | leftover row index from the CSV export (dropped) | |
| `trans_date_trans_time` | text | transaction timestamp | |
| `cc_num` | int | card number | D |
| `merchant` | text | merchant name (every name carries a `fraud_` prefix, a generator artefact) | |
| `category` | text | merchant category (14 values) | |
| `amt` | float | amount in USD | |
| `first`, `last` | text | card holder name | D |
| `gender` | text | F / M | |
| `street` | text | street address | D |
| `city`, `state`, `zip` | text/int | card holder address | Q (`zip`) |
| `lat`, `long` | float | card holder location | Q |
| `city_pop` | int | population of the card holder's city | |
| `job` | text | card holder occupation | Q |
| `dob` | date | date of birth | Q |
| `trans_num` | text | transaction id (unique) | |
| `unix_time` | int | timestamp as Unix seconds (offset in public copies; not used) | |
| `merch_lat`, `merch_long` | float | merchant location (random in this dataset) | |
| `is_fraud` | 0/1 | label | |

## Silver: `transactions`

Lake: `s3://fraudlens-silver/transactions/txn_month=YYYY-MM/`; Trino: `hive.silver.transactions`; local:
`data/silver/transactions.parquet`. One row per transaction; typed, de-duplicated, privacy-reduced.

| Column | Type | Notes | PII |
|---|---|---|---|
| `transaction_id` | string | from `trans_num` | |
| `ts` | timestamp | from `trans_date_trans_time` | |
| `card_id` | string | salted SHA-256 of `cc_num`, 16 hex chars | P |
| `merchant` | category | `fraud_` prefix removed | |
| `category` | category | 14 merchant categories | |
| `amount` | float | from `amt` | |
| `gender` | category | | |
| `dob` | date | | Q |
| `job`, `city`, `state`, `zip`, `city_pop` | | | Q (`job`, `zip`) |
| `lat`, `lon`, `merch_lat`, `merch_lon` | float | | Q (`lat`, `lon`) |
| `is_fraud` | int8 | label | |
| `source_split` | category | `train` / `test` (which Kaggle file) | |

## Gold: `features` (model input)

Lake: `s3://fraudlens-gold/features/split={train,val,test}/`; local: `data/gold/features_{split}.parquet`.
All behavioural features use only the card's **earlier** transactions.

| Feature | Definition | Used by champion |
|---|---|---|
| `amount`, `log_amount` | amount, log(1 + amount) | yes |
| `hour`, `day_of_week`, `is_weekend` | time of day / week | yes |
| `is_night` | hour in 22:00-03:59 | yes |
| `category` | merchant category (native categorical) | yes |
| `gender`, `age` | age in years at transaction time | yes |
| `log_city_pop` | log of city population | yes |
| `secs_since_last_txn` | seconds since the card's previous transaction | yes |
| `amt_to_card_mean` | amount ÷ mean of the card's earlier amounts | yes |
| `amt_zscore_card` | (amount − card mean) ÷ card std (≥ 2 earlier transactions) | yes |
| `category_share_card` | share of the card's earlier transactions in this category (derived in preprocessing) | yes |
| `txn_count_{1h,24h,7d}` | number of the card's transactions in the previous 1 h / 24 h / 7 days | yes |
| `amt_sum_{1h,24h,7d}` | sum of those amounts | yes |
| `card_prior_txn_count`, `card_prior_category_count` | running counts | **excluded** (grow over time, drift) |
| `is_new_merchant_for_card` | first time this card uses this merchant | **excluded** (drifts, weak) |
| `distance_km` | card holder ↔ merchant distance | **excluded** (no signal: merchant location is random) |
| `is_fraud` | label | target |

## Gold: analytics tables

| Table | Grain | Key columns |
|---|---|---|
| `scored_transactions` (partition `split` = val/test) | transaction | `score`, `alert`, `outcome` (caught / missed / false_alert / cleared), evidence features, `reason_1..3` (top SHAP contributions) |
| `daily_kpis` | day × split | transactions, frauds, alerts, caught, false_alerts, missed, amounts, review_cost, net_benefit |
| `alert_explanations` | test alert | `reason_1..3` |
| `model_comparison` | model | CV / validation / test metrics, champion flag |
| `monitoring_monthly` | month | drift share (Evidently, PSI), PR-AUC, precision, recall, alerts/day, retrain flag and reasons |
| `feature_drift` | month × feature | PSI, Evidently score and verdict |

Trino views for BI: `gold.v_alert_queue`, `gold.v_category_performance`, `gold.v_state_performance`.
ClickHouse database `fraudlens` holds copies of the gold analytics tables (MergeTree).

## Reports (`reports/`)

| File | Content |
|---|---|
| `acquisition_manifest.json`, `acquisition_metadata.json` | source, licence, checksums, row counts, column profile, NiFi ingest records |
| `validation_report.json`, `cleaning_audit.json`, `pipeline_run.json` | data contract, cleaning decisions, split |
| `eda_summary.json`, `figures/01-09` | EDA |
| `preprocessing_report.json` | model views, excluded features, drift between periods |
| `model_results.json`, `figures/10-15` | candidates, tuning, selection, metrics, CIs, calibration |
| `explanations.json`, `figures/16-18` | SHAP |
| `monitoring/` | monthly drift and performance, Evidently HTML |
| `verification_report.json` | 56 end-to-end checks |
| `data_catalog.json`, `lake_publication.json`, `olap_benchmark.json`, `governance_catalog.json` | platform stages |
