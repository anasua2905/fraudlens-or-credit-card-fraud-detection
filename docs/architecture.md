# Architecture

FraudLens has two halves that share one code base (`src/fraudlens`) and one configuration
(`config/pipeline.yaml`):

* the **ML lifecycle** (stages 1-8), runnable on a laptop with `python -m fraudlens.run_all`;
* the **open-source data platform** (`platform/docker-compose.yml`), where the same stages publish to and
  read from a governed data lake, and where the model is served and monitored.

## 1. Lifecycle stages and modules

```
acquire ─► prepare ─► quality ─► eda ─► preprocess ─► train ─► explain ─► score ─► monitor ─► verify ─► bundle ─► publish ─► crawl ─► olap
 [2]        [3]        [3]       [4]      [5]          [5]       [6]       [7]      [8]       [all]      [7]      [platform] [platform] [platform]
```

| Stage | Module | Reads | Writes |
|---|---|---|---|
| 2 Acquisition | `acquire.py` (+ NiFi) | Kaggle API / local folder | `data/raw/*.csv`, manifest (checksums), column profile; raw layer in MinIO |
| 3 Preparation | `validate.py`, `clean.py`, `features.py`, `split.py`, `prepare.py`, `quality.py` | raw CSVs | `data/silver/transactions.parquet`, `data/gold/features_{train,val,test}.parquet`, online feature-store snapshot |
| 4 EDA | `eda.py`, `interactive.py` | gold (train only) | figures 01-09, `eda_summary.json`, interactive explorer |
| 5 Modelling | `preprocess.py`, `models.py`, `tuning.py`, `train.py`, `selection.py`, `tracking.py` | gold | model matrices, `model_*.joblib`, `fraudlens_model.joblib`, `model_results.json`, MLflow runs + registry |
| 6 Evaluation | `evaluate.py`, `explain.py` | champion + test | metrics, CIs, calibration, SHAP figures, per-alert reasons |
| 7 Deployment | `batch_score.py`, `bundle.py`, `api.py`, `runtime.py`, `online.py`, `dashboard/app.py` | champion, feature store | scored transactions, deploy bundle, API, dashboard |
| 8 Monitoring | `monitor.py`, `drift.py` | scored periods | monthly drift/performance, Evidently reports, retraining recommendation |
| Verification | `verify.py` | everything | `verification_report.json` (56 checks across all stages) |
| Platform | `lake.py`, `publish.py`, `crawler.py`, `olap.py` | gold | MinIO tables, Hive Metastore entries, ClickHouse tables, benchmark |

## 2. Data platform

```
                ┌──────────────┐
 Kaggle/local ─►│ Apache NiFi  │─► s3://fraudlens-raw/sparkov/...   (versioned bucket, write-only ingest account)
                └──────────────┘          │
                                          ▼
                         python -m fraudlens.run_all   (pandas / Dask; scikit-learn, XGBoost, LightGBM; MLflow)
                                          │ publish (Hive-partitioned Parquet)
                                          ▼
              s3://fraudlens-silver/transactions/txn_month=YYYY-MM/   s3://fraudlens-gold/<table>/split=.../
                                          │ crawl (schema inference, CREATE TABLE ... external_location)
                                          ▼
                   Hive Metastore (PostgreSQL) ◄── Trino (hive catalog, file-based access control)
                                          │                    │
                    ┌─────────────────────┤                    └─► Superset dashboards (BI)
                    ▼                     ▼
          ClickHouse (MergeTree,   OpenMetadata (catalog, PII tags, lineage)
          s3() via named collection)
                    └─► Grafana business dashboard
 FastAPI scoring service ─► /metrics ─► Prometheus (alert rules) ─► Grafana operations dashboard
 MLflow tracking server + model registry
```

| Service | Image | Profile | Port |
|---|---|---|---|
| MinIO | `cgr.dev/chainguard/minio` | lake | 9000 (S3), 9001 (console) |
| PostgreSQL (metastore DB) | `postgres:16-alpine` | lake | - |
| Hive Metastore | `apache/hive:4.0.1` + S3A and PostgreSQL jars | lake | 9083 |
| Trino | `trinodb/trino:483` | lake | 8080 |
| Apache NiFi | `apache/nifi:2.12.0` | ingest | 8443 (HTTPS) |
| ClickHouse | `clickhouse/clickhouse-server:26.3` | olap | 8123 |
| Apache Superset | `apache/superset:4.1.2` + Trino driver | bi | 8088 |
| MLflow | project image | mlops | 5000 |
| Scoring API (FastAPI) | project image | mlops | 8000 |
| Prometheus | `prom/prometheus:v3.4.0` | mlops | 9090 |
| Grafana | `grafana/grafana:11.6.0` + ClickHouse plugin | mlops | 3000 |
| OpenMetadata | `openmetadata/server:1.13.4` + Elasticsearch + PostgreSQL | `platform/openmetadata/docker-compose.yml` | 8585 |

### Governance

| Control | Where | What it does |
|---|---|---|
| Least-privilege storage accounts | `platform/minio/policies/*.json`, `init_minio.py` | pipeline: read/write lake; trino: read-only; analyst (also ClickHouse): gold only, raw and silver denied; ingest (NiFi): put under `raw/sparkov/` only |
| Immutable raw layer | MinIO bucket versioning | acquired files can never be silently replaced |
| Query-time masking | `platform/trino/etc/rules.json` | analysts see `dob` truncated to the year, `zip` to 100s, `lat/lon` rounded, `job` denied; BI role cannot query raw/silver; card ids masked in BI |
| Pseudonymisation | `clean.py` | card numbers replaced by a salted SHA-256 id; names and streets dropped in silver |
| Catalog and classification | crawler (`data_catalog.json`), OpenMetadata (`govern.py`) | table descriptions, PII tags (PII.Sensitive / NonSensitive), lineage raw → silver → gold |
| Secrets | `platform/.env` (not committed), ClickHouse named collection with `from_env` | no credentials in code, SQL or logs |

## 3. Serving path

`FraudModel` (`scoring.py`) bundles the fitted preprocessor, the champion and the validation-chosen threshold,
so training and serving cannot diverge; a test proves the saved object reproduces the training matrices'
scores exactly (`float32` representation, `preprocess.MATRIX_DTYPE`). The **online feature store**
(`online.py`, `CardStore`) keeps each card's running statistics and recent window and computes the same
features as the batch pipeline from a raw transaction; a verification check compares online and batch
features on 2,000 test transactions.

## 4. Key design decisions

| Decision | Alternative rejected | Reason |
|---|---|---|
| Chronological train / validation / test split | random split, k-fold | fraud models are deployed forward in time; random splits leak the future |
| Class weights | SMOTE / undersampling | keeps the real distribution, no synthetic records, cheaper on 1.2 M rows |
| PR-AUC + cost-based threshold | accuracy, ROC-AUC alone, 0.5 threshold | accuracy is 99.6% for "never alert"; the business cares about money and queue size |
| Simplest non-inferior champion | the top validation score | the four boosted models are within noise; simpler = fewer dependencies, smaller, faster |
| Prior-only features | features over all history | must match what is known at authorisation time |
| Excluding running totals | keeping them | they grow with time and drift by construction (measured PSI) |
| Permutation SHAP for the champion | TreeSHAP | TreeSHAP did not reproduce the model output with native categorical splits |
| Docker platform verified in CI | only local runs | the team's laptops do not all run Docker; the runner proves the compose stack works |
