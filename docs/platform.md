# Platform guide: running the open-source data platform

Everything runs in Docker (`platform/docker-compose.yml`). The platform is optional: the pipeline, the API and
the dashboard all run without it, and the platform stages skip themselves when it is not reachable.

**Requirements:** Docker Desktop (Windows/macOS, WSL 2 backend on Windows) or Docker Engine + Compose v2;
4 CPUs; 8 GB RAM for `lake + mlops`, about 12 GB for every profile; 20 GB of disk for images.

## 1. Configure

```bash
cp platform/.env.example platform/.env
```

Edit `platform/.env` and change every password. It holds:

| Variables | Used by |
|---|---|
| `MINIO_ROOT_*` | MinIO administrator |
| `PIPELINE_*`, `TRINO_S3_*`, `ANALYST_*`, `INGEST_*` | least-privilege MinIO service accounts (created by `minio-init`) |
| `NIFI_USERNAME`, `NIFI_PASSWORD` (≥ 12 characters) | NiFi single-user login |
| `CLICKHOUSE_PASSWORD` | ClickHouse user `fraudlens` |
| `POSTGRES_PASSWORD` | Hive Metastore database |
| `SUPERSET_*`, `GRAFANA_ADMIN_PASSWORD` | web logins (user `admin`) |
| `FRAUDLENS_*` | read by the pipeline on the host (`load_config` picks them up from `platform/.env`) |

`platform/.env` is in `.gitignore`; never commit it.

## 2. Start (in layers)

```bash
cd platform
docker compose --profile lake up -d --build          # MinIO + init, PostgreSQL, Hive Metastore, Trino
docker compose --profile ingest up -d                # Apache NiFi + flow builder (needs lake)
docker compose --profile olap up -d                  # ClickHouse
docker compose --profile bi up -d                    # Superset (dashboard built automatically)
docker compose --profile mlops up -d                 # MLflow, scoring API, Prometheus, Grafana
docker compose ps
```

| UI | URL | Login |
|---|---|---|
| MinIO console | http://localhost:9001 | `MINIO_ROOT_USER` |
| NiFi | https://localhost:8443/nifi (self-signed certificate) | `NIFI_USERNAME` |
| Trino | http://localhost:8080 (UI), `trino --server localhost:8080 --user analyst` | any user name; rules decide access |
| ClickHouse | http://localhost:8123/play | `fraudlens` / `CLICKHOUSE_PASSWORD` |
| Superset | http://localhost:8088 | `admin` / `SUPERSET_ADMIN_PASSWORD` |
| MLflow | http://localhost:5000 | - |
| Scoring API | http://localhost:8000/docs | - |
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | `admin` / `GRAFANA_ADMIN_PASSWORD` |
| OpenMetadata | http://localhost:8585 | `admin@open-metadata.org` / `admin` (change it) |

## 3. Run the pipeline through the platform

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000      # optional: log to the MLflow server
python -m fraudlens.run_all --ingest nifi             # NiFi ingest ... publish, crawl, olap
# or only the platform stages after a local run:
python -m fraudlens.run_all --only publish crawl olap
```

What each platform stage does:

| Stage | Result |
|---|---|
| `acquire --ingest nifi` | files dropped in `data/landing/`; NiFi writes them to `s3://fraudlens-raw/sparkov/...` with an ingest record each; the stage waits and checks sizes |
| `publish` | silver and gold as Hive-partitioned Parquet in MinIO (`reports/lake_publication.json`) |
| `crawl` | tables discovered, schemas inferred, registered in the Hive Metastore through Trino; BI views created (`reports/data_catalog.json`) |
| `olap` | gold loaded into ClickHouse; four business questions run on Trino and ClickHouse, answers compared, times recorded (`reports/olap_benchmark.json`, figure 20) |

## 4. Governance with OpenMetadata

```bash
docker compose -f platform/openmetadata/docker-compose.yml up -d      # after the lake profile is up
python platform/openmetadata/govern.py                                # ingest Trino metadata, tag PII, add lineage
```

`govern.py` runs the OpenMetadata `metadata` CLI in the official ingestion image, then tags PII columns from the
crawler catalog and records lineage raw → silver → gold. Result: `reports/governance_catalog.json` and the
catalog in the OpenMetadata UI.

## 5. Try the governance rules

```sql
-- as user "analyst" (trino --user analyst)
SELECT dob, zip, lat FROM hive.silver.transactions LIMIT 3;   -- dob truncated to the year, zip to 100s, lat rounded
SELECT job FROM hive.silver.transactions LIMIT 1;             -- Access Denied
-- as user "superset"
SELECT * FROM hive.raw.sparkov_train LIMIT 1;                 -- Access Denied
```

## 6. Automated end-to-end test

`.github/workflows/platform.yml` starts all profiles on a GitHub runner, runs the lifecycle on sample data with
NiFi ingest, and runs `scripts/ci/smoke_platform.py`, which checks every component. A second job brings up
OpenMetadata and runs `govern.py`. Each step reports failures as run annotations, and the results are published
as notices; `python scripts/ci/collect_platform_results.py` saves them to `reports/platform_ci.json`.

### Latest verified run

[Run 39430878611](https://github.com/anasua2905/fraudlens-or-credit-card-fraud-detection/actions/runs/39430878611) on commit `992cdc8`: **both jobs passed**.

| Component | Check | Result |
|---|---|---|
| lake | three layer buckets exist | pass |
| lake | raw CSVs, silver and gold tables are in the lake | pass |
| lake | analyst account cannot read the raw layer (least privilege) | pass |
| ingest | NiFi flow is running and wrote an ingest record per file | pass |
| catalog | crawler registered the lake tables in the Hive Metastore | pass |
| trino | gold row count matches what the pipeline published | pass |
| governance | Trino masks PII and hides `job` from the analyst role | pass |
| governance | BI role cannot read the raw layer | pass |
| olap | ClickHouse loaded gold and agrees with Trino | pass |
| mlflow | MLflow server has the run and the registered champion | pass |
| api | scoring API is healthy and scores a transaction | pass |
| monitoring | Prometheus scrapes the API | pass |
| monitoring | Grafana datasources (Prometheus, ClickHouse) are healthy | pass |
| bi | Superset is up with the FraudLens dashboard | pass |

OLAP benchmark (sample data, median of five runs on the runner):

| Question | Trino (ms) | ClickHouse (ms) | ClickHouse faster | Same answer |
|---|---|---|---|---|
| Daily caught, false and missed alerts | 293 | 14.8 | 20× | yes |
| Fraud and alerts by category and hour | 226 | 16.3 | 14× | yes |
| Merchants with the most missed fraud | 212 | 10.6 | 20× | yes |
| Alert precision by state | 232 | 9.9 | 23× | yes |

Governance: OpenMetadata 1.13.4 catalogued 13 tables and views from Trino, tagged
29 PII columns and recorded 9 lineage edges from the raw files to
`gold.scored_transactions`.

### Fixes the runner found

| Symptom | Cause | Fix |
|---|---|---|
| `minio-init` exited with an error | minio-py 7.2.20's `MinioAdmin` takes keyword arguments only | `MinioAdmin(endpoint=..., credentials=...)` |
| MLflow never became healthy | four server workers in a 1 GB container kept dying | `--workers 1`, 1.5 GB limit |
| partitioned tables missing from the catalog | Trino's file-based access control denied `system.sync_partition_metadata` | `procedures` rule for the pipeline user |
| crawler stopped at the first failing table | one exception ended registration | each table registered independently; failures reported and fail CI |

## 7. Stop and clean up

```bash
docker compose --profile lake --profile ingest --profile olap --profile bi --profile mlops down       # keep data
docker compose --profile lake --profile ingest --profile olap --profile bi --profile mlops down -v    # delete volumes
docker compose -f platform/openmetadata/docker-compose.yml down -v
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `minio-init` exits with an error | check the keys in `.env`; re-run `docker compose up minio-init` |
| Hive Metastore restarts | PostgreSQL not ready yet; it retries. `docker compose logs hive-metastore` |
| Trino query fails with *Access Denied* | expected for restricted users; use `--user pipeline` for administration |
| NiFi flow missing | `docker compose run --rm nifi-init` (or `python platform/nifi/build_flow.py --rebuild`) |
| acquisition times out waiting for NiFi | `python platform/nifi/build_flow.py --status`; check `data/landing/` is mounted and readable |
| ClickHouse stage skipped | `FRAUDLENS_CLICKHOUSE_PASSWORD` must match `CLICKHOUSE_PASSWORD` |
| API container restarting | it waits for `dashboard/bundle`; run the pipeline (or `python -m fraudlens.bundle`) |
| Out of memory | start fewer profiles; Trino (3 GB) and Superset (2 GB) are the largest |
