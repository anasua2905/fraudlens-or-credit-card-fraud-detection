# Deployment guide

FraudLens is deployed as two services built from the same model artefact:

* the **scoring API** (FastAPI): real-time `/score`, `/feedback`, `/metrics`;
* the **dashboard** (Streamlit): alert queue, live scoring, threshold and cost, monitoring.

Both load the **deploy bundle** `dashboard/bundle/` (≈ 40 MB, committed to Git): the `FraudModel`
(preprocessor + HistGradientBoosting champion + threshold 0.7397), the online feature-store snapshot (999 cards),
the scored validation and test transactions, the reports and the figures. `bundle/manifest.json` lists every file
with its SHA-256 and the pinned library versions; `dashboard/requirements.txt` pins the same versions.

Four ways to run it, from simplest to most production-like:

## 1. Local, from the virtual environment (no Docker)

```bash
uvicorn fraudlens.api:app --port 8000          # http://localhost:8000/docs
streamlit run dashboard/app.py                  # http://localhost:8501
```

On Windows the helper script starts both in the background, waits until they are healthy, and can publish
temporary public links through a Cloudflare quick tunnel (no account needed; the links change on every start and
stop working when the PC sleeps or the script is stopped):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\deploy\run_local.ps1 -Public
powershell -ExecutionPolicy Bypass -File scripts\deploy\stop_local.ps1
```

The URLs are printed and saved in `data\deploy\deployment.json`; logs are in `data\deploy\*.log`. Only the two
services are exposed (bound to 127.0.0.1 and reached through the tunnel), nothing else on the machine.

## 2. Container image (GHCR)

`platform/docker/Dockerfile.app` builds one image with the bundle, the API and the dashboard. The `deploy`
workflow builds it on every push to `main`/`final-a4`, starts both services in the runner, scores a
transaction through the API, checks the dashboard health endpoint, and only then pushes it to the GitHub
Container Registry:

```bash
docker run -d -p 8000:8000 ghcr.io/anasua2905/fraudlens-app:latest api
docker run -d -p 8501:8501 ghcr.io/anasua2905/fraudlens-app:latest dashboard
```

Tags: `latest`, the branch name, and `sha-<commit>` (use the SHA tag for reproducible deployments and rollback).
The same image runs on any container host (Azure Container Apps, Google Cloud Run, AWS App Runner, Render, Fly.io,
a VM with Docker): expose port 8000 or 8501, set `PORT` if the platform requires a specific port.

Build locally instead: `docker build -f platform/docker/Dockerfile.app -t fraudlens-app .`

## 3. Hosted dashboard: Streamlit Community Cloud (free)

1. Sign in at https://share.streamlit.io with the GitHub account that can see the repository (deploying from a
   repository requires admin access to it: the repository owner deploys, or deploy from your fork).
2. **Create app** → *Deploy a public app from GitHub* →
   * Repository: `anasua2905/fraudlens-or-credit-card-fraud-detection` (or your fork)
   * Branch: `final-a4` (later `main`)
   * Main file path: `dashboard/app.py`
   * App URL: e.g. `fraudlens-prt661`
   * **Advanced settings → Python version: 3.11**
3. **Deploy**. The first build installs `dashboard/requirements.txt` (a few minutes); the app then runs from the
   committed bundle. Every push to the branch redeploys it automatically.

No secrets are needed: the bundle holds only simulated, pseudonymised data.

## 4. The full platform (Docker Compose)

For the complete system (lake, catalog, query engines, BI, MLflow, Prometheus, Grafana) see the
[platform guide](platform.md):

```bash
cd platform && docker compose --profile lake --profile ingest --profile olap --profile bi --profile mlops up -d
```

The `api` service mounts `dashboard/bundle`; Prometheus scrapes it and Grafana shows the operations dashboard.

## Release checklist

1. Pipeline run complete; `reports/verification_report.json` status **PASS**.
2. `python -m fraudlens.run_all --only bundle` rebuilt the bundle; commit `dashboard/bundle` and `dashboard/requirements.txt`.
3. `pytest` green; CI (`ci.yml`) green, including the dashboard render check.
4. Push; `deploy.yml` publishes `ghcr.io/…/fraudlens-app:sha-<commit>`; Streamlit Cloud redeploys.
5. Check `/health` (model and threshold), open the dashboard Overview and compare its KPIs with
   `reports/model_results.json`.
6. Rollback: redeploy the previous `sha-` tag, or revert the bundle commit.
