# GitHub Actions workflows (to install)

These are the project's CI/CD workflows. They live here until a maintainer installs them
into `.github/workflows/` (writing CI configuration is left to a person on purpose):

| Workflow | What it does |
|---|---|
| `ci.yml` | Tests the whole lifecycle on synthetic sample data; checks the dashboard renders from the committed bundle |
| `platform.yml` | Starts the Docker platform on a runner and runs the pipeline through it: NiFi -> MinIO -> Hive Metastore -> Trino (governance) -> ClickHouse, MLflow, API, Prometheus, Grafana, Superset; a second job catalogues the lake in OpenMetadata |
| `deploy.yml` | Builds the `fraudlens-app` image (model + API + dashboard), smoke-tests it and publishes it to GHCR |

Install (Windows, from the repository root):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\github\install_workflows.ps1
```

or by hand: copy the three `.yml` files into `.github/workflows/`, commit and push.
Pushing workflow files needs a GitHub login with the `workflow` scope
(`gh auth login --scopes workflow`, or a personal access token with *repo* + *workflow*).
