# GitHub configuration (to install into `.github/`)

This folder mirrors `.github/`. It lives here until a maintainer installs it (writing CI configuration
is left to a person on purpose):

| Path | What it is |
|---|---|
| `workflows/ci.yml` | tests the whole lifecycle on synthetic sample data; checks the dashboard renders from the committed bundle |
| `workflows/platform.yml` | starts the Docker platform on a runner and runs the pipeline through it (NiFi → MinIO → Hive Metastore → Trino with governance → ClickHouse, MLflow, API, Prometheus, Grafana, Superset); a second job catalogues the lake in OpenMetadata |
| `workflows/deploy.yml` | builds the `fraudlens-app` image (model + API + dashboard), smoke-tests it and publishes it to GHCR |
| `ISSUE_TEMPLATE/` | *Lifecycle task* and *Bug report* forms used by the GitHub Project |
| `pull_request_template.md` | PR checklist (lifecycle stage, tests, verification, no secrets) |

Install (Windows, from the repository root):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\github\install_workflows.ps1
```

or by hand: copy the contents of this folder into `.github/`, commit and push.
Pushing workflow files needs a GitHub login with the `workflow` scope
(`gh auth login --scopes workflow`, or a personal access token with *repo* + *workflow*).
