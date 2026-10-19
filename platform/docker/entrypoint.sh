#!/bin/sh
# Start one FraudLens component: `api` (FastAPI on :8000) or `dashboard` (Streamlit on :8501).
set -e
PORT_API="${PORT:-8000}"
case "$1" in
  api)       exec uvicorn fraudlens.api:app --host 0.0.0.0 --port "$PORT_API" ;;
  dashboard) exec streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port "${PORT:-8501}" \
                  --server.headless true --browser.gatherUsageStats false ;;
  *)         exec "$@" ;;
esac
