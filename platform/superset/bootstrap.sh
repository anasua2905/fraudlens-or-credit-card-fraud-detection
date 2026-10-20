#!/bin/bash
# First start: create the metadata DB and the admin user, then build the FraudLens
# dashboard over Trino. Later starts skip straight to the web server.
set -e
superset db upgrade
superset fab create-admin --username admin --firstname FraudLens --lastname Admin \
  --email admin@fraudlens.local --password "${SUPERSET_ADMIN_PASSWORD}" || true
superset init
( sleep 20 && python /app/bootstrap_dashboard.py || echo "dashboard bootstrap failed; re-run: docker compose exec superset python /app/bootstrap_dashboard.py" ) &
exec gunicorn --bind 0.0.0.0:8088 --workers 2 --timeout 120 --limit-request-line 0 \
  --limit-request-field_size 0 "superset.app:create_app()"
