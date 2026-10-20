"""Governance in OpenMetadata: catalog the lake from Trino, classify PII, record lineage.

  1. waits for the OpenMetadata server and signs in;
  2. ingests Trino's metadata (catalog `hive`: raw, silver, gold schemas) with the
     OpenMetadata `metadata` CLI, run in the official ingestion image (or a local CLI);
  3. tags every PII column found by the FraudLens crawler (reports/data_catalog.json):
     direct and quasi-identifiers -> PII.Sensitive, pseudonymous ids -> PII.NonSensitive;
  4. fills empty table descriptions from the crawler catalog;
  5. records table-level lineage raw -> silver -> gold, as built by the pipeline;
  6. reads everything back and writes reports/governance_catalog.json.

Trino access rules and MinIO policies enforce access; OpenMetadata documents it
(what exists, what is sensitive, where it came from) for people.

    python platform/openmetadata/govern.py                 # ingestion in Docker (default)
    python platform/openmetadata/govern.py --ingest cli    # `metadata` CLI installed locally
    python platform/openmetadata/govern.py --ingest skip   # tags and lineage only
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "ci"))
import ci_annotate  # noqa: E402

ci_annotate.install("governance")

import requests

ROOT = Path(__file__).resolve().parents[2]
SERVICE = "fraudlens_trino"
CATALOG = "hive"
OM_VERSION = "1.13.4"

# Table-level lineage of the pipeline: (upstream, downstream, what the step does)
LINEAGE = [
    ("raw.sparkov_train", "silver.transactions", "fraudlens.prepare: validate, type, de-duplicate, pseudonymise"),
    ("raw.sparkov_test", "silver.transactions", "fraudlens.prepare: validate, type, de-duplicate, pseudonymise"),
    ("silver.transactions", "gold.features", "fraudlens.features: leakage-safe card-history features, chronological split"),
    ("gold.features", "gold.scored_transactions", "fraudlens.batch_score: champion model score, alert and outcome"),
    ("gold.features", "gold.alert_explanations", "fraudlens.explain: top-3 SHAP reasons per alert"),
    ("gold.alert_explanations", "gold.scored_transactions", "joined onto the scored transactions"),
    ("gold.scored_transactions", "gold.daily_kpis", "fraudlens.publish: daily alert, catch and cost KPIs"),
    ("gold.scored_transactions", "gold.monitoring_monthly", "fraudlens.monitor: monthly performance and drift"),
    ("gold.features", "gold.feature_drift", "fraudlens.monitor: per-feature PSI and Evidently drift"),
]


def pii_tag(label: str) -> str:
    return "PII.NonSensitive" if label.startswith("pseudonymous") else "PII.Sensitive"


class OpenMetadata:
    def __init__(self, url: str):
        self.url = url.rstrip("/") + "/api/v1"
        self.s = requests.Session()

    def wait(self, timeout: int = 900) -> str:
        t0 = time.time()
        while True:
            try:
                r = self.s.get(f"{self.url}/system/version", timeout=10)
                if r.ok:
                    return r.json().get("version", "?")
            except requests.RequestException:
                pass
            if time.time() - t0 > timeout:
                raise SystemExit(f"OpenMetadata not ready at {self.url}")
            print("waiting for OpenMetadata ...", flush=True)
            time.sleep(10)

    def login(self, email: str, password: str) -> None:
        r = self.s.post(f"{self.url}/users/login",
                        json={"email": email, "password": base64.b64encode(password.encode()).decode()}, timeout=30)
        r.raise_for_status()
        self.s.headers["Authorization"] = f"Bearer {r.json()['accessToken']}"

    def get(self, path: str, **params):
        r = self.s.get(f"{self.url}{path}", params=params, timeout=60)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def patch(self, path: str, ops: list[dict]):
        r = self.s.patch(f"{self.url}{path}", data=json.dumps(ops),
                         headers={"Content-Type": "application/json-patch+json"}, timeout=60)
        r.raise_for_status()
        return r.json()

    def put(self, path: str, body: dict):
        r = self.s.put(f"{self.url}{path}", json=body, timeout=60)
        r.raise_for_status()
        return r.json() if r.text else None

    def bot_token(self, bot: str = "ingestion-bot") -> str:
        user = self.get(f"/bots/name/{bot}")["botUser"]
        return self.get(f"/users/auth-mechanism/{user['id']}")["config"]["JWTToken"]

    def table(self, name: str) -> dict | None:
        return self.get(f"/tables/name/{SERVICE}.{CATALOG}.{name}", fields="columns,tags")


def ingestion_config(token: str, om_host: str, trino_host: str, trino_user: str) -> str:
    return f"""source:
  type: trino
  serviceName: {SERVICE}
  serviceConnection:
    config:
      type: Trino
      hostPort: {trino_host}
      username: {trino_user}
      catalog: {CATALOG}
  sourceConfig:
    config:
      type: DatabaseMetadata
      includeViews: true
      schemaFilterPattern:
        includes: ["raw", "silver", "gold"]
sink:
  type: metadata-rest
  config: {{}}
workflowConfig:
  loggerLevel: INFO
  openMetadataServerConfig:
    hostPort: {om_host}
    authProvider: openmetadata
    securityConfig:
      jwtToken: "{token}"
"""


def run_ingestion(mode: str, om: OpenMetadata, args) -> None:
    if mode == "skip":
        return
    token = om.bot_token()
    with tempfile.TemporaryDirectory() as tmp:
        if mode == "docker":
            cfg = ingestion_config(token, "http://openmetadata-server:8585/api", "trino:8080", args.trino_user)
            (Path(tmp) / "trino.yaml").write_text(cfg, encoding="utf-8")
            os.chmod(tmp, 0o755)
            cmd = ["docker", "run", "--rm", "--network", args.network, "-v", f"{tmp}:/cfg:ro",
                   "--entrypoint", "metadata", f"docker.getcollate.io/openmetadata/ingestion:{OM_VERSION}",
                   "ingest", "-c", "/cfg/trino.yaml"]
        else:
            cfg = ingestion_config(token, args.url.rstrip("/") + "/api", args.trino, args.trino_user)
            (Path(tmp) / "trino.yaml").write_text(cfg, encoding="utf-8")
            cmd = ["metadata", "ingest", "-c", str(Path(tmp) / "trino.yaml")]
        print("running:", " ".join(c if "jwt" not in c.lower() else "***" for c in cmd), flush=True)
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout[-4000:], res.stderr[-4000:], sep="\n")
        if res.returncode != 0:
            raise SystemExit(f"metadata ingestion failed (exit {res.returncode})")


def enrich(om: OpenMetadata, catalog: dict) -> dict:
    tagged, described, missing = [], [], []
    for entry in catalog.get("tables", []):
        if entry.get("format") == "VIEW":
            continue
        t = om.table(entry["table"])
        if t is None:
            missing.append(entry["table"])
            continue
        ops = []
        if entry.get("description") and not t.get("description"):
            ops.append({"op": "add", "path": "/description", "value": entry["description"]})
            described.append(entry["table"])
        pii = {c["name"]: c["pii"] for c in entry.get("columns", []) if c.get("pii")}
        for i, col in enumerate(t.get("columns", [])):
            label = pii.get(col["name"])
            if not label:
                continue
            fqn = pii_tag(label)
            if any(tag.get("tagFQN") == fqn for tag in col.get("tags") or []):
                continue
            tag = {"tagFQN": fqn, "source": "Classification", "labelType": "Manual", "state": "Confirmed"}
            if col.get("tags") is None:
                ops.append({"op": "add", "path": f"/columns/{i}/tags", "value": [tag]})
            else:
                ops.append({"op": "add", "path": f"/columns/{i}/tags/-", "value": tag})
            if not col.get("description"):
                ops.append({"op": "add", "path": f"/columns/{i}/description", "value": f"PII: {label}"})
            tagged.append(f"{entry['table']}.{col['name']} -> {fqn}")
        if ops:
            om.patch(f"/tables/{t['id']}", ops)
    return {"tagged_columns": tagged, "described_tables": described, "not_in_openmetadata": missing}


def add_lineage(om: OpenMetadata) -> list[str]:
    done = []
    for up, down, what in LINEAGE:
        a, b = om.table(up), om.table(down)
        if a is None or b is None:
            continue
        om.put("/lineage", {"edge": {"fromEntity": {"id": a["id"], "type": "table"},
                                     "toEntity": {"id": b["id"], "type": "table"},
                                     "lineageDetails": {"description": what}}})
        done.append(f"{up} -> {down}")
    return done


def read_back(om: OpenMetadata) -> dict:
    tables = om.get("/tables", database=f"{SERVICE}.{CATALOG}", limit=100, fields="columns,tags")["data"]
    out = {"tables": [], "pii_columns": 0}
    for t in tables:
        pii = [c["name"] for c in t.get("columns", []) if any(g["tagFQN"].startswith("PII.") for g in c.get("tags") or [])]
        out["pii_columns"] += len(pii)
        out["tables"].append({"fqn": t["fullyQualifiedName"], "columns": len(t.get("columns", [])), "pii": pii,
                              "description": (t.get("description") or "")[:120]})
    lin = om.get(f"/lineage/table/name/{SERVICE}.{CATALOG}.gold.scored_transactions", upstreamDepth=3,
                 downstreamDepth=1) or {}
    names = {n["id"]: n.get("fullyQualifiedName", n.get("name")) for n in lin.get("nodes", [])}
    out["scored_transactions_upstream"] = sorted(
        names.get(e["fromEntity"], e["fromEntity"]) for e in lin.get("upstreamEdges", []))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=os.environ.get("OPENMETADATA_URL", "http://localhost:8585"))
    ap.add_argument("--ingest", choices=["docker", "cli", "skip"], default="docker")
    ap.add_argument("--network", default="fraudlens_default")
    ap.add_argument("--trino", default="localhost:8080", help="Trino host:port for --ingest cli")
    ap.add_argument("--trino-user", default="pipeline")
    ap.add_argument("--admin", default=os.environ.get("OPENMETADATA_ADMIN", "admin@open-metadata.org"))
    ap.add_argument("--password", default=os.environ.get("OPENMETADATA_PASSWORD", "admin"))
    a = ap.parse_args()

    om = OpenMetadata(a.url)
    version = om.wait()
    om.login(a.admin, a.password)
    print(f"OpenMetadata {version}: signed in", flush=True)
    run_ingestion(a.ingest, om, a)
    catalog = json.loads((ROOT / "reports" / "data_catalog.json").read_text(encoding="utf-8"))
    report = {"openmetadata_version": version, "service": SERVICE, **enrich(om, catalog), "lineage": add_lineage(om)}
    report.update(read_back(om))
    (ROOT / "reports" / "governance_catalog.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in report.items()}, indent=1))
    ci_annotate.notice("governance result", {"version": version, "tables": [t["fqn"] for t in report["tables"]],
                                             "pii_columns": report["pii_columns"],
                                             "upstream_of_scored": report["scored_transactions_upstream"],
                                             "lineage_edges": len(report["lineage"])})
    ok = report["tables"] and report["pii_columns"] > 0 and report["scored_transactions_upstream"]
    return 0 if ok else 1


if __name__ == "__main__":
    ci_annotate.run(main, "governance")
