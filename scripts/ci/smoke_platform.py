"""End-to-end checks of the running FraudLens platform (after run_sample_pipeline.py).

Every check prints PASS/FAIL, failures are also emitted as GitHub annotations,
and the results go to reports/platform_smoke.json and the job summary.

    python scripts/ci/smoke_platform.py [--skip superset nifi]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
import urllib3

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import ci_annotate  # noqa: E402

ci_annotate.install("smoke test")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from fraudlens import lake  # noqa: E402
from fraudlens.utils import load_config, read_json  # noqa: E402

RESULTS: list[dict] = []


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def check(group: str, name: str, retries: int = 1, wait: float = 5):
    def deco(fn):
        detail, ok = None, False
        for attempt in range(retries):
            try:
                detail = fn()
                ok = True
                break
            except Exception as exc:          # noqa: BLE001 - every failure is reported
                detail = f"{type(exc).__name__}: {exc}"[:600]
                if attempt < retries - 1:
                    time.sleep(wait)
        RESULTS.append({"group": group, "check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'} | {group:10s} | {name} | {detail}", flush=True)
        if not ok and env("GITHUB_ACTIONS"):
            print(f"::error title=smoke {group}: {name}::{str(detail)[:400]}", flush=True)
        return fn
    return deco


def run(skip: set[str]) -> int:
    cfg = load_config()
    reports = cfg["paths"]["reports"]
    ingest = cfg["platform"].get("ingest", "direct")

    # ------------------------------------------------------------ lake (MinIO)
    @check("lake", "three layer buckets exist")
    def _():
        names = {b["Name"] for b in lake.client(cfg).list_buckets()["Buckets"]}
        assert {"fraudlens-raw", "fraudlens-silver", "fraudlens-gold"} <= names, names
        return sorted(names)

    @check("lake", "raw CSVs, silver and gold tables are in the lake")
    def _():
        raw = lake.list_keys(cfg, "fraudlens-raw", "sparkov/")
        assert any(k.endswith("train/fraudTrain.csv") for k in raw) and any(k.endswith("test/fraudTest.csv") for k in raw), raw
        silver = lake.list_keys(cfg, "fraudlens-silver", "transactions/")
        gold = {k.split("/")[0] for k in lake.list_keys(cfg, "fraudlens-gold")}
        assert silver and {"scored_transactions", "daily_kpis", "features"} <= gold, gold
        return {"raw": len(raw), "silver_files": len(silver), "gold_tables": sorted(gold)}

    @check("lake", "analyst account cannot read the raw layer (least privilege)")
    def _():
        import boto3
        from botocore.config import Config
        s3 = boto3.client("s3", endpoint_url=cfg["platform"]["s3_endpoint"], region_name="us-east-1",
                          aws_access_key_id=env("ANALYST_ACCESS_KEY"), aws_secret_access_key=env("ANALYST_SECRET_KEY"),
                          config=Config(s3={"addressing_style": "path"}))
        s3.list_objects_v2(Bucket="fraudlens-gold", MaxKeys=1)
        try:
            s3.list_objects_v2(Bucket="fraudlens-raw", MaxKeys=1)
        except Exception as exc:
            return f"gold readable, raw denied ({type(exc).__name__})"
        raise AssertionError("analyst could list the raw bucket")

    # ------------------------------------------------------------ ingest (NiFi)
    if "nifi" not in skip:
        @check("ingest", "NiFi flow is running and wrote an ingest record per file", retries=3)
        def _():
            assert ingest == "nifi", f"pipeline ingest route was {ingest!r}"
            man = read_json(reports / "acquisition_manifest.json")
            recs = man["lake"]["nifi_ingest_records"]
            assert {"fraudTrain.csv", "fraudTest.csv"} <= set(recs), recs.keys()
            for name, r in recs.items():
                assert r["lake.key"].startswith("sparkov/") and r.get("s3.etag"), (name, r)
            s = requests.Session()
            s.verify = False
            tok = s.post("https://localhost:8443/nifi-api/access/token",
                         data={"username": env("NIFI_USERNAME"), "password": env("NIFI_PASSWORD")}, timeout=20)
            tok.raise_for_status()
            s.headers["Authorization"] = f"Bearer {tok.text}"
            root = s.get("https://localhost:8443/nifi-api/flow/process-groups/root", timeout=20).json()
            groups = root["processGroupFlow"]["flow"]["processGroups"]
            g = next(x for x in groups if x["component"]["name"] == "FraudLens ingest")
            assert g["runningCount"] >= 6 and g["invalidCount"] == 0, (g["runningCount"], g["invalidCount"])
            return {"files": {k: v["lake.key"] for k, v in recs.items()}, "running_processors": g["runningCount"]}

    # ------------------------------------------------------------ catalog + Trino
    @check("catalog", "crawler registered the lake tables in the Hive Metastore")
    def _():
        cat = read_json(reports / "data_catalog.json")
        assert cat["status"] == "registered", cat.get("reason")
        tables = {t["table"] for t in cat["tables"]}
        assert {"raw.sparkov_train", "silver.transactions", "gold.scored_transactions"} <= tables, tables
        return sorted(tables)

    import trino

    def trino_rows(user: str, sql: str):
        conn = trino.dbapi.connect(host="localhost", port=8080, user=user, catalog="hive")
        cur = conn.cursor()
        cur.execute(sql)
        return cur.fetchall()

    @check("trino", "gold row count matches what the pipeline published")
    def _():
        pub = read_json(reports / "lake_publication.json")["tables"]["gold.scored_transactions"]["rows"]
        n = trino_rows("pipeline", "SELECT count(*) FROM hive.gold.scored_transactions")[0][0]
        assert n == pub, (n, pub)
        return n

    @check("governance", "Trino masks PII and hides `job` from the analyst role")
    def _():
        dob = trino_rows("analyst", "SELECT dob FROM hive.silver.transactions LIMIT 50")
        assert all(str(r[0])[:10].endswith("-01-01") for r in dob if r[0] is not None), dob[:3]
        try:
            trino_rows("analyst", "SELECT job FROM hive.silver.transactions LIMIT 1")
        except Exception as exc:
            assert "Access Denied" in str(exc) or "denied" in str(exc).lower(), exc
            return "dob truncated to the year; job denied"
        raise AssertionError("analyst could read the job column")

    @check("governance", "BI role cannot read the raw layer")
    def _():
        try:
            trino_rows("superset", "SELECT count(*) FROM hive.raw.sparkov_train")
        except Exception as exc:
            return f"denied ({type(exc).__name__})"
        raise AssertionError("superset user could read raw")

    # ------------------------------------------------------------ ClickHouse
    @check("olap", "ClickHouse loaded gold and agrees with Trino")
    def _():
        rep = read_json(reports / "olap_benchmark.json")
        assert rep["status"] == "ok", rep.get("reason") or rep.get("status")
        assert all(q.get("answers_agree") for q in rep["queries"].values()), rep["queries"]
        return {k: {"clickhouse_ms": v["clickhouse_ms"]["median"], "trino_ms": v["trino_ms"]["median"]}
                for k, v in rep["queries"].items()}

    # ------------------------------------------------------------ MLOps
    @check("mlflow", "MLflow server has the run and the registered champion")
    def _():
        base = "http://localhost:5000"
        assert requests.get(f"{base}/health", timeout=10).ok
        r = requests.get(f"{base}/api/2.0/mlflow/registered-models/get",
                         params={"name": cfg["platform"]["registered_model"]}, timeout=10)
        r.raise_for_status()
        versions = r.json()["registered_model"].get("latest_versions", [])
        assert versions, r.json()
        return {"versions": [v["version"] for v in versions]}

    sample_txn: dict = {}

    @check("api", "scoring API is healthy and scores a transaction", retries=12, wait=10)
    def _():
        base = "http://localhost:8000"
        assert requests.get(f"{base}/health", timeout=10).json()["status"] == "ok"
        spec = requests.get(f"{base}/openapi.json", timeout=10).json()
        txn = spec["components"]["schemas"]["Transaction"]["examples"][0]
        sample_txn.update(txn)
        r = requests.post(f"{base}/score", json=txn, params={"update_state": "false"}, timeout=10)
        r.raise_for_status()
        out = r.json()
        assert 0 <= out["score"] <= 1 and out["alert"] == (out["score"] >= out["threshold"])
        fb = requests.post(f"{base}/feedback", json={"transaction_id": "smoke-1", "label": "fraud"}, timeout=10)
        assert fb.json()["status"] == "recorded"
        return {k: out[k] for k in ("score", "alert", "threshold", "model", "reasons")}

    @check("monitoring", "Prometheus scrapes the API", retries=12, wait=10)
    def _():
        t = requests.get("http://localhost:9090/api/v1/targets", timeout=10).json()["data"]["activeTargets"]
        api = [x for x in t if x["labels"]["job"] == "fraudlens-api"]
        assert api and api[0]["health"] == "up", [(x["labels"]["job"], x["health"]) for x in t]
        q = requests.get("http://localhost:9090/api/v1/query",
                         params={"query": "fraudlens_transactions_scored_total"}, timeout=10).json()
        assert q["data"]["result"], q
        return {"api_target": "up", "scored_total": q["data"]["result"][0]["value"][1]}

    @check("monitoring", "Grafana datasources (Prometheus, ClickHouse) are healthy", retries=12, wait=10)
    def _():
        auth = ("admin", env("GRAFANA_ADMIN_PASSWORD"))
        out = {}
        for uid in ("prometheus", "clickhouse"):
            r = requests.get(f"http://localhost:3000/api/datasources/uid/{uid}/health", auth=auth, timeout=20)
            assert r.ok and r.json().get("status") == "OK", (uid, r.status_code, r.text[:300])
            out[uid] = "OK"
        dash = requests.get("http://localhost:3000/api/search", params={"query": "FraudLens"}, auth=auth, timeout=10).json()
        out["dashboards"] = sorted(d["title"] for d in dash)
        assert len(out["dashboards"]) >= 2, out
        return out

    if "superset" not in skip:
        @check("bi", "Superset is up with the FraudLens dashboard", retries=18, wait=10)
        def _():
            s = requests.Session()
            assert s.get("http://localhost:8088/health", timeout=10).ok
            tok = s.post("http://localhost:8088/api/v1/security/login",
                         json={"username": "admin", "password": env("SUPERSET_ADMIN_PASSWORD"), "provider": "db"},
                         timeout=20).json()["access_token"]
            s.headers["Authorization"] = f"Bearer {tok}"
            d = s.get("http://localhost:8088/api/v1/dashboard/", timeout=20).json()
            titles = [x["dashboard_title"] for x in d.get("result", [])]
            assert any("FraudLens" in t for t in titles), titles
            return titles

    failed = [r for r in RESULTS if r["status"] == "FAIL"]
    out = {"checks": RESULTS, "passed": len(RESULTS) - len(failed), "failed": len(failed),
           "status": "PASS" if not failed else "FAIL"}
    (reports / "platform_smoke.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    if env("GITHUB_STEP_SUMMARY"):
        with open(env("GITHUB_STEP_SUMMARY"), "a", encoding="utf-8") as fh:
            fh.write(f"## Platform smoke test: {out['status']} ({out['passed']}/{len(RESULTS)})\n\n")
            fh.write("| Group | Check | Result |\n|---|---|---|\n")
            for r in RESULTS:
                fh.write(f"| {r['group']} | {r['check']} | {'✅' if r['status'] == 'PASS' else '❌'} |\n")
    print(f"\n{out['status']}: {out['passed']}/{len(RESULTS)} checks passed")
    return 0 if not failed else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[])
    skip = set(ap.parse_args().skip)
    ci_annotate.run(lambda: run(skip), "smoke test")
