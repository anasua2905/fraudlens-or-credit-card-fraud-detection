"""Lake crawler: discover tables in MinIO and register them in the Hive Metastore via Trino.

This plays the role of an AWS Glue crawler. For every bucket it

  1. lists the objects and groups them into tables (first path segment),
  2. reads Hive-style partition keys from the path (`split=test/`),
  3. infers the column schema from a Parquet footer (or a CSV header),
  4. issues CREATE SCHEMA / CREATE TABLE ... WITH (external_location=...)
     through Trino, which writes the definition into the Hive Metastore,
  5. syncs partitions (`system.sync_partition_metadata`), counts rows, and
  6. writes reports/data_catalog.json: tables, columns, types, partitions,
     row counts, descriptions and PII tags (the human-readable catalog).

Re-running is safe: a table whose schema changed is dropped and re-created
(the data are external, so dropping the definition never deletes files).

    python -m fraudlens.crawler [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
from dataclasses import dataclass, field

from . import lake
from .utils import get_logger, load_config, write_json

log = get_logger()

LAYER_SCHEMAS = {"raw": "raw", "silver": "silver", "gold": "gold"}

DESCRIPTIONS = {
    "raw.sparkov_train": "Bronze: fraudTrain.csv exactly as acquired from Kaggle (all columns VARCHAR).",
    "raw.sparkov_test": "Bronze: fraudTest.csv exactly as acquired from Kaggle (all columns VARCHAR).",
    "silver.transactions": "Validated, typed, de-duplicated transactions; card numbers pseudonymised, "
                           "names and street removed. Partitioned by month.",
    "gold.features": "Model-ready, leakage-safe features (card history, velocity). Partitioned by "
                     "chronological split: train / val / test.",
    "gold.scored_transactions": "Validation and test transactions scored by the champion model, with alert "
                                "flag, outcome and top SHAP reasons.",
    "gold.daily_kpis": "Daily alert, catch and cost figures per split (Superset's main dataset).",
    "gold.alert_explanations": "Top three SHAP contributions for every test-period alert.",
    "gold.model_comparison": "Validation and test metrics for every candidate model.",
    "gold.monitoring_monthly": "Monthly drift and performance monitoring (Evidently + PSI).",
    "gold.feature_drift": "Per-feature, per-month drift statistics.",
}

# Columns tagged for governance; Trino access-control rules mask these for analysts.
PII_TAGS = {
    "card_id": "pseudonymous identifier",
    "cc_num": "direct identifier (raw only)",
    "first": "direct identifier (raw only)",
    "last": "direct identifier (raw only)",
    "street": "direct identifier (raw only)",
    "dob": "quasi-identifier",
    "zip": "quasi-identifier",
    "lat": "quasi-identifier",
    "lon": "quasi-identifier",
    "long": "quasi-identifier (raw only)",
    "job": "quasi-identifier",
}


@dataclass
class DiscoveredTable:
    layer: str
    name: str
    bucket: str
    fmt: str                              # PARQUET | CSV
    keys: list[str] = field(default_factory=list)
    partition_keys: list[str] = field(default_factory=list)
    columns: list[tuple[str, str]] = field(default_factory=list)   # (name, trino type)

    @property
    def schema(self) -> str:
        return LAYER_SCHEMAS[self.layer]

    @property
    def location(self) -> str:
        return f"s3a://{self.bucket}/{self.prefix}"

    @property
    def prefix(self) -> str:
        return self.keys[0].split("/")[0] + "/" + (self.keys[0].split("/")[1] + "/" if self.layer == "raw" else "")


def arrow_to_trino(t) -> str:
    import pyarrow as pa
    if pa.types.is_boolean(t):
        return "BOOLEAN"
    if pa.types.is_int8(t):
        return "TINYINT"
    if pa.types.is_int16(t):
        return "SMALLINT"
    if pa.types.is_int32(t) or pa.types.is_uint8(t) or pa.types.is_uint16(t):
        return "INTEGER"
    if pa.types.is_integer(t):
        return "BIGINT"
    if pa.types.is_float32(t):
        return "REAL"
    if pa.types.is_floating(t):
        return "DOUBLE"
    if pa.types.is_timestamp(t):
        return "TIMESTAMP(3)"
    if pa.types.is_date(t):
        return "DATE"
    if pa.types.is_dictionary(t):
        return arrow_to_trino(t.value_type)
    return "VARCHAR"


def parse_partitions(key: str) -> list[tuple[str, str]]:
    return [tuple(seg.split("=", 1)) for seg in key.split("/")[:-1] if "=" in seg]


def discover(cfg: dict) -> list[DiscoveredTable]:
    """Group lake objects into tables and infer their schemas."""
    import pyarrow.parquet as pq
    found: list[DiscoveredTable] = []
    for layer, bucket in cfg["platform"]["buckets"].items():
        keys = [k for k in lake.list_keys(cfg, bucket) if not k.split("/")[-1].startswith(("_", "."))
                and "/_" not in k]
        groups: dict[str, list[str]] = {}
        for k in keys:
            segs = k.split("/")
            if layer == "raw":           # raw/<dataset>/<file-group>/<file>.csv
                if len(segs) < 3 or not k.endswith(".csv"):
                    continue
                name = f"{segs[0]}_{segs[1]}"
            else:
                if not k.endswith(".parquet"):
                    continue
                name = segs[0]
            groups.setdefault(name, []).append(k)
        for name, ks in sorted(groups.items()):
            t = DiscoveredTable(layer, name, bucket, "CSV" if layer == "raw" else "PARQUET", sorted(ks))
            if t.fmt == "PARQUET":
                t.partition_keys = [p for p, _ in parse_partitions(ks[0])]
                schema = pq.read_schema(io.BytesIO(lake.read_bytes(cfg, bucket, ks[0])))
                t.columns = [(f.name, arrow_to_trino(f.type)) for f in schema if f.name not in t.partition_keys]
                t.columns += [(p, "VARCHAR") for p in t.partition_keys]
            else:
                head = lake.read_bytes(cfg, bucket, ks[0], nbytes=8192).decode("utf-8", "replace")
                header = next(csv.reader([head.splitlines()[0]]))
                t.columns = [((h or "row_index").strip().lower(), "VARCHAR") for h in header]
            found.append(t)
            log.info("Discovered %s.%s: %s columns, %s files, partitions %s",
                     t.schema, t.name, len(t.columns), len(ks), t.partition_keys or "-")
    return found


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def ddl(t: DiscoveredTable, catalog: str) -> list[str]:
    cols = ",\n  ".join(f"{_q(c)} {typ}" for c, typ in t.columns)
    props = [f"external_location = '{t.location}'", f"format = '{t.fmt}'"]
    if t.partition_keys:
        props.append("partitioned_by = ARRAY[" + ", ".join(f"'{p}'" for p in t.partition_keys) + "]")
    if t.fmt == "CSV":
        props.append("skip_header_line_count = 1")
    comment = DESCRIPTIONS.get(f"{t.schema}.{t.name}", "").replace("'", "''")
    stmt = (f"CREATE TABLE IF NOT EXISTS {catalog}.{t.schema}.{t.name} (\n  {cols}\n)"
            + (f"\nCOMMENT '{comment}'" if comment else "")
            + "\nWITH (" + ", ".join(props) + ")")
    return [stmt]


VIEWS = {
    "gold.v_alert_queue": """
        SELECT s.transaction_id, s.ts, s.card_id, s.merchant, s.category, s.amount, s.state,
               s.score, s.outcome, s.reason_1, s.reason_2, s.reason_3, s.split
        FROM {c}.gold.scored_transactions s
        WHERE s.alert = 1""",
    "gold.v_category_performance": """
        SELECT split, category, count(*) AS transactions, sum(is_fraud) AS frauds, sum(alert) AS alerts,
               sum(CASE WHEN outcome = 'caught' THEN 1 ELSE 0 END) AS caught,
               sum(CASE WHEN outcome = 'false_alert' THEN 1 ELSE 0 END) AS false_alerts,
               sum(CASE WHEN outcome = 'missed' THEN 1 ELSE 0 END) AS missed,
               round(100.0 * sum(is_fraud) / count(*), 3) AS fraud_rate_pct
        FROM {c}.gold.scored_transactions
        GROUP BY split, category""",
    "gold.v_state_performance": """
        SELECT split, state, count(*) AS transactions, sum(is_fraud) AS frauds,
               sum(CASE WHEN outcome = 'caught' THEN 1 ELSE 0 END) AS caught,
               round(100.0 * sum(is_fraud) / count(*), 3) AS fraud_rate_pct
        FROM {c}.gold.scored_transactions
        GROUP BY split, state""",
}


def connect(cfg: dict):
    import trino
    p = cfg["platform"]
    return trino.dbapi.connect(host=p["trino_host"], port=int(p["trino_port"]), user=p["trino_user"],
                               catalog=p.get("trino_catalog", "hive"), http_scheme="http")


def _run(cur, sql: str):
    cur.execute(sql)
    try:
        return cur.fetchall()
    except Exception:
        return None


def register(cfg: dict, found: list[DiscoveredTable]) -> list[dict]:
    catalog = cfg["platform"].get("trino_catalog", "hive")
    conn = connect(cfg)
    cur = conn.cursor()
    entries = []
    for layer, bucket in cfg["platform"]["buckets"].items():
        _run(cur, f"CREATE SCHEMA IF NOT EXISTS {catalog}.{LAYER_SCHEMAS[layer]} "
                  f"WITH (location = 's3a://{bucket}/')")
    for t in found:
        fq = f"{catalog}.{t.schema}.{t.name}"
        existing = _run(cur, f"SELECT column_name, data_type FROM {catalog}.information_schema.columns "
                             f"WHERE table_schema = '{t.schema}' AND table_name = '{t.name}' ORDER BY ordinal_position")
        if existing and [c for c, _ in existing] != [c for c, _ in t.columns]:
            log.info("Schema of %s changed; re-registering", fq)
            _run(cur, f"DROP TABLE {fq}")      # external table: files are untouched
        for stmt in ddl(t, catalog):
            _run(cur, stmt)
        if t.partition_keys:
            _run(cur, f"CALL {catalog}.system.sync_partition_metadata('{t.schema}', '{t.name}', 'FULL')")
        rows = _run(cur, f"SELECT count(*) FROM {fq}")[0][0]
        parts = len({tuple(parse_partitions(k)) for k in t.keys}) if t.partition_keys else 0
        entries.append({
            "table": f"{t.schema}.{t.name}", "layer": t.layer, "format": t.fmt,
            "location": t.location.replace("s3a://", "s3://"), "rows": int(rows), "files": len(t.keys),
            "partition_keys": t.partition_keys, "partitions": parts,
            "description": DESCRIPTIONS.get(f"{t.schema}.{t.name}", ""),
            "columns": [{"name": c, "type": typ, **({"pii": PII_TAGS[c]} if c in PII_TAGS else {})}
                        for c, typ in t.columns],
        })
        log.info("Registered %s (%s rows)", fq, f"{rows:,}")
    for name, body in VIEWS.items():
        if any(e["table"] == "gold.scored_transactions" for e in entries):
            _run(cur, f"CREATE OR REPLACE VIEW {catalog}.{name} AS {body.format(c=catalog)}")
            entries.append({"table": name, "layer": "gold", "format": "VIEW", "description": "Trino view for BI",
                            "sql": " ".join(body.format(c=catalog).split())})
    return entries


def run_crawler(cfg: dict, dry_run: bool = False) -> dict:
    report = {"crawled_at": dt.datetime.now().isoformat(timespec="seconds"), "status": "skipped",
              "metastore": "Hive Metastore (via Trino hive catalog)", "tables": []}
    if not lake.available(cfg):
        report["reason"] = "MinIO not reachable"
        write_json(report, cfg["paths"]["reports"] / "data_catalog.json")
        log.warning("Crawler skipped: MinIO not reachable")
        return report
    found = discover(cfg)
    if dry_run:
        report["status"] = "dry_run"
        report["ddl"] = [s for t in found for s in ddl(t, cfg["platform"].get("trino_catalog", "hive"))]
    else:
        try:
            report["tables"] = register(cfg, found)
            report["status"] = "registered"
        except Exception as exc:
            report["status"] = "failed"
            report["reason"] = f"{type(exc).__name__}: {exc}"
            log.error("Registration failed: %s", report["reason"])
    write_json(report, cfg["paths"]["reports"] / "data_catalog.json")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Discover lake tables and register them in the Hive Metastore")
    ap.add_argument("--dry-run", action="store_true", help="print the DDL without executing it")
    a = ap.parse_args()
    rep = run_crawler(load_config(), a.dry_run)
    if a.dry_run:
        print("\n;\n".join(rep.get("ddl", [])))


if __name__ == "__main__":
    main()
