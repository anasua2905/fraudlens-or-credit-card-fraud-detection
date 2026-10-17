"""Publish the silver and gold layers to the MinIO lake (Hive-style partitioned Parquet).

The layout is what the crawler (`fraudlens.crawler`) discovers and registers
in the Hive Metastore, and what Trino and Superset query:

    s3://fraudlens-silver/transactions/txn_month=2019-01/part-0.parquet
    s3://fraudlens-gold/features/split=train/part-0.parquet
    s3://fraudlens-gold/scored_transactions/split=test/part-0.parquet
    s3://fraudlens-gold/daily_kpis/part-0.parquet
    ...

Partition columns live in the path, not in the files (Hive convention).

    python -m fraudlens.publish
"""
from __future__ import annotations

import datetime as dt
from collections.abc import Iterator

import pandas as pd

from . import lake
from .datasets import load_features, load_silver
from .utils import get_logger, load_config, read_json, read_table, table_exists, write_json

log = get_logger()


def lakeify(df: pd.DataFrame) -> pd.DataFrame:
    """Types every engine reads the same way: strings, microsecond timestamps, plain numerics."""
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object or pd.api.types.is_string_dtype(s):
            out[c] = s.astype(str).where(s.notna(), None)
        elif pd.api.types.is_datetime64_any_dtype(s):
            out[c] = s.dt.tz_localize(None).astype("datetime64[us]") if s.dt.tz is not None else s.astype("datetime64[us]")
        elif pd.api.types.is_bool_dtype(s):
            out[c] = s.astype(bool)
    return out


def daily_kpis(scored: pd.DataFrame, cost: float, recovery: float) -> pd.DataFrame:
    """One row per day and split: the Superset dashboard's main dataset."""
    d = scored.assign(day=scored["ts"].dt.floor("D"))
    d["caught_amt"] = d["amount"].where(d["outcome"] == "caught", 0.0)
    d["missed_amt"] = d["amount"].where(d["outcome"] == "missed", 0.0)
    g = d.groupby(["split", "day"], observed=True).agg(
        transactions=("transaction_id", "size"), frauds=("is_fraud", "sum"), alerts=("alert", "sum"),
        caught=("outcome", lambda s: int((s == "caught").sum())),
        false_alerts=("outcome", lambda s: int((s == "false_alert").sum())),
        missed=("outcome", lambda s: int((s == "missed").sum())),
        amount=("amount", "sum"), fraud_amount_caught=("caught_amt", "sum"),
        fraud_amount_missed=("missed_amt", "sum")).reset_index()
    g["review_cost"] = g["alerts"] * cost
    g["net_benefit"] = g["fraud_amount_caught"] * recovery - g["review_cost"]
    return g


def tables(cfg: dict) -> Iterator[tuple[str, str, dict[str, str], pd.DataFrame]]:
    """Yield (layer, table, partition, frame) for everything the lake should hold."""
    P = cfg["paths"]
    silver = load_silver(cfg)
    silver["txn_month"] = silver["ts"].dt.strftime("%Y-%m")
    for month, g in silver.groupby("txn_month", observed=True):
        yield "silver", "transactions", {"txn_month": month}, g.drop(columns="txn_month")
    del silver

    for split in ("train", "val", "test"):
        yield "gold", "features", {"split": split}, load_features(cfg, split)

    scored = read_table(P["gold"] / "scored_transactions")
    for split, g in scored.groupby("split", observed=True):
        yield "gold", "scored_transactions", {"split": str(split)}, g.drop(columns="split")
    tcfg = cfg["model"]["threshold"]
    yield "gold", "daily_kpis", {}, daily_kpis(scored, tcfg["investigation_cost"], tcfg["recovery_rate"])

    if table_exists(P["gold"] / "alert_explanations"):
        yield "gold", "alert_explanations", {}, read_table(P["gold"] / "alert_explanations")
    results = read_json(P["reports"] / "model_results.json")
    yield "gold", "model_comparison", {}, pd.DataFrame(results["comparison"]).assign(
        champion=lambda t: t["model"] == results["champion"])
    if table_exists(P["monitoring"] / "monthly"):
        yield "gold", "monitoring_monthly", {}, read_table(P["monitoring"] / "monthly")
    if (P["monitoring"] / "feature_drift.csv").exists():
        yield "gold", "feature_drift", {}, pd.read_csv(P["monitoring"] / "feature_drift.csv")


def object_key(table: str, partition: dict[str, str]) -> str:
    parts = [table] + [f"{k}={v}" for k, v in partition.items()]
    return "/".join(parts) + "/part-0.parquet"


def run_publish(cfg: dict) -> dict:
    report = {"published_at": dt.datetime.now().isoformat(timespec="seconds"), "status": "skipped", "tables": {}}
    if not lake.available(cfg):
        report["reason"] = f"MinIO not reachable at {cfg['platform']['s3_endpoint']}"
        write_json(report, cfg["paths"]["reports"] / "lake_publication.json")
        log.warning("Lake publication skipped: %s", report["reason"])
        return report
    lake.ensure_buckets(cfg)
    cleared: set[tuple[str, str]] = set()
    for layer, table, partition, frame in tables(cfg):
        bucket = lake.bucket(cfg, layer)
        if (bucket, table) not in cleared:          # re-publishing replaces the whole table
            lake.delete_prefix(cfg, bucket, f"{table}/")
            cleared.add((bucket, table))
        uri = lake.upload_dataframe(cfg, lakeify(frame), bucket, object_key(table, partition))
        t = report["tables"].setdefault(f"{layer}.{table}", {"location": f"s3://{bucket}/{table}/", "rows": 0,
                                                               "files": 0, "partition_keys": list(partition)})
        t["rows"] += len(frame)
        t["files"] += 1
        log.info("Published %s rows -> %s", f"{len(frame):,}", uri)
    report["status"] = "published"
    write_json(report, cfg["paths"]["reports"] / "lake_publication.json")
    return report


if __name__ == "__main__":
    run_publish(load_config())
