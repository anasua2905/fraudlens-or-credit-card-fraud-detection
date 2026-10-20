"""ClickHouse: fast OLAP serving of the gold layer, compared with Trino on the same data.

Trino queries the lake in place (federated SQL over Parquet in MinIO, the
"Redshift Spectrum" role). ClickHouse is the columnar analytics database
(the "Redshift" role): this stage loads the gold tables from MinIO into
ClickHouse MergeTree tables with its s3() table function, then runs the same
business questions on both engines, checks that the answers agree, and records
the response times.

ClickHouse reads MinIO through the named collection `gold_lake`, defined in
platform/clickhouse/config.d/named_collections.xml with the read-only analyst
account (gold bucket only), so no credentials appear in SQL or query logs.

    python -m fraudlens.olap              # load + benchmark (skips if ClickHouse is not running)
    python -m fraudlens.olap --query "SELECT count() FROM scored_transactions"
"""
from __future__ import annotations

import argparse
import datetime as dt
import statistics
import time

import pandas as pd

from . import lake
from .crawler import parse_partitions
from .utils import get_logger, load_config, write_json

log = get_logger()

ORDER_BY = {"scored_transactions": "(split, ts)", "daily_kpis": "(split, day)"}
SKIP_TABLES = {"features"}            # model inputs: not an analytics table

# The same business questions in each engine's dialect.
QUERIES = {
    "daily_outcomes": {
        "question": "Fraud caught, missed and false alerts per day (test period)",
        "trino": """SELECT CAST(date_trunc('day', ts) AS date) AS day, count(*) AS txns, sum(is_fraud) AS frauds,
                       sum(CASE WHEN outcome = 'caught' THEN 1 ELSE 0 END) AS caught,
                       sum(CASE WHEN outcome = 'missed' THEN 1 ELSE 0 END) AS missed,
                       sum(CASE WHEN outcome = 'false_alert' THEN 1 ELSE 0 END) AS false_alerts
                FROM {c}.gold.scored_transactions WHERE split = 'test' GROUP BY 1 ORDER BY 1""",
        "clickhouse": """SELECT toDate(ts) AS day, count() AS txns, sum(is_fraud) AS frauds,
                       countIf(outcome = 'caught') AS caught, countIf(outcome = 'missed') AS missed,
                       countIf(outcome = 'false_alert') AS false_alerts
                FROM scored_transactions WHERE split = 'test' GROUP BY day ORDER BY day""",
        "checksum": "caught",
    },
    "category_by_hour": {
        "question": "Fraud rate by merchant category and hour of day",
        "trino": """SELECT category, hour(ts) AS hr, count(*) AS txns, sum(is_fraud) AS frauds
                FROM {c}.gold.scored_transactions GROUP BY 1, 2 ORDER BY 1, 2""",
        "clickhouse": """SELECT category, toHour(ts) AS hr, count() AS txns, sum(is_fraud) AS frauds
                FROM scored_transactions GROUP BY category, hr ORDER BY category, hr""",
        "checksum": "frauds",
    },
    "missed_by_merchant": {
        "question": "Merchants with the most fraud value the model missed",
        "trino": """SELECT merchant, count(*) AS missed, round(sum(amount), 2) AS missed_amount
                FROM {c}.gold.scored_transactions WHERE outcome = 'missed'
                GROUP BY 1 ORDER BY 3 DESC, 1 LIMIT 20""",
        "clickhouse": """SELECT merchant, count() AS missed, round(sum(amount), 2) AS missed_amount
                FROM scored_transactions WHERE outcome = 'missed'
                GROUP BY merchant ORDER BY missed_amount DESC, merchant LIMIT 20""",
        "checksum": "missed",
    },
    "alert_precision_by_state": {
        "question": "Alert volume and precision by state",
        "trino": """SELECT state, sum(alert) AS alerts, sum(CASE WHEN outcome = 'caught' THEN 1 ELSE 0 END) AS caught
                FROM {c}.gold.scored_transactions GROUP BY 1 ORDER BY 1""",
        "clickhouse": """SELECT state, sum(alert) AS alerts, countIf(outcome = 'caught') AS caught
                FROM scored_transactions GROUP BY state ORDER BY state""",
        "checksum": "alerts",
    },
}


class ClickHouse:
    """Minimal client for ClickHouse's HTTP interface (no extra driver needed)."""

    def __init__(self, url: str, user: str, password: str, database: str = "default"):
        import requests
        self.url, self.database = url.rstrip("/"), database
        self.s = requests.Session()
        self.s.headers.update({"X-ClickHouse-User": user, "X-ClickHouse-Key": password})

    def execute(self, sql: str, database: str | None = None, timeout: int = 600) -> str:
        r = self.s.post(self.url, params={"database": database or self.database}, data=sql.encode(),
                        timeout=timeout)
        if r.status_code != 200:
            raise RuntimeError(f"ClickHouse HTTP {r.status_code}: {r.text[:600]}")
        return r.text

    def query_df(self, sql: str) -> pd.DataFrame:
        import io
        text = self.execute(sql.rstrip().rstrip(";") + "\nFORMAT CSVWithNames")
        return pd.read_csv(io.StringIO(text)) if text.strip() else pd.DataFrame()

    def ping(self) -> bool:
        try:
            return self.s.get(f"{self.url}/ping", timeout=3).ok
        except Exception:
            return False


def client(cfg: dict) -> ClickHouse:
    p = cfg["platform"]
    return ClickHouse(p["clickhouse_url"], p["clickhouse_user"], p["clickhouse_password"],
                      p.get("clickhouse_database", "fraudlens"))


def available(cfg: dict) -> bool:
    p = cfg.get("platform", {})
    if not p.get("clickhouse_url"):
        return False
    ch = client(cfg)
    if not ch.ping():
        return False
    try:
        ch.execute("SELECT 1", database="default")
        return True
    except Exception as exc:
        log.warning("ClickHouse answered but rejected the login (%s)", exc)
        return False


def gold_tables(cfg: dict) -> dict[str, list[str]]:
    """Gold tables in the lake and their partition keys (from the Hive-style paths)."""
    out: dict[str, list[str]] = {}
    for k in lake.list_keys(cfg, lake.bucket(cfg, "gold")):
        if not k.endswith(".parquet") or "/_" in k:
            continue
        name = k.split("/")[0]
        if name not in SKIP_TABLES:
            out.setdefault(name, [p for p, _ in parse_partitions(k)])
    return out


def load_sql(table: str, partition_keys: list[str], db: str, source: str = "gold_lake") -> str:
    extra = "".join(f", extract(_path, '{k}=([^/]+)') AS {k}" for k in partition_keys)
    return (f"CREATE OR REPLACE TABLE {db}.{table}\n"
            f"ENGINE = MergeTree ORDER BY {ORDER_BY.get(table, 'tuple()')}\n"
            f"SETTINGS allow_nullable_key = 1\n"
            f"AS SELECT *{extra}\n"
            f"FROM s3({source}, filename = '{table}/**.parquet', format = 'Parquet')\n"
            f"SETTINGS use_hive_partitioning = 0")


def load_gold(cfg: dict, ch: ClickHouse) -> dict:
    db = cfg["platform"].get("clickhouse_database", "fraudlens")
    ch.execute(f"CREATE DATABASE IF NOT EXISTS {db}", database="default")
    loaded = {}
    for table, keys in sorted(gold_tables(cfg).items()):
        t0 = time.perf_counter()
        ch.execute(load_sql(table, keys, db))
        rows = int(ch.execute(f"SELECT count() FROM {db}.{table}").strip())
        loaded[table] = {"rows": rows, "seconds": round(time.perf_counter() - t0, 2), "partition_keys": keys,
                         "order_by": ORDER_BY.get(table, "tuple()")}
        log.info("ClickHouse loaded %s.%s: %s rows in %.1fs", db, table, f"{rows:,}", loaded[table]["seconds"])
    return loaded


def _timed(fn, repeats: int) -> tuple[pd.DataFrame, list[float]]:
    times, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append((time.perf_counter() - t0) * 1000)
    return out, times


def benchmark(cfg: dict, ch: ClickHouse, repeats: int) -> dict:
    trino_ok = True
    try:
        from .crawler import connect
        tconn = connect(cfg)
        tcur = tconn.cursor()
        tcur.execute("SELECT 1")
        tcur.fetchall()
    except Exception as exc:
        log.warning("Trino not reachable (%s); benchmarking ClickHouse only", type(exc).__name__)
        trino_ok = False
    catalog = cfg["platform"].get("trino_catalog", "hive")
    out_dir = cfg["paths"]["reports"] / "olap"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, q in QUERIES.items():
        ch_df, ch_ms = _timed(lambda: ch.query_df(q["clickhouse"]), repeats)
        ch_df.to_csv(out_dir / f"{name}.csv", index=False)
        r = {"question": q["question"], "rows": len(ch_df),
             "clickhouse_ms": {"median": round(statistics.median(ch_ms), 1), "min": round(min(ch_ms), 1)}}
        if trino_ok:
            def run_trino():
                tcur.execute(q["trino"].format(c=catalog))
                rows = tcur.fetchall()
                return pd.DataFrame(rows, columns=[d[0] for d in tcur.description])
            tr_df, tr_ms = _timed(run_trino, repeats)
            col = q["checksum"]
            same = len(tr_df) == len(ch_df) and float(tr_df[col].sum()) == float(ch_df[col].sum())
            r.update({"trino_ms": {"median": round(statistics.median(tr_ms), 1), "min": round(min(tr_ms), 1)},
                      "answers_agree": bool(same),
                      "speedup": round(statistics.median(tr_ms) / max(statistics.median(ch_ms), 1e-6), 1)})
        results[name] = r
        log.info("%-26s ClickHouse %7.1f ms%s", name, r["clickhouse_ms"]["median"],
                 f" | Trino {r['trino_ms']['median']:7.1f} ms | agree={r['answers_agree']}" if trino_ok else "")
    return results


def benchmark_figure(cfg: dict, results: dict) -> str | None:
    if not all("trino_ms" in r for r in results.values()):
        return None
    import matplotlib.pyplot as plt
    from .palette import MUTED, SERIES
    from .viz import save, setup
    setup()
    names = list(results)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = range(len(names))
    ax.barh([i + 0.2 for i in y], [results[n]["trino_ms"]["median"] for n in names], height=0.4,
            label="Trino (Parquet in MinIO)", color=MUTED)
    ax.barh([i - 0.2 for i in y], [results[n]["clickhouse_ms"]["median"] for n in names], height=0.4,
            label="ClickHouse (MergeTree)", color=SERIES[0])
    ax.set_yticks(list(y), [n.replace("_", " ") for n in names])
    ax.set_xlabel("median response time (ms)")
    ax.set_title("Same business questions, two engines")
    ax.legend(loc="lower right", frameon=False)
    from .utils import rel
    return rel(save(fig, cfg["paths"]["figures"] / "20_olap_benchmark"), cfg)


def run_olap(cfg: dict) -> dict:
    report = {"run_at": dt.datetime.now().isoformat(timespec="seconds"), "status": "skipped"}
    if not lake.available(cfg):
        report["reason"] = "MinIO not reachable"
    elif not available(cfg):
        report["reason"] = f"ClickHouse not reachable at {cfg['platform'].get('clickhouse_url')}"
    if report.get("reason"):
        log.warning("OLAP stage skipped: %s", report["reason"])
        write_json(report, cfg["paths"]["reports"] / "olap_benchmark.json")
        return report
    ch = client(cfg)
    report["clickhouse_version"] = ch.execute("SELECT version()", database="default").strip()
    report["tables"] = load_gold(cfg, ch)
    report["queries"] = benchmark(cfg, ch, int(cfg["platform"].get("benchmark_repeats", 5)))
    try:
        report["figure"] = benchmark_figure(cfg, report["queries"])
    except Exception as exc:          # a missing figure never fails the stage
        log.warning("benchmark figure not drawn: %s", exc)
    agree = [r.get("answers_agree") for r in report["queries"].values() if "answers_agree" in r]
    report["status"] = "ok" if all(agree) else "answers_differ"
    write_json(report, cfg["paths"]["reports"] / "olap_benchmark.json")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Load the gold layer into ClickHouse and benchmark it against Trino")
    ap.add_argument("--query", help="run one ClickHouse query and print the result")
    a = ap.parse_args()
    cfg = load_config()
    if a.query:
        print(client(cfg).query_df(a.query).to_string(index=False))
    else:
        rep = run_olap(cfg)
        print(rep["status"], rep.get("reason", ""))


if __name__ == "__main__":
    main()
