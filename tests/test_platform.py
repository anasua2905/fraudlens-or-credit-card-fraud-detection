"""Data platform: lake layout, crawler schema inference, governance rules, service configs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from fraudlens.crawler import DiscoveredTable, arrow_to_trino, ddl, parse_partitions
from fraudlens.monitor import parse_evidently
from fraudlens.publish import daily_kpis, lakeify, object_key
from fraudlens.utils import read_table

PLATFORM = Path(__file__).resolve().parents[1] / "platform"


def test_partition_paths_round_trip():
    key = object_key("scored_transactions", {"split": "test"})
    assert key == "scored_transactions/split=test/part-0.parquet"
    assert parse_partitions(key) == [("split", "test")]


def test_lakeify_produces_engine_friendly_types():
    df = pd.DataFrame({"c": pd.Categorical(["a", "b"]), "ts": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                       "x": [1.0, 2.0]})
    out = lakeify(df)
    assert str(out["ts"].dtype) == "datetime64[us]" and out["c"].tolist() == ["a", "b"]


def test_arrow_types_map_to_trino():
    pa = pytest.importorskip("pyarrow")
    assert arrow_to_trino(pa.int8()) == "TINYINT"
    assert arrow_to_trino(pa.float32()) == "REAL"
    assert arrow_to_trino(pa.timestamp("us")) == "TIMESTAMP(3)"
    assert arrow_to_trino(pa.dictionary(pa.int32(), pa.string())) == "VARCHAR"


def test_ddl_registers_external_partitioned_table():
    t = DiscoveredTable("gold", "scored_transactions", "fraudlens-gold", "PARQUET",
                        ["scored_transactions/split=test/part-0.parquet"], ["split"],
                        [("score", "REAL"), ("split", "VARCHAR")])
    stmt = ddl(t, "hive")[0]
    assert "external_location = 's3a://fraudlens-gold/scored_transactions/'" in stmt
    assert "partitioned_by = ARRAY['split']" in stmt


def test_daily_kpis_reconcile_with_scored(project):
    scored = read_table(project["cfg"]["paths"]["gold"] / "scored_transactions")
    k = daily_kpis(scored, 15, 0.9)
    assert int(k["alerts"].sum()) == int(scored["alert"].sum())
    assert int(k["caught"].sum()) == int((scored["outcome"] == "caught").sum())


def test_parse_evidently_result():
    d = {"metrics": [{"metric_id": "DriftedColumnsCount(drift_share=0.5)", "value": {"count": 2, "share": 0.25}},
                     {"metric_id": "ValueDrift(column=amount,method=Wasserstein distance (normed),threshold=0.1)",
                      "value": 0.3}]}
    out = parse_evidently(d)
    assert out["share"] == 0.25 and out["columns"]["amount"]["drifted"] is True


def test_trino_rules_mask_existing_silver_columns():
    rules = json.loads((PLATFORM / "trino" / "etc" / "rules.json").read_text())
    from fraudlens.clean import SILVER_COLUMNS
    masked = [c["name"] for r in rules["tables"] for c in r.get("columns", []) if r.get("schema") == "silver"]
    assert masked and set(masked) <= set(SILVER_COLUMNS)


def test_compose_and_dashboards_are_valid():
    compose = yaml.safe_load((PLATFORM / "docker-compose.yml").read_text())
    assert {"minio", "hive-metastore", "trino", "superset", "mlflow", "api", "prometheus", "grafana",
            "nifi", "nifi-init", "clickhouse"} <= set(compose["services"])
    graf = json.loads((PLATFORM / "grafana" / "dashboards" / "fraudlens.json").read_text())
    assert len(graf["panels"]) >= 8
    biz = json.loads((PLATFORM / "grafana" / "dashboards" / "fraudlens_business.json").read_text())
    assert all(t["datasource"]["uid"] == "clickhouse" for p in biz["panels"] for t in p["targets"])
    for p in ("pipeline", "trino", "analyst", "ingest"):
        json.loads((PLATFORM / "minio" / "policies" / f"{p}.json").read_text())


def test_clickhouse_load_sql_reads_gold_through_named_collection():
    from fraudlens.olap import QUERIES, load_sql
    sql = load_sql("scored_transactions", ["split"], "fraudlens")
    assert "s3(gold_lake, filename = 'scored_transactions/**.parquet'" in sql
    assert "extract(_path, 'split=([^/]+)') AS split" in sql and "ORDER BY (split, ts)" in sql
    assert "ORDER BY tuple()" in load_sql("daily_kpis_x", [], "fraudlens")
    for q in QUERIES.values():                       # both dialects for every question
        assert q["trino"].count("{c}.gold.") == 1 and "FROM scored_transactions" in q["clickhouse"]
    xml = (PLATFORM / "clickhouse" / "config.d" / "named_collections.xml").read_text()
    assert 'from_env="ANALYST_ACCESS_KEY"' in xml and "fraudlens-gold" in xml


def test_nifi_flow_maps_display_names_to_property_names():
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_flow", PLATFORM / "nifi" / "build_flow.py")
    bf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bf)
    component = {"descriptors": {
        "Bucket": {"name": "Bucket", "displayName": "Bucket"},
        "s3-path-style": {"name": "s3-path-style", "displayName": "Use Path Style Access",
                          "allowableValues": [{"allowableValue": {"value": "true", "displayName": "true"}},
                                              {"allowableValue": {"value": "false", "displayName": "false"}}]},
        "Region": {"name": "Region", "displayName": "Region",
                   "allowableValues": [{"allowableValue": {"value": "us-east-1",
                                                           "displayName": "US East (N. Virginia)"}}]}}}
    out = bf.NiFi._map_properties(component, {"Bucket": "fraudlens-raw", "Use Path Style Access": "true",
                                              "Region": "us-east-1", "lake.key": "x"})
    assert out == {"Bucket": "fraudlens-raw", "s3-path-style": "true", "Region": "us-east-1", "lake.key": "x"}
    with pytest.raises(SystemExit):
        bf.NiFi._map_properties(component, {"Region": "mars-1"})
