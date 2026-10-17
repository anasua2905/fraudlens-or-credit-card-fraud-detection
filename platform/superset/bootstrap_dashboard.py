"""Build the FraudLens Superset dashboard over Trino through Superset's REST API (idempotent).

Creates: the Trino database connection, datasets on the gold layer, nine charts and the
"FraudLens - fraud operations" dashboard. Re-running updates nothing that already exists.

    docker compose exec superset python /app/bootstrap_dashboard.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import requests

BASE = os.environ.get("SUPERSET_URL", "http://localhost:8088")
USER, PASSWORD = "admin", os.environ["SUPERSET_ADMIN_PASSWORD"]
TRINO_URI = os.environ.get("TRINO_URI", "trino://superset@trino:8080/hive")
DB_NAME = "FraudLens lake (Trino)"
SLUG = "fraudlens"
TEST = [{"expressionType": "SIMPLE", "subject": "split", "operator": "==", "comparator": "test", "clause": "WHERE"}]


def sql_metric(expr: str, label: str) -> dict:
    return {"expressionType": "SQL", "sqlExpression": expr, "label": label}


class Client:
    def __init__(self):
        self.s = requests.Session()
        for attempt in range(60):
            try:
                r = self.s.post(f"{BASE}/api/v1/security/login",
                                json={"username": USER, "password": PASSWORD, "provider": "db", "refresh": True})
                if r.ok:
                    break
            except requests.ConnectionError:
                pass
            time.sleep(5)
        else:
            sys.exit("Superset did not become ready")
        self.s.headers["Authorization"] = f"Bearer {r.json()['access_token']}"
        csrf = self.s.get(f"{BASE}/api/v1/security/csrf_token/").json()["result"]
        self.s.headers.update({"X-CSRFToken": csrf, "Referer": BASE})

    def get_all(self, resource: str) -> list[dict]:
        r = self.s.get(f"{BASE}/api/v1/{resource}/", params={"q": "(page_size:100)"})
        r.raise_for_status()
        return r.json()["result"]

    def post(self, resource: str, body: dict) -> int:
        r = self.s.post(f"{BASE}/api/v1/{resource}/", json=body)
        if not r.ok:
            raise RuntimeError(f"POST {resource} failed: {r.status_code} {r.text[:400]}")
        return r.json()["id"]

    def put(self, resource: str, rid: int, body: dict) -> None:
        r = self.s.put(f"{BASE}/api/v1/{resource}/{rid}", json=body)
        if not r.ok:
            print(f"PUT {resource}/{rid} failed: {r.status_code} {r.text[:300]}")


def main() -> None:
    c = Client()
    if any(d.get("slug") == SLUG for d in c.get_all("dashboard")):
        print("FraudLens dashboard already exists; nothing to do")
        return

    dbs = {d["database_name"]: d["id"] for d in c.get_all("database")}
    db = dbs.get(DB_NAME) or c.post("database", {"database_name": DB_NAME, "sqlalchemy_uri": TRINO_URI,
                                                 "expose_in_sqllab": True, "allow_run_async": False})
    existing = {(d["schema"], d["table_name"]): d["id"] for d in c.get_all("dataset")}

    def dataset(table: str, dttm: str | None = None) -> int:
        key = ("gold", table)
        if key in existing:
            return existing[key]
        did = c.post("dataset", {"database": db, "schema": "gold", "table_name": table})
        if dttm:
            c.put("dataset", did, {"main_dttm_col": dttm})
        existing[key] = did
        return did

    kpi = dataset("daily_kpis", "day")
    cat = dataset("v_category_performance")
    queue = dataset("v_alert_queue", "ts")
    mon = dataset("monitoring_monthly")
    comp = dataset("model_comparison")

    def chart(name: str, ds: int, viz: str, params: dict) -> int:
        params = {"viz_type": viz, "datasource": f"{ds}__table", **params}
        return c.post("chart", {"slice_name": name, "viz_type": viz, "datasource_id": ds,
                                "datasource_type": "table", "params": json.dumps(params)})

    big = lambda expr, label, fmt: {"metric": sql_metric(expr, label), "adhoc_filters": TEST,  # noqa: E731
                                     "y_axis_format": fmt, "header_font_size": 0.4, "subheader_font_size": 0.15}
    charts = [
        [chart("Alerts per day (test)", kpi, "big_number_total",
               big("SUM(alerts) * 1.0 / COUNT(DISTINCT day)", "Alerts per day", ",.1f")),
         chart("Precision (test)", kpi, "big_number_total",
               big("SUM(caught) * 1.0 / NULLIF(SUM(alerts), 0)", "Precision", ".1%")),
         chart("Recall (test)", kpi, "big_number_total",
               big("SUM(caught) * 1.0 / NULLIF(SUM(caught) + SUM(missed), 0)", "Recall", ".1%")),
         chart("Net benefit (test)", kpi, "big_number_total",
               big("SUM(net_benefit)", "Net benefit ($)", "$,.0f"))],
        [chart("Weekly alert outcomes (test)", kpi, "echarts_timeseries_bar", {
            "x_axis": "day", "time_grain_sqla": "P1W", "adhoc_filters": TEST, "stack": "Stack",
            "metrics": [sql_metric("SUM(caught)", "Fraud caught"), sql_metric("SUM(false_alerts)", "False alerts"),
                        sql_metric("SUM(missed)", "Fraud missed")], "row_limit": 10000, "show_legend": True}),
         chart("Outcomes by merchant category (test)", cat, "echarts_timeseries_bar", {
             "x_axis": "category", "adhoc_filters": TEST, "stack": "Stack", "orientation": "horizontal",
             "metrics": [sql_metric("SUM(caught)", "Fraud caught"), sql_metric("SUM(false_alerts)", "False alerts"),
                         sql_metric("SUM(missed)", "Fraud missed")], "row_limit": 100, "show_legend": True})],
        [chart("Monthly recall and precision (monitoring)", mon, "echarts_timeseries_line", {
            "x_axis": "month", "metrics": [sql_metric("MAX(recall)", "Recall"), sql_metric("MAX(precision)", "Precision")],
            "row_limit": 100, "show_legend": True, "y_axis_format": ".0%"}),
         chart("Model comparison", comp, "table", {
             "query_mode": "raw", "all_columns": ["model", "val_pr_auc", "test_pr_auc", "test_precision",
                                                  "test_recall", "champion"], "row_limit": 20, "order_by_cols": []})],
        [chart("Alert queue: highest scores (test, card ids masked)", queue, "table", {
            "query_mode": "raw", "all_columns": ["ts", "card_id", "merchant", "category", "amount", "score",
                                                 "outcome", "reason_1", "reason_2"],
            "adhoc_filters": TEST, "order_by_cols": [json.dumps(["score", False])], "row_limit": 500})],
    ]

    position = {"DASHBOARD_VERSION_KEY": "v2", "ROOT_ID": {"type": "ROOT", "id": "ROOT_ID", "children": ["GRID_ID"]},
                "GRID_ID": {"type": "GRID", "id": "GRID_ID", "children": [], "parents": ["ROOT_ID"]},
                "HEADER_ID": {"id": "HEADER_ID", "type": "HEADER", "meta": {"text": "FraudLens - fraud operations"}}}
    heights = [30, 60, 55, 70]
    for i, row in enumerate(charts):
        rid = f"ROW-{i}"
        position["GRID_ID"]["children"].append(rid)
        position[rid] = {"type": "ROW", "id": rid, "children": [], "parents": ["ROOT_ID", "GRID_ID"],
                         "meta": {"background": "BACKGROUND_TRANSPARENT"}}
        for cid in row:
            key = f"CHART-{cid}"
            position[rid]["children"].append(key)
            position[key] = {"type": "CHART", "id": key, "children": [], "parents": ["ROOT_ID", "GRID_ID", rid],
                             "meta": {"chartId": cid, "width": 12 // len(row), "height": heights[i]}}
    dash = c.post("dashboard", {"dashboard_title": "FraudLens - fraud operations", "slug": SLUG, "published": True,
                                "position_json": json.dumps(position),
                                "json_metadata": json.dumps({"color_scheme": "supersetColors", "refresh_frequency": 0})})
    for row in charts:
        for cid in row:
            c.put("chart", cid, {"dashboards": [dash]})
    print(f"Dashboard ready: {BASE}/superset/dashboard/{SLUG}/")


if __name__ == "__main__":
    main()
