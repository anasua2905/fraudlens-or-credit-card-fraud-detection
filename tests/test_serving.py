"""Stages 7-8: batch scoring, online feature store, API, dashboard data, bundle, monitoring."""
from __future__ import annotations

import importlib
import json

import pytest

from fraudlens import dashboard_data as D
from fraudlens.datasets import load_features, load_silver
from fraudlens.verify import online_batch_parity


def test_online_store_reproduces_batch_features(project):
    cfg = project["cfg"]
    parity = online_batch_parity(cfg, load_silver(cfg), load_features(cfg, "test"), n=500)
    assert parity["match_rate"] >= 0.995, parity


def test_dashboard_kpis_match_model_results(project):
    data = D.load_dashboard(project["root"])
    meta, res = data["meta"], data["results"]
    k = D.kpis(D.apply_filters(data["df"], "test"), meta["threshold"], meta["investigation_cost"], meta["recovery_rate"])
    r = res["results"][meta["champion"]]["test"]
    assert (k["true_positives"], k["false_positives"], k["false_negatives"]) == \
        (r["true_positives"], r["false_positives"], r["false_negatives"])
    assert abs(k["net_benefit"] - r["net_benefit"]) < 0.01


def test_dashboard_views_reconcile(project):
    data = D.load_dashboard(project["root"])
    df = D.apply_filters(data["df"], "test")
    thr = data["meta"]["threshold"]
    alerts = D.kpis(df, thr, 15, 0.9)["alerts"]
    assert int(D.daily(df, thr)["alerts"].sum()) == alerts
    assert int(D.card_queue(df, thr)["alerts"].sum()) == alerts
    for by in ["category", "hour", "amount_band"]:
        assert int(D.segment(df, thr, by)["alerts"].sum()) == alerts


def test_bundle_is_self_contained_and_matches_pipeline(project):
    bundle = project["root"] / "dashboard" / "bundle"
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["total_mb"] < 95
    full = D.load_dashboard(project["root"])
    small = D.load_dashboard(project["root"], prefer="bundle")
    assert small["kind"] == "bundle" and len(small["df"]) == len(full["df"])
    thr = full["meta"]["threshold"]
    assert D.kpis(D.apply_filters(small["df"], "test"), thr, 15, 0.9)["alerts"] == \
        D.kpis(D.apply_filters(full["df"], "test"), thr, 15, 0.9)["alerts"]
    req = (project["root"] / "dashboard" / "requirements.txt").read_text()
    assert "scikit-learn==" in req


@pytest.fixture(scope="module")
def client(project, monkeypatch_module):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    monkeypatch_module.setenv("FRAUDLENS_MODEL_DIR", str(project["root"] / "dashboard" / "bundle"))
    monkeypatch_module.setenv("FRAUDLENS_FEEDBACK_FILE", str(project["root"] / "feedback.csv"))
    import fraudlens.api as api
    api = importlib.reload(api)
    return TestClient(api.app), api


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_api_health_score_feedback_metrics(client):
    c, api = client
    assert c.get("/health").json()["status"] == "ok"
    from fraudlens.runtime import example_transaction
    txn = example_transaction(api.RT.store)
    r = c.post("/score", json=txn, params={"update_state": False}).json()
    assert 0 <= r["score"] <= 1 and r["alert"] == (r["score"] >= r["threshold"])
    small = c.post("/score", json={**txn, "amount": 1.0}, params={"update_state": False}).json()
    assert small["score"] <= r["score"]
    batch = c.post("/score/batch", json=[txn, {**txn, "amount": 5.0}], params={"update_state": False})
    assert batch.status_code == 200 and len(batch.json()) == 2
    assert c.post("/feedback", json={"transaction_id": "t1", "label": "fraud"}).json()["status"] == "recorded"
    m = c.get("/metrics").text
    assert "fraudlens_transactions_scored_total 4.0" in m and "fraudlens_feedback_total" in m
    assert c.post("/score", json={**txn, "amount": -1}).status_code == 422


def test_monitoring_summary_and_policy(project):
    s = json.loads((project["cfg"]["paths"]["monitoring"] / "summary.json").read_text())
    assert s["months"] == len(s["monthly"]) >= 2
    for row in s["monthly"]:
        assert row["retrain_recommended"] == bool(row["reasons"])


def test_serving_path_does_not_need_the_plotting_stack():
    """The hosted dashboard installs only dashboard/requirements.txt (no seaborn/matplotlib)."""
    import subprocess
    import sys
    code = ("import sys; import fraudlens.runtime, fraudlens.dashboard_data, fraudlens.explain; "
            "bad = [m for m in ('seaborn', 'matplotlib') if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
