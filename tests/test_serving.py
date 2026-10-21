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
    monkeypatch_module.setenv("FRAUDLENS_CASE_DB", str(project["root"] / "cases_dev.db"))
    monkeypatch_module.delenv("FRAUDLENS_API_KEYS", raising=False)
    import fraudlens.api as api
    api = importlib.reload(api)
    return TestClient(api.app), api


def test_new_transaction_flows_to_a_decision_and_into_monitoring(project, monkeypatch_module):
    """The complete live path: transaction -> features -> score -> case -> analyst decision -> monitoring,
    behind role-based API keys."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    monkeypatch_module.setenv("FRAUDLENS_MODEL_DIR", str(project["root"] / "dashboard" / "bundle"))
    monkeypatch_module.setenv("FRAUDLENS_CASE_DB", str(project["root"] / "cases_auth.db"))
    monkeypatch_module.setenv("FRAUDLENS_API_KEYS", "scorer:s-key,analyst:a-key,admin:x-key")
    import fraudlens.api as api
    api = importlib.reload(api)
    c = TestClient(api.app)
    from fraudlens.runtime import example_transaction
    txn = {**example_transaction(api.RT.store), "amount": 2500.0, "transaction_id": "live-1"}

    assert c.get("/health").json()["access_control"] == "api-keys"
    assert c.post("/score", json=txn).status_code == 401                                  # no key
    assert c.post("/score", json=txn, headers={"X-API-Key": "a-key"}).status_code == 403  # analyst can't score
    r = c.post("/score", json=txn, params={"include_features": True}, headers={"X-API-Key": "s-key"}).json()
    assert r["case_id"] and r["features"]["amount"] == 2500.0
    assert r["routed_to"] == ("review queue" if r["alert"] else "approved")
    assert c.get("/alerts", headers={"X-API-Key": "s-key"}).status_code == 403           # scorer can't review

    # force a known alert so the review path is exercised whatever the model says about this card
    case_id = api.CASES.record_scored({**r, "alert": True, "transaction_id": "live-2"}, txn, source="test")
    queue = c.get("/alerts", headers={"X-API-Key": "a-key"}).json()
    assert case_id in {q["case_id"] for q in queue} and all(q["status"] == "open" for q in queue)
    case = c.get(f"/alerts/{case_id}", headers={"X-API-Key": "a-key"}).json()
    assert case["features"] and case["threshold"] == r["threshold"]
    d = c.post(f"/alerts/{case_id}/decision", json={"label": "fraud", "analyst": "alice", "note": "confirmed"},
               headers={"X-API-Key": "a-key"}).json()
    assert d["case"]["status"] == "closed" and d["case"]["decision"] == "fraud"
    assert c.post("/alerts/nope/decision", json={"label": "fraud", "analyst": "a"},
                  headers={"X-API-Key": "a-key"}).status_code == 404

    fb = c.get("/monitoring/feedback", headers={"X-API-Key": "a-key"}).json()
    assert fb["reviewed_alerts"] >= 1 and fb["confirmed_fraud"] >= 1 and fb["labelled_examples"] >= 1
    assert "alice" in fb["analysts"] and fb["verdict"] == "insufficient"
    audit = api.CASES.audit_log()
    assert {"scored", "decided"} <= set(audit["action"]) and "alice" in set(audit["actor"])
    m = c.get("/metrics").text
    assert "fraudlens_open_alerts" in m and "fraudlens_access_denied_total" in m
    monkeypatch_module.delenv("FRAUDLENS_API_KEYS")


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


def test_explanations_are_faithful_and_alerts_have_a_counterfactual(project):
    from fraudlens.explain_quality import run_explain_quality
    q = run_explain_quality(project["cfg"], n_alerts=25)
    d = q["deletion"]
    assert q["alerts_sampled"] > 0 and d["mean_score_drop_top_k"] >= d["mean_score_drop_random_k"]
    assert 0 <= q["stability"]["mean_top_k_jaccard"] <= 1 and q["verdict"] in ("faithful", "weak")


def test_monthly_stability_and_prevalence(project):
    from fraudlens.stability import run_stability
    s = run_stability(project["cfg"])
    assert len(s["monthly"]["months"]) >= 1 and len(s["monthly"]["checks"]) == 4
    rows = s["prevalence"]["rows"]
    prec = [r["precision_fpr_x1"] for r in rows if not r["is_test_rate"]]
    assert prec == sorted(prec)                       # rarer fraud -> lower precision
    assert all(r["precision_fpr_x1"] >= r["precision_fpr_x5"] for r in rows)
