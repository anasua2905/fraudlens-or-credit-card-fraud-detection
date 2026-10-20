"""Stages 5-6: modelling, selection, evaluation, explanation, verification."""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import average_precision_score as ap

from fraudlens import evaluate as E
from fraudlens import models as M
from fraudlens.selection import select_champion
from fraudlens.verify import run_verification

TCFG = {"alerts_per_day": 5, "investigation_cost": 15, "recovery_rate": 0.9}


def _results(project):
    return json.loads((project["cfg"]["paths"]["reports"] / "model_results.json").read_text())


def test_ranking_metrics_are_sane():
    y = np.array([0, 0, 1, 1])
    assert E.ranking_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))["roc_auc"] == 1.0
    assert E.ranking_metrics(y, np.array([0.9, 0.8, 0.2, 0.1]))["roc_auc"] == 0.0


def test_confusion_and_business_metrics_agree():
    y = np.array([0, 1, 1, 0])
    s = np.array([0.1, 0.9, 0.4, 0.8])
    a = np.array([10.0, 100.0, 50.0, 20.0])
    c = E.confusion_at(y, s, 0.5)
    assert (c["true_positives"], c["false_positives"], c["false_negatives"]) == (1, 1, 1)
    b = E.business_metrics(y, s, a, 0.5, days=1, investigation_cost=10, recovery_rate=1.0)
    assert b["fraud_amount_caught"] == 100.0 and b["net_benefit"] == 100.0 - 2 * 10


def test_alert_budget_threshold_respects_capacity():
    rng = np.random.default_rng(0)
    s, y, a = rng.random(10000), (rng.random(10000) < 0.01).astype(int), rng.random(10000) * 100
    t = E.choose_threshold(y, s, a, days=10, cfg_thr=TCFG, strategy="alert_budget")
    assert abs(int((s >= t).sum()) - 50) <= 5


def _scores(n, rate, seed):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    best = y * 1.2 + rng.normal(0, 0.5, n)
    return y, best, best + rng.normal(0, 0.02, n), y * 0.6 + rng.normal(0, 0.5, n)


def test_tie_goes_to_simpler_model():
    y, best, near, weak = _scores(100_000, 0.006, 0)
    sc = {"hybrid_gb": best, "hist_gb": near, "logistic": weak}
    pr = {k: ap(y, v) for k, v in sc.items()}
    pr["hybrid_gb"] = max(pr.values()) + 1e-4
    sel = select_champion(y, sc, pr, n_boot=200)
    assert sel["champion"] == "hist_gb"
    assert not sel["comparisons"]["logistic"]["equivalent_to_best"]


def test_scarce_data_keeps_best_model():
    y, best, near, weak = _scores(3_000, 0.006, 1)
    sc = {"hybrid_gb": best, "logistic": weak}
    pr = {k: ap(y, v) for k, v in sc.items()}
    pr["hybrid_gb"] = max(pr.values()) + 1e-4
    assert select_champion(y, sc, pr, n_boot=200)["champion"] == "hybrid_gb"


def test_all_models_trained_tuned_and_saved(project):
    res = _results(project)
    names = {r["model"] for r in res["comparison"]}
    expected = {m for m in project["cfg"]["model"]["models"] if M.available(m)}
    assert names == expected
    assert set(res["tuning"]) == {m for m in project["cfg"]["model"]["tuning"]["models"] if M.available(m)}
    for n in names:
        assert (project["cfg"]["paths"]["artifacts"] / f"model_{n}.joblib").exists()


def test_imbalance_by_weighting_not_resampling(project):
    res = _results(project)
    assert not any("smote" in v.lower() for v in res["imbalance_handling"].values())


def test_champion_beats_baseline_and_thresholds_from_validation(project):
    res = _results(project)
    r = res["results"][res["champion"]]
    assert r["test"]["pr_auc"] > r["test"]["pr_auc_baseline"]
    for name, rr in res["results"].items():
        assert abs(rr["test"]["threshold"] - rr["thresholds_from_validation"][res["threshold_strategy"]]) < 1e-6


def test_confidence_intervals_and_monthly_breakdown(project):
    res = _results(project)
    ci = res["champion_test_ci95"]["pr_auc"]
    assert ci["low"] <= res["results"][res["champion"]]["test"]["pr_auc"] <= ci["high"]
    assert len(res["champion_monthly_test"]) >= 2


def test_shap_explanations_are_additive_and_cover_alerts(project):
    P = project["cfg"]["paths"]
    ex = json.loads((P["reports"] / "explanations.json").read_text())
    res = _results(project)
    assert ex["alerts_explained"] == res["results"][res["champion"]]["test"]["alerts"]
    assert ex["explainer"] in {"tree", "permutation"}
    assert (P["figures"] / "17_shap_beeswarm.png").exists()


def test_verification_passes_and_catches_tampering(project):
    cfg = project["cfg"]
    assert run_verification(cfg, verify_checksums=True)["status"] == "PASS"
    path = cfg["paths"]["reports"] / "model_results.json"
    original = path.read_text()
    try:
        res = json.loads(original)
        res["champion"] = [m for m in res["results"] if m not in (res["champion"], "isolation_forest")][0]
        path.write_text(json.dumps(res))
        rep = run_verification(cfg, verify_checksums=False)
        assert rep["status"] == "FAIL"
        assert "reported champion = champion chosen by the rule" in {c["check"] for c in rep["checks"]
                                                                    if c["status"] == "FAIL"}
    finally:
        path.write_text(original)


def test_deployed_model_scores_exactly_like_training(project):
    """Train/serve parity: the packaged FraudModel, scoring raw features, must give the same
    scores as the champion scoring the stored training matrices (found and fixed: a
    float32/float64 mismatch moved values across tree split thresholds)."""
    import pandas as pd

    from fraudlens.datasets import load_features
    from fraudlens.scoring import FraudModel
    from fraudlens.utils import read_table

    cfg = project["cfg"]
    fm = FraudModel.load(cfg["paths"]["artifacts"] / "fraudlens_model.joblib")
    feats = load_features(cfg, "val")
    X = read_table(cfg["paths"]["processed"] / f"{fm.view}_val")
    assert (X.pop("transaction_id").to_numpy() == feats["transaction_id"].to_numpy()).all()
    X = X.drop(columns=["is_fraud"])
    for c in M.CATEGORICAL_IN_TREE_VIEW:
        if fm.view == "tree" and c in X:
            X[c] = X[c].fillna(-1).astype("int32")
    if fm.champion == "hybrid_gb":
        lin = read_table(cfg["paths"]["processed"] / "linear_val").drop(columns=["transaction_id", "is_fraud"])
        X = X.assign(anomaly_score=M.anomaly_score(fm.iso_model, lin))
    assert np.abs(fm.predict(feats) - M.positive_scores(fm.model, X)).max() < 1e-12
    assert isinstance(fm.transform(feats.head(5)), pd.DataFrame)
