"""Tests for model training, evaluation and threshold selection (Step 4)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_step3 import _step3  # noqa: E402
from fraud_pipeline import evaluate as E  # noqa: E402
from fraud_pipeline import models as M  # noqa: E402
from fraud_pipeline import train as T  # noqa: E402

TCFG = {"investigation_cost": 10, "recovery_rate": 1.0, "alerts_per_day": 5}
_RUN: dict = {}


def _run():
    if not _RUN:
        cfg = _step3()["cfg"]
        cfg["model"]["models"] = ["logistic", "hist_gb", "isolation_forest", "hybrid_gb"]
        cfg["model"]["permutation_importance_rows"] = 2000
        _RUN.update(cfg=cfg, report=T.run_training(cfg))
    return _RUN


def test_ranking_metrics_are_sane():
    y = np.array([0, 0, 1, 1])
    assert E.ranking_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))["roc_auc"] == 1.0
    assert E.ranking_metrics(y, np.array([0.9, 0.8, 0.2, 0.1]))["roc_auc"] == 0.0


def test_confusion_and_business_metrics_agree():
    y = np.array([0, 1, 1, 0])
    scores = np.array([0.1, 0.9, 0.4, 0.8])
    amounts = np.array([10.0, 100.0, 50.0, 20.0])
    c = E.confusion_at(y, scores, 0.5)
    assert (c["true_positives"], c["false_positives"], c["false_negatives"]) == (1, 1, 1)
    b = E.business_metrics(y, scores, amounts, 0.5, days=1, investigation_cost=10, recovery_rate=1.0)
    assert b["fraud_amount_caught"] == 100.0 and b["fraud_amount_missed"] == 50.0
    assert b["net_benefit"] == 100.0 - 2 * 10


def test_alert_budget_threshold_respects_capacity():
    rng = np.random.default_rng(0)
    scores = rng.random(10000)
    y = (rng.random(10000) < 0.01).astype(int)
    amounts = rng.random(10000) * 100
    t = E.choose_threshold(y, scores, amounts, days=10, cfg_thr=TCFG, strategy="alert_budget")
    assert abs(int((scores >= t).sum()) - 50) <= 5  # 5 alerts/day x 10 days


def test_all_models_trained_and_saved():
    r = _run()
    names = {row["model"] for row in r["report"]["comparison"]}
    assert names == {"logistic", "hist_gb", "isolation_forest", "hybrid_gb"}
    for name in names:
        assert (r["cfg"]["paths"]["artifacts"] / f"model_{name}.joblib").exists()


def test_champion_is_supervised_and_beats_baseline():
    rep = _run()["report"]
    assert rep["champion"] in M.SUPERVISED
    res = rep["results"][rep["champion"]]["test"]
    assert res["pr_auc"] > res["pr_auc_baseline"]


def test_thresholds_come_from_validation_only():
    """The reported threshold must be the one selected on validation."""
    rep = _run()["report"]
    for name, r in rep["results"].items():
        chosen = r["thresholds_from_validation"][rep["threshold_strategy"]]
        assert abs(r["test"]["threshold"] - chosen) < 1e-6, name


def test_score_split_matches_training_scores():
    r = _run()
    champ = r["report"]["champion"]
    scored = T.score_split(r["cfg"], champ, "test")
    assert len(scored) == r["report"]["results"][champ]["test"]["alerts"] + \
        r["report"]["results"][champ]["test"]["false_negatives"] + \
        r["report"]["results"][champ]["test"]["true_negatives"]
    assert scored["score"].notna().all()


def test_error_breakdown_adds_up():
    r = _run()
    champ = r["report"]["champion"]
    res = r["report"]["results"][champ]
    thr = res["thresholds_from_validation"][r["report"]["threshold_strategy"]]
    tabs = T.error_breakdown(r["cfg"], champ, "test", thr)
    assert int(tabs["by_category"]["caught"].sum()) == res["test"]["true_positives"]
    assert int(tabs["missed_fraud_summary"]["count"].iloc[0]) == res["test"]["false_negatives"]
    assert int(tabs["false_alert_summary"]["count"].iloc[0]) == res["test"]["false_positives"]


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
