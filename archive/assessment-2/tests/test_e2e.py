"""End-to-end run, champion selection, verification and dashboard data."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import make_sample_data  # noqa: E402
from fraud_pipeline import dashboard_data as D  # noqa: E402
from fraud_pipeline.run_all import run_all  # noqa: E402
from fraud_pipeline.selection import select_champion  # noqa: E402
from fraud_pipeline.utils import load_config  # noqa: E402
from fraud_pipeline.verify import run_verification  # noqa: E402

_RUN: dict = {}


def _e2e():
    if not _RUN:
        tmp = Path(tempfile.mkdtemp())
        shutil.copytree(ROOT / "config", tmp / "config")
        sample = make_sample_data.main(str(tmp / "sample"), n_cards=150)
        cfg = load_config(root=tmp)
        cfg["eda"]["min_state_txns"] = 10
        cfg["model"]["selection_bootstrap"] = 100
        cfg["model"]["permutation_importance_rows"] = 2000
        summary = run_all(cfg, source="local", local_dir=str(sample))
        _RUN.update(cfg=cfg, summary=summary, root=tmp)
    return _RUN


# ---------------------------------------------------------------- selection
def _scores(n, rate, seed):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    best = y * 1.2 + rng.normal(0, 0.5, n)
    return y, best, best + rng.normal(0, 0.02, n), y * 0.6 + rng.normal(0, 0.5, n)


def test_tie_goes_to_simpler_model():
    from sklearn.metrics import average_precision_score as ap
    y, best, near, weak = _scores(100_000, 0.006, 0)
    sc = {"hybrid_gb": best, "hist_gb": near, "logistic": weak}
    pr = {k: ap(y, v) for k, v in sc.items()}
    pr["hybrid_gb"] = max(pr.values()) + 1e-4
    sel = select_champion(y, sc, pr, n_boot=200)
    assert sel["best_by_validation_pr_auc"] == "hybrid_gb"
    assert sel["champion"] == "hist_gb"
    assert not sel["comparisons"]["logistic"]["equivalent_to_best"]


def test_scarce_data_keeps_best_model():
    """A wide interval from few frauds must not count as a tie."""
    from sklearn.metrics import average_precision_score as ap
    y, best, near, weak = _scores(3_000, 0.006, 1)
    sc = {"hybrid_gb": best, "logistic": weak}
    pr = {k: ap(y, v) for k, v in sc.items()}
    pr["hybrid_gb"] = max(pr.values()) + 1e-4
    assert select_champion(y, sc, pr, n_boot=200)["champion"] == "hybrid_gb"


# ------------------------------------------------------------- end to end
def test_full_pipeline_runs_and_verifies():
    s = _e2e()["summary"]
    assert s["stages_run"] == ["ingest", "quality", "eda", "preprocess", "train", "score", "verify"]
    assert s["verification"] == "PASS"


def test_results_record_selection_and_imbalance_method():
    cfg = _e2e()["cfg"]
    res = json.loads((cfg["paths"]["reports"] / "model_results.json").read_text())
    assert res["selection"]["champion"] == res["champion"]
    assert all("class_weight='balanced'" in v for k, v in res["imbalance_handling"].items()
               if k in {"logistic", "hist_gb", "hybrid_gb"})
    assert not any("smote" in v.lower() for v in res["imbalance_handling"].values())


def test_verification_catches_a_tampered_champion():
    """If the reported champion disagrees with the code's rule, verification must FAIL."""
    cfg = _e2e()["cfg"]
    path = cfg["paths"]["reports"] / "model_results.json"
    original = path.read_text()
    try:
        res = json.loads(original)
        others = [m for m in res["results"] if m != res["champion"] and m != "isolation_forest"]
        res["champion"] = others[0]
        path.write_text(json.dumps(res))
        rep = run_verification(cfg, verify_checksums=False)
        failed = {c["check"] for c in rep["checks"] if c["status"] == "FAIL"}
        assert rep["status"] == "FAIL"
        assert "reported champion = champion chosen by the rule" in failed
    finally:
        path.write_text(original)
    assert run_verification(cfg, verify_checksums=False)["status"] == "PASS"


# ---------------------------------------------------------------- dashboard
def test_dashboard_kpis_match_model_results():
    cfg = _e2e()["cfg"]
    data = D.load_dashboard(cfg)
    meta, res = data["meta"], data["results"]
    df = D.apply_filters(data["df"], "test")
    k = D.kpis(df, meta["threshold"], meta["investigation_cost"], meta["recovery_rate"])
    r = res["results"][meta["champion"]]["test"]
    assert (k["true_positives"], k["false_positives"], k["false_negatives"]) == \
        (r["true_positives"], r["false_positives"], r["false_negatives"])
    assert abs(k["net_benefit"] - r["net_benefit"]) < 0.01


def test_dashboard_views_reconcile():
    cfg = _e2e()["cfg"]
    data = D.load_dashboard(cfg)
    df = D.apply_filters(data["df"], "test")
    thr = data["meta"]["threshold"]
    k = D.kpis(df, thr, 15, 0.9)
    assert int(D.daily(df, thr)["alerts"].sum()) == k["alerts"]
    assert int(D.card_queue(df, thr)["alerts"].sum()) == k["alerts"]
    for by in ["category", "hour", "amount_band"]:
        assert int(D.segment(df, thr, by)["alerts"].sum()) == k["alerts"]


def test_card_queue_reduces_review_items():
    cfg = _e2e()["cfg"]
    data = D.load_dashboard(cfg)
    df = D.apply_filters(data["df"], "test")
    thr = data["meta"]["threshold"]
    assert len(D.card_queue(df, thr)) <= D.kpis(df, thr, 15, 0.9)["alerts"]


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
