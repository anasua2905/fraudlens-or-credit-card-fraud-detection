"""Tests for the cleaning audit, EDA and preprocessing (Step 3)."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_pipeline import _run_once  # noqa: E402
from fraud_pipeline import preprocess as P  # noqa: E402
from fraud_pipeline.datasets import load_features  # noqa: E402
from fraud_pipeline.drift import psi  # noqa: E402
from fraud_pipeline.eda import run_eda  # noqa: E402
from fraud_pipeline.quality import run_quality_audit  # noqa: E402
from fraud_pipeline.utils import read_table  # noqa: E402

_STEP3: dict = {}


def _step3():
    if not _STEP3:
        cfg = _run_once()["cfg"]
        cfg["eda"]["min_state_txns"] = 10
        _STEP3.update(cfg=cfg, audit=run_quality_audit(cfg), eda=run_eda(cfg),
                      pre=P.run_preprocessing(cfg))
    return _STEP3


def test_audit_records_decisions_and_outliers():
    a = _step3()["audit"]
    assert len(a["decisions"]) >= 4
    assert set(a["outliers_train"]) == {"amount", "city_pop", "age", "distance_km"}


def test_eda_writes_summary_and_figures():
    s = _step3()
    for key in ["class_balance", "category", "feature_signal", "drift_psi", "card_level"]:
        assert key in s["eda"]
    figs = list(s["cfg"]["paths"]["figures"].glob("*.png"))
    assert len(figs) >= 8


def test_no_missing_values_in_linear_view():
    for split, info in _step3()["pre"]["views"]["linear"]["splits"].items():
        assert info["nan_cells"] == 0, split


def test_linear_train_is_standardised():
    lin = _step3()["pre"]["views"]["linear"]
    assert lin["train_scaled_mean_abs_max"] < 1e-3
    lo, hi = lin["train_scaled_std_range"]
    assert 0.99 < lo and hi < 1.01


def test_excluded_features_absent_from_both_views():
    pre = _step3()["pre"]
    for view in pre["views"].values():
        for col in pre["excluded_features"]:
            assert col not in view["feature_names"]


def test_saved_preprocessor_reproduces_saved_matrix():
    cfg = _step3()["cfg"]
    pre = joblib.load(cfg["paths"]["artifacts"] / "preprocessor_linear.joblib")
    val = P.add_stationary_features(load_features(cfg, "val"))
    for c in ["category", "gender"]:
        val[c] = val[c].astype(str)
    X = pre.transform(val).astype(np.float32)
    saved = read_table(cfg["paths"]["processed"] / "linear_val")
    assert np.allclose(X.to_numpy(), saved[X.columns].to_numpy(), atol=1e-4)


def test_preprocessor_fitted_on_train_only():
    """The scaler's learned mean must equal the TRAIN mean, not the all-data mean."""
    cfg = _step3()["cfg"]
    pre = joblib.load(cfg["paths"]["artifacts"] / "preprocessor_linear.joblib")
    scaler = pre.named_transformers_["dense"].named_steps["scale"]
    clipper = pre.named_transformers_["dense"].named_steps["clip"]
    train_age = np.clip(load_features(cfg, "train")["age"], clipper.low_[1], clipper.high_[1])
    assert np.isclose(scaler.mean_[1], train_age.mean())


def test_psi_behaviour():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(size=5000))
    assert psi(a, a) < 1e-9
    assert psi(a, a + 2) > 0.25
    b = pd.Series([0] * 90 + [1] * 10)
    assert psi(b, pd.Series([0] * 60 + [1] * 40)) > 0.25


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
