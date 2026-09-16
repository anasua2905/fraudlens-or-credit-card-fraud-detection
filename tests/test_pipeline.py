"""End-to-end tests on a schema-identical synthetic sample."""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import make_sample_data  # noqa: E402
from fraud_pipeline.pipeline import run  # noqa: E402
from fraud_pipeline.utils import load_config, read_table  # noqa: E402
from fraud_pipeline.validate import DataValidationError, validate_raw, assert_valid  # noqa: E402

_CACHE: dict = {}


def _run_once():
    """Run the pipeline once on sample data in a temp project and cache the result."""
    if not _CACHE:
        tmp = Path(tempfile.mkdtemp())
        shutil.copytree(ROOT / "config", tmp / "config")
        sample = make_sample_data.main(str(tmp / "sample"), n_cards=40)
        cfg = load_config(root=tmp)
        report = run(cfg, source="local", local_dir=str(sample))
        _CACHE.update(cfg=cfg, report=report, sample=sample)
    return _CACHE


def _gold():
    c = _run_once()
    parts = [read_table(c["cfg"]["paths"]["gold"] / f"features_{s}").assign(split=s)
             for s in ["train", "val", "test"]]
    return pd.concat(parts, ignore_index=True)


def test_raw_sample_passes_validation():
    raw = pd.read_csv(_run_once()["sample"] / "fraudTrain.csv")
    report = validate_raw(raw, "fraudTrain.csv")
    assert report["status"] in {"PASS", "WARN"}


def test_validation_fails_on_bad_amount():
    raw = pd.read_csv(_run_once()["sample"] / "fraudTrain.csv")
    raw.loc[0, "amt"] = -5
    try:
        assert_valid(validate_raw(raw, "bad"))
    except DataValidationError:
        return
    raise AssertionError("negative amount should fail validation")


def test_pii_removed_from_silver():
    silver = read_table(_run_once()["cfg"]["paths"]["silver"] / "transactions")
    for col in ["cc_num", "first", "last", "street"]:
        assert col not in silver.columns
    assert silver["card_id"].str.len().eq(16).all()
    assert not silver["merchant"].astype(str).str.startswith("fraud_").any()


def test_splits_are_time_ordered_and_complete():
    rep = _run_once()["report"]["split"]["splits"]
    assert pd.Timestamp(rep["train"]["end"]) <= pd.Timestamp(rep["val"]["start"])
    assert pd.Timestamp(rep["val"]["end"]) <= pd.Timestamp(rep["test"]["start"])
    manifest = _run_once()["report"]["manifest"]["files"]
    assert sum(s["rows"] for s in rep.values()) == sum(f["rows"] for f in manifest.values())


def test_velocity_features_match_brute_force_and_exclude_current_row():
    g = _gold().sort_values(["card_id", "ts", "transaction_id"])
    card = g["card_id"].iloc[0]
    sub = g[g["card_id"] == card].reset_index(drop=True)
    for i in range(len(sub)):
        t = sub.loc[i, "ts"]
        prior = sub.iloc[:i]
        in_win = prior[prior["ts"] >= t - pd.Timedelta(hours=24)]
        assert sub.loc[i, "txn_count_24h"] == len(in_win)
        assert np.isclose(sub.loc[i, "amt_sum_24h"], in_win["amount"].sum())
    assert sub.loc[0, "txn_count_1h"] == 0 and sub.loc[0, "card_prior_txn_count"] == 0


def test_history_features_use_only_past():
    g = _gold().sort_values(["card_id", "ts", "transaction_id"])
    sub = g[g["card_id"] == g["card_id"].iloc[0]].reset_index(drop=True)
    i = len(sub) // 2
    expected_mean = sub.loc[: i - 1, "amount"].mean()
    assert np.isclose(sub.loc[i, "amt_to_card_mean"], sub.loc[i, "amount"] / expected_mean)


def test_no_label_derived_features():
    feats = _run_once()["report"]["feature_columns"]["numeric"]
    assert not any("fraud" in f for f in feats)


if __name__ == "__main__":  # allows running without pytest
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
