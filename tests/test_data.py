"""Stages 2-4: acquisition, validation, cleaning, features, split, EDA."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fraudlens.utils import read_table
from fraudlens.validate import DataValidationError, assert_valid, validate_raw


def _gold(cfg):
    return pd.concat([read_table(cfg["paths"]["gold"] / f"features_{s}").assign(split=s)
                      for s in ["train", "val", "test"]], ignore_index=True)


def test_manifest_records_checksums_and_licence(project):
    m = json.loads((project["cfg"]["paths"]["raw"] / "manifest.json").read_text())
    assert m["licence"] == "CC0-1.0"
    assert all(len(f["sha256"]) == 64 for f in m["files"].values())


def test_raw_sample_passes_validation(project):
    raw = pd.read_csv(project["sample"] / "fraudTrain.csv")
    assert validate_raw(raw, "fraudTrain.csv")["status"] in {"PASS", "WARN"}


def test_validation_fails_on_bad_amount(project):
    raw = pd.read_csv(project["sample"] / "fraudTrain.csv")
    raw.loc[0, "amt"] = -5
    with pytest.raises(DataValidationError):
        assert_valid(validate_raw(raw, "bad"))


def test_pii_removed_from_silver(project):
    silver = read_table(project["cfg"]["paths"]["silver"] / "transactions")
    assert not {"cc_num", "first", "last", "street"} & set(silver.columns)
    assert silver["card_id"].astype(str).str.len().eq(16).all()
    assert not silver["merchant"].astype(str).str.startswith("fraud_").any()


def test_splits_are_time_ordered_and_complete(project):
    rep = json.loads((project["cfg"]["paths"]["reports"] / "pipeline_run.json").read_text())["split"]["splits"]
    assert pd.Timestamp(rep["train"]["end"]) <= pd.Timestamp(rep["val"]["start"])
    assert pd.Timestamp(rep["val"]["end"]) <= pd.Timestamp(rep["test"]["start"])


def test_velocity_features_match_brute_force_and_exclude_current_row(project):
    g = _gold(project["cfg"]).sort_values(["card_id", "ts", "transaction_id"])
    sub = g[g["card_id"] == g["card_id"].iloc[0]].reset_index(drop=True)
    for i in range(len(sub)):
        prior = sub.iloc[:i]
        in_win = prior[prior["ts"] >= sub.loc[i, "ts"] - pd.Timedelta(hours=24)]
        assert sub.loc[i, "txn_count_24h"] == len(in_win)
        assert np.isclose(sub.loc[i, "amt_sum_24h"], in_win["amount"].sum())
    assert sub.loc[0, "txn_count_1h"] == 0 and sub.loc[0, "card_prior_txn_count"] == 0


def test_history_features_use_only_past(project):
    g = _gold(project["cfg"]).sort_values(["card_id", "ts", "transaction_id"])
    sub = g[g["card_id"] == g["card_id"].iloc[0]].reset_index(drop=True)
    i = len(sub) // 2
    assert np.isclose(sub.loc[i, "amt_to_card_mean"], sub.loc[i, "amount"] / sub.loc[: i - 1, "amount"].mean())


def test_no_label_derived_features(project):
    rep = json.loads((project["cfg"]["paths"]["reports"] / "pipeline_run.json").read_text())
    assert not any("fraud" in f for f in rep["feature_columns"]["numeric"])


def test_eda_outputs(project):
    P = project["cfg"]["paths"]
    summary = json.loads((P["reports"] / "eda_summary.json").read_text())
    assert {"class_balance", "category", "time", "feature_signal"} <= set(summary)
    assert len(list(P["figures"].glob("0*.png"))) >= 9
    assert (P["interactive"] / "eda_explorer.html").exists()
