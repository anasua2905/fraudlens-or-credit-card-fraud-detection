"""Preprocessing: turn gold feature tables into model-ready matrices.

Two views are produced, because model families need different inputs:

  linear  For scale-sensitive models (logistic regression, neural networks,
          autoencoders): log1p on skewed features, clipping at train
          quantiles, median imputation with missing-indicators, cyclical
          hour/day encoding, standard scaling and one-hot categories.
  tree    For tree ensembles (Random Forest, XGBoost, LightGBM): raw values,
          NaN kept (trees handle it), ordinal-encoded categories.

Every transformer is fitted on the TRAINING split only and then applied
unchanged to validation and test. Class imbalance is NOT handled here:
resampling or class weights belong inside model training on the training
split, otherwise they distort validation and test.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from . import features as F
from .datasets import load_features
from .drift import psi, psi_flag
from .utils import get_logger, write_json, write_table

log = get_logger()

# Features listed in config `preprocess.exclude_features` are left out of both
# views. By default these are the running totals, which only ever grow, so
# later periods contain values never seen in training.
# `category_share_card` is a bounded (0-1) replacement for the category count.
DERIVED = ["category_share_card"]
DEFAULT_EXCLUDE = ["card_prior_txn_count", "card_prior_category_count"]

# Non-negative, right-skewed -> log1p before clipping/scaling (linear view)
SKEWED = [
    "secs_since_last_txn", "amt_to_card_mean", "distance_km",
    "txn_count_1h", "txn_count_24h", "txn_count_7d",
    "amt_sum_1h", "amt_sum_24h", "amt_sum_7d",
]
# Bounded 0-1 share -> impute + scale
SHARE = ["category_share_card"]
# Signed, heavy-tailed -> clip only
HEAVY_TAILED = ["amt_zscore_card"]
# Roughly symmetric already -> scale only
DENSE = ["log_amount", "age", "log_city_pop"]
# Periodic -> sin/cos
CYCLICAL = {"hour": 24, "day_of_week": 7}
# Already 0/1
BINARY = ["is_weekend", "is_night", "is_new_merchant_for_card"]
# `amount` is dropped from the linear view: `log_amount` carries the same information
LINEAR_DROPPED = ["amount"]


def add_stationary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bounded replacement for the running category count."""
    prior = df["card_prior_txn_count"]
    df["category_share_card"] = (df["card_prior_category_count"] / prior.where(prior > 0)).astype(float)
    return df


def model_numeric_features(windows: dict, exclude: list[str]) -> list[str]:
    """Numeric inputs used by the models, after exclusions."""
    return [c for c in F.numeric_features(windows) + DERIVED if c not in exclude]


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each column to quantiles learned on the training data (NaN-safe)."""

    def __init__(self, lower: float = 0.001, upper: float = 0.999):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        X = np.asarray(X, dtype=float)
        self.low_ = np.nanquantile(X, self.lower, axis=0)
        self.high_ = np.nanquantile(X, self.upper, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)


def _cyclical(X: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for col, period in CYCLICAL.items():
        angle = 2 * np.pi * X[col].to_numpy(float) / period
        out[f"{col}_sin"], out[f"{col}_cos"] = np.sin(angle), np.cos(angle)
    return pd.DataFrame(out, index=X.index)


def _cyclical_names(_, input_features):
    return np.array([f"{c}_{f}" for c in CYCLICAL for f in ("sin", "cos")], dtype=object)


def _keep(cols: list[str], allowed: list[str]) -> list[str]:
    return [c for c in cols if c in allowed]


def build_linear_preprocessor(q_low: float, q_high: float, allowed: list[str]) -> ColumnTransformer:
    skewed = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("clip", QuantileClipper(q_low, q_high)),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    heavy = Pipeline([
        ("clip", QuantileClipper(q_low, q_high)),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    share = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    dense = Pipeline([("clip", QuantileClipper(q_low, q_high)), ("scale", StandardScaler())])
    cyc = FunctionTransformer(_cyclical, feature_names_out=_cyclical_names)
    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    return ColumnTransformer([
        ("skewed", skewed, _keep(SKEWED, allowed)),
        ("heavy", heavy, _keep(HEAVY_TAILED, allowed)),
        ("share", share, _keep(SHARE, allowed)),
        ("dense", dense, _keep(DENSE, allowed)),
        ("cyclical", cyc, list(CYCLICAL)),  # hour/day are never excluded
        ("binary", "passthrough", _keep(BINARY, allowed)),
        ("cat", cat, F.CATEGORICAL_FEATURES),
    ], remainder="drop", verbose_feature_names_out=False)


def build_tree_preprocessor(numeric: list[str]) -> ColumnTransformer:
    cat = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
    return ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", cat, F.CATEGORICAL_FEATURES),
    ], remainder="drop", verbose_feature_names_out=False)


def _check_columns(numeric: list[str]) -> None:
    covered = set(SKEWED + HEAVY_TAILED + SHARE + DENSE + list(CYCLICAL) + BINARY + LINEAR_DROPPED)
    missing = set(numeric) - covered
    if missing:
        raise ValueError(
            f"Linear view has no transform for {sorted(missing)}. "
            "Add them to preprocess.exclude_features in config/pipeline.yaml "
            "or to a column group in preprocess.py.")


def run_preprocessing(cfg: dict) -> dict:
    pcfg = cfg.get("preprocess", {})
    exclude = list(pcfg.get("exclude_features", DEFAULT_EXCLUDE))
    numeric = model_numeric_features(cfg["features"]["velocity_windows"], exclude)
    _check_columns(numeric)
    q_low, q_high = pcfg.get("winsor_quantiles", [0.001, 0.999])

    splits = {s: load_features(cfg, s) for s in ["train", "val", "test"]}
    for s, df in splits.items():
        splits[s] = df = add_stationary_features(df)
        for c in F.CATEGORICAL_FEATURES:
            splits[s][c] = df[c].astype(str)

    views = {
        "linear": build_linear_preprocessor(q_low, q_high, numeric),
        "tree": build_tree_preprocessor(numeric),
    }
    y_train = splits["train"][F.LABEL]
    report: dict = {
        "fit_on": "train",
        "class_counts_train": {"legit": int((y_train == 0).sum()), "fraud": int(y_train.sum())},
        "suggested_scale_pos_weight": round(float((y_train == 0).sum() / y_train.sum()), 1),
        "excluded_features": exclude,
        "added_features": DERIVED,
        # Drift of each model input between train and VALIDATION (test is never
        # used for decisions). Features flagged 'major' should be reviewed.
        "validation_drift_psi": {
            c: {"psi": round(psi(splits["train"][c], splits["val"][c]), 4),
                "flag": psi_flag(psi(splits["train"][c], splits["val"][c]))}
            for c in numeric
        },
        "views": {},
    }

    for name, pre in views.items():
        pre.set_output(transform="pandas")
        X_train = pre.fit_transform(splits["train"])
        joblib.dump(pre, cfg["paths"]["artifacts"] / f"preprocessor_{name}.joblib")
        info = {"n_features": X_train.shape[1], "feature_names": list(X_train.columns), "splits": {}}
        for s, df in splits.items():
            X = X_train if s == "train" else pre.transform(df)
            info["splits"][s] = {"rows": len(X), "nan_cells": int(X.isna().sum().sum())}
            if pcfg.get("save_matrices", True):
                out = X.astype(np.float32)
                out.insert(0, "transaction_id", df["transaction_id"].to_numpy())
                out[F.LABEL] = df[F.LABEL].to_numpy()
                write_table(out, cfg["paths"]["processed"] / f"{name}_{s}", cfg["storage"]["format"])
        if name == "linear":
            cont = [c for c in X_train.columns if c not in BINARY and not c.startswith(("category_", "gender_", "missingindicator_"))
                    and not c.endswith(("_sin", "_cos"))]
            info["train_scaled_mean_abs_max"] = round(float(X_train[cont].mean().abs().max()), 4)
            info["train_scaled_std_range"] = [round(float(X_train[cont].std().min()), 3),
                                              round(float(X_train[cont].std().max()), 3)]
        report["views"][name] = info
        log.info("Preprocessed '%s' view: %s features", name, X_train.shape[1])

    write_json(report, cfg["paths"]["reports"] / "preprocessing_report.json")
    return report


if __name__ == "__main__":
    from .utils import load_config
    run_preprocessing(load_config())
