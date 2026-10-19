"""Model zoo (lifecycle stage 5).

| Model            | Library       | Input view | Role |
|------------------|---------------|------------|------|
| logistic         | scikit-learn  | linear     | Interpretable baseline |
| hist_gb          | scikit-learn  | tree       | Gradient boosting with no extra dependency |
| xgboost          | XGBoost       | tree       | Gradient boosting |
| lightgbm         | LightGBM      | tree       | Gradient boosting |
| isolation_forest | scikit-learn  | linear     | Unsupervised anomaly detector, fitted on legitimate transactions only |
| hybrid_gb        | scikit-learn  | tree       | Gradient boosting with the anomaly score as an extra feature |

Class imbalance is handled inside training with class weights only
(`class_weight="balanced"`, or XGBoost's equivalent `scale_pos_weight`).
Nothing is resampled, and validation and test are never altered.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression

VIEW_BY_MODEL = {
    "logistic": "linear",
    "hist_gb": "tree",
    "xgboost": "tree",
    "lightgbm": "tree",
    "isolation_forest": "linear",
    "hybrid_gb": "tree",
}
SUPERVISED = ["logistic", "hist_gb", "xgboost", "lightgbm", "hybrid_gb"]
CATEGORICAL_IN_TREE_VIEW = ["category", "gender"]
LIBRARY = {"logistic": "scikit-learn", "hist_gb": "scikit-learn", "xgboost": "xgboost",
           "lightgbm": "lightgbm", "isolation_forest": "scikit-learn", "hybrid_gb": "scikit-learn"}

# Default hyper-parameters; tuning.py searches around them.
DEFAULTS = {
    "hist_gb": {"learning_rate": 0.06, "max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 1.0},
    "xgboost": {"n_estimators": 400, "learning_rate": 0.06, "max_depth": 6, "min_child_weight": 5,
                "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 3.0},
    "lightgbm": {"n_estimators": 400, "learning_rate": 0.06, "num_leaves": 31, "min_child_samples": 50,
                 "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 3.0},
}

SEARCH_SPACE = {
    "hist_gb": {"learning_rate": [0.03, 0.06, 0.1], "max_leaf_nodes": [15, 31, 63],
                "min_samples_leaf": [20, 50, 100], "l2_regularization": [0.0, 1.0, 3.0]},
    "xgboost": {"n_estimators": [200, 400, 800], "learning_rate": [0.03, 0.06, 0.1], "max_depth": [4, 6, 8],
                "min_child_weight": [1, 5, 10], "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0], "reg_lambda": [1.0, 3.0, 10.0]},
    "lightgbm": {"n_estimators": [200, 400, 800], "learning_rate": [0.03, 0.06, 0.1], "num_leaves": [15, 31, 63],
                 "min_child_samples": [20, 50, 100], "subsample": [0.7, 0.85, 1.0],
                 "colsample_bytree": [0.7, 0.85, 1.0], "reg_lambda": [1.0, 3.0, 10.0]},
}


def available(name: str) -> bool:
    lib = {"xgboost": "xgboost", "lightgbm": "lightgbm"}.get(name)
    if lib is None:
        return True
    try:
        __import__(lib)
        return True
    except ImportError:
        return False


def build_logistic(random_state: int) -> LogisticRegression:
    return LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=random_state)


def build_hist_gb(random_state: int, categorical: list[str] | None = None, params: dict | None = None):
    p = {**DEFAULTS["hist_gb"], **(params or {})}
    return HistGradientBoostingClassifier(
        max_iter=400, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        class_weight="balanced", categorical_features=categorical or None,
        random_state=random_state, **p)


def build_xgboost(random_state: int, scale_pos_weight: float, params: dict | None = None):
    from xgboost import XGBClassifier
    p = {**DEFAULTS["xgboost"], **(params or {})}
    return XGBClassifier(
        tree_method="hist", objective="binary:logistic", eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,      # XGBoost's form of class weighting
        n_jobs=-1, random_state=random_state, **p)


def build_lightgbm(random_state: int, params: dict | None = None):
    from lightgbm import LGBMClassifier
    p = {**DEFAULTS["lightgbm"], **(params or {})}
    return LGBMClassifier(
        objective="binary", class_weight="balanced", subsample_freq=1,
        n_jobs=-1, random_state=random_state, verbose=-1, **p)


def build_isolation_forest(random_state: int, n_estimators: int, max_samples) -> IsolationForest:
    return IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination="auto",
                           n_jobs=-1, random_state=random_state)


def build(name: str, seed: int, *, categorical: list[str] | None = None, params: dict | None = None,
          scale_pos_weight: float = 1.0, iso_cfg: dict | None = None):
    if name == "logistic":
        return build_logistic(seed)
    if name in ("hist_gb", "hybrid_gb"):
        return build_hist_gb(seed, categorical, params)
    if name == "xgboost":
        return build_xgboost(seed, scale_pos_weight, params)
    if name == "lightgbm":
        return build_lightgbm(seed, params)
    if name == "isolation_forest":
        iso_cfg = iso_cfg or {"n_estimators": 200, "max_samples": 256}
        return build_isolation_forest(seed, iso_cfg["n_estimators"], iso_cfg["max_samples"])
    raise ValueError(f"Unknown model {name!r}")


def fit(model, name: str, X, y, categorical: list[str] | None = None):
    """Fit with library-specific options (LightGBM is told which columns are categorical)."""
    if name == "lightgbm" and categorical:
        model.fit(X, y, categorical_feature=[c for c in categorical if c in X.columns])
    else:
        model.fit(X, y)
    return model


def imbalance_method(name: str, model) -> str:
    if name == "isolation_forest":
        return "none needed: unsupervised, fitted on legitimate transactions only"
    if name == "xgboost":
        return f"scale_pos_weight={getattr(model, 'scale_pos_weight', None):.1f} (class weighting); no resampling"
    return f"class_weight={getattr(model, 'class_weight', None)!r}; no resampling"


def uses_class_weights(name: str, model) -> bool:
    if name == "isolation_forest":
        return True
    if name == "xgboost":
        return (getattr(model, "scale_pos_weight", 1) or 1) > 1
    return getattr(model, "class_weight", None) == "balanced"


def anomaly_score(model: IsolationForest, X) -> np.ndarray:
    """Higher = more anomalous (IsolationForest.score_samples is the other way round)."""
    return -model.score_samples(X)


def positive_scores(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
