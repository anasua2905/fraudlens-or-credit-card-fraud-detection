"""Model zoo for fraud detection.

| Model            | Input view | Role |
|------------------|-----------|------|
| logistic         | linear    | Interpretable baseline everything else must beat |
| hist_gb          | tree      | Main supervised model (gradient boosting) |
| isolation_forest | linear    | Unsupervised anomaly detector, trained on legitimate transactions only |
| hybrid_gb        | tree      | Gradient boosting with the anomaly score as an extra feature |
| mlp              | linear    | Neural network (off by default: slow on 1.2M rows) |

Class imbalance is handled inside training, on the training split only, via
class weights — never by resampling the validation or test data.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

VIEW_BY_MODEL = {
    "logistic": "linear",
    "hist_gb": "tree",
    "isolation_forest": "linear",
    "hybrid_gb": "tree",
    "mlp": "linear",
}
SUPERVISED = ["logistic", "hist_gb", "hybrid_gb", "mlp"]
CATEGORICAL_IN_TREE_VIEW = ["category", "gender"]


def build_logistic(random_state: int) -> LogisticRegression:
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",   # compensates for 1 fraud per ~172 legitimate rows
        solver="lbfgs",
        random_state=random_state,
    )


def build_hist_gb(random_state: int, categorical: list[str] | None = None) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=True,        # holds out part of TRAIN to stop at the right depth
        validation_fraction=0.1,
        n_iter_no_change=20,
        class_weight="balanced",
        categorical_features=categorical or None,
        random_state=random_state,
    )


def build_isolation_forest(random_state: int, n_estimators: int, max_samples) -> IsolationForest:
    return IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination="auto",
        n_jobs=-1,
        random_state=random_state,
    )


def build_mlp(random_state: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-4,
        batch_size=1024,
        learning_rate_init=1e-3,
        max_iter=60,
        early_stopping=True,
        n_iter_no_change=5,
        random_state=random_state,
    )


def anomaly_score(model: IsolationForest, X) -> np.ndarray:
    """Higher = more anomalous (IsolationForest.score_samples is the other way round)."""
    return -model.score_samples(X)


def positive_scores(model, X) -> np.ndarray:
    """Fraud score in [0, 1] for supervised models."""
    return model.predict_proba(X)[:, 1]
