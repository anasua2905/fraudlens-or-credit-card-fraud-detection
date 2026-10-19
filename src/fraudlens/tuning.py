"""Hyper-parameter tuning with time-series cross-validation (lifecycle stage 5).

Random K-fold would let a model learn from the future. Here TRAIN is cut into
consecutive time blocks and each fold trains on everything before a block and
validates on that block (an expanding window). The validation and test splits
are never touched.

To keep tuning affordable on 1.2M rows, each fold trains on all fraud plus a
random share of legitimate transactions (`legit_sample_frac`). The PR-AUC
used to compare candidates is rank-based, so sub-sampling legitimate rows
inside the training folds does not bias the comparison. Validation folds are
never sub-sampled.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import models as M
from .utils import get_logger

log = get_logger()


def time_folds(n_rows: int, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds over rows already sorted by time: block i+1 validates on blocks 0..i."""
    edges = np.linspace(0, n_rows, k + 2).astype(int)
    return [(np.arange(0, edges[i + 1]), np.arange(edges[i + 1], edges[i + 2])) for i in range(k)]


def sample_candidates(space: dict, n_iter: int, seed: int) -> list[dict]:
    """Random search without replacement over a discrete grid; the defaults are always candidate 0."""
    rng = np.random.default_rng(seed)
    keys = list(space)
    grid = list(itertools.product(*(space[k] for k in keys)))
    picks = rng.choice(len(grid), size=min(n_iter, len(grid)), replace=False)
    return [dict(zip(keys, grid[i])) for i in picks]


def _legit_subsample(idx: np.ndarray, y: np.ndarray, frac: float, rng) -> np.ndarray:
    if frac >= 1:
        return idx
    yy = y[idx]
    keep = idx[(yy == 1) | (rng.random(len(idx)) < frac)]
    return np.sort(keep)


def tune(name: str, X: pd.DataFrame, y: np.ndarray, cfg: dict, tracker=None) -> dict:
    """Return the best parameters and the CV score of every candidate."""
    mcfg, tcfg = cfg["model"], cfg["model"]["tuning"]
    seed = mcfg["random_state"]
    rng = np.random.default_rng(seed)
    folds = time_folds(len(X), tcfg["cv_folds"])
    candidates = [dict(M.DEFAULTS[name])] + sample_candidates(M.SEARCH_SPACE[name], tcfg["n_iter"], seed)
    categorical = [c for c in M.CATEGORICAL_IN_TREE_VIEW if c in X.columns]
    results = []
    t0 = time.perf_counter()
    for ci, params in enumerate(candidates):
        fold_scores = []
        for tr, va in folds:
            tr = _legit_subsample(tr, y, tcfg["legit_sample_frac"], rng)
            if y[va].sum() == 0 or y[tr].sum() == 0:
                continue
            spw = float((y[tr] == 0).sum() / max(y[tr].sum(), 1))
            model = M.build(name, seed, categorical=categorical, params=params, scale_pos_weight=spw)
            M.fit(model, name, X.iloc[tr], y[tr], categorical)
            fold_scores.append(average_precision_score(y[va], M.positive_scores(model, X.iloc[va])))
        res = {"candidate": ci, "params": params, "cv_pr_auc_mean": float(np.mean(fold_scores)),
               "cv_pr_auc_std": float(np.std(fold_scores)), "fold_scores": [round(s, 5) for s in fold_scores]}
        results.append(res)
        if tracker is not None:
            tracker.log_candidate(name, ci, params, res)
        log.info("Tuning %-9s candidate %d/%d  CV PR-AUC %.4f +/- %.4f", name, ci + 1, len(candidates),
                 res["cv_pr_auc_mean"], res["cv_pr_auc_std"])
    best = max(results, key=lambda r: r["cv_pr_auc_mean"])
    return {"model": name, "best_params": best["params"], "best_cv_pr_auc_mean": best["cv_pr_auc_mean"],
            "best_cv_pr_auc_std": best["cv_pr_auc_std"], "folds": len(folds),
            "fold_val_rows": [int(len(va)) for _, va in folds], "candidates": results,
            "seconds": round(time.perf_counter() - t0, 1)}
