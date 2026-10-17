"""Champion selection: a written, reproducible rule.

Rule (a non-inferiority test)
-----------------------------
1. Rank supervised models by validation PR-AUC.
2. For every other model, estimate the PR-AUC gap to the best model with a
   paired bootstrap on the validation split (same resampled rows for both).
3. A model is "non-inferior" to the best if the upper end of the 95% interval
   of that gap is below a margin (config `model.selection_margin`, default
   0.01 PR-AUC). This demands positive evidence that the model is close; a
   wide interval from scarce data is NOT treated as a tie.
4. Among non-inferior models, choose the SIMPLEST one (config
   `model.complexity_order`). Extra complexity has to buy measurable
   performance; a tie goes to the model that is cheaper to run and explain.

Only validation data is used. The test split plays no part in selection.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from . import models as M

# Simplest first. The hybrid needs an Isolation Forest scored before every
# prediction, so it ranks above standard gradient boosting.
DEFAULT_COMPLEXITY_ORDER = ["logistic", "hist_gb", "mlp", "hybrid_gb"]


def paired_bootstrap_gap(y: np.ndarray, best: np.ndarray, other: np.ndarray,
                         n_boot: int = 500, seed: int = 0, margin: float = 0.01) -> dict:
    """Bootstrap distribution of PR-AUC(best) - PR-AUC(other) on the same resamples."""
    rng = np.random.default_rng(seed)
    n = len(y)
    gaps = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        gaps.append(average_precision_score(yy, best[idx]) - average_precision_score(yy, other[idx]))
    gaps = np.asarray(gaps)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return {
        "gap_mean": round(float(gaps.mean()), 5),
        "ci95_low": round(float(lo), 5),
        "ci95_high": round(float(hi), 5),
        "equivalent_to_best": bool(hi <= margin),  # non-inferior within the margin
        "n_boot": int(len(gaps)),
    }


def select_champion(y_val: np.ndarray, val_scores: dict[str, np.ndarray], val_pr_auc: dict[str, float],
                    complexity_order: list[str] | None = None, n_boot: int = 500, seed: int = 0,
                    margin: float = 0.01) -> dict:
    order = complexity_order or DEFAULT_COMPLEXITY_ORDER
    candidates = [m for m in val_scores if m in M.SUPERVISED]
    if not candidates:
        raise ValueError("No supervised model available for selection")
    best = max(candidates, key=lambda m: val_pr_auc[m])

    comparisons = {best: {"gap_mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0,
                          "equivalent_to_best": True, "n_boot": 0}}
    for m in candidates:
        if m != best:
            comparisons[m] = paired_bootstrap_gap(y_val, val_scores[best], val_scores[m], n_boot, seed, margin)

    equivalent = [m for m in candidates if comparisons[m]["equivalent_to_best"]]
    ranked = sorted(equivalent, key=lambda m: order.index(m) if m in order else len(order))
    champion = ranked[0]

    if champion == best:
        reason = (f"{best} has the highest validation PR-AUC ({val_pr_auc[best]:.4f}) and no simpler "
                  f"model is non-inferior to it within a margin of {margin} PR-AUC.")
    else:
        c = comparisons[champion]
        reason = (f"{best} has the highest validation PR-AUC ({val_pr_auc[best]:.4f}), but {champion} "
                  f"({val_pr_auc[champion]:.4f}) is non-inferior: the 95% bootstrap interval of the gap is "
                  f"[{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}], entirely below the {margin} margin. "
                  f"{champion} is simpler, so it is selected.")

    return {
        "rule": "simplest supervised model that is non-inferior to the best on validation PR-AUC "
                "(paired bootstrap; upper 95% bound of the gap below the margin)",
        "margin": margin,
        "complexity_order": order,
        "best_by_validation_pr_auc": best,
        "champion": champion,
        "reason": reason,
        "comparisons": comparisons,
    }
