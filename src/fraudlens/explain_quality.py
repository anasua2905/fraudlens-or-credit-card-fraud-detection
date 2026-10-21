"""Do the explanations explain the model? Faithfulness checks for the per-alert SHAP reasons.

An analyst reads "top 3 reasons" next to every alert, so they must reflect what the model
actually used. Three tests on a sample of test-period alerts:

  1. Deletion fidelity: replace an alert's top-3 SHAP features with typical values (the
     background median, or mode for categories) and re-score. If the explanation is faithful,
     the score falls far more than when 3 *random* features are replaced, and many alerts clear.
  2. Additivity: base value + sum of SHAP values should reproduce the model's log-odds output
     (exact for TreeSHAP, approximate for the permutation explainer); the error is reported.
  3. Stability: explaining the same alert twice (different permutation seeds) should give the
     same top-3 features (Jaccard overlap).

    python -m fraudlens.explain_quality      # writes reports/explanation_quality.json
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import features as F
from .datasets import load_features
from .explain import Explainer, log_odds
from .scoring import FraudModel
from .utils import get_logger, load_config, read_table, write_json

log = get_logger()


def typical_values(background: pd.DataFrame) -> dict:
    out = {}
    for c in background.columns:
        s = background[c]
        out[c] = s.mode().iloc[0] if (c in F.CATEGORICAL_FEATURES or s.dtype.kind in "iub" and s.nunique() < 20) \
            else s.median()
    return out


def deletion_scores(fm: FraudModel, X: pd.DataFrame, cols_per_row: list[list[str]], typical: dict) -> np.ndarray:
    Xd = X.copy()
    for i, cols in enumerate(cols_per_row):
        for c in cols:
            Xd.iloc[i, Xd.columns.get_loc(c)] = typical[c]
    Xd = Xd.astype(X.dtypes.to_dict())
    p = np.clip(1 / (1 + np.exp(-log_odds(fm.model, Xd))), 0, 1)
    return p


def run_explain_quality(cfg: dict, n_alerts: int = 200, k: int = 3) -> dict:
    P = cfg["paths"]
    rng = np.random.default_rng(cfg["model"]["random_state"])
    fm = FraudModel.load(P["artifacts"] / "fraudlens_model.joblib")
    background = read_table(P["artifacts"] / "shap_background")
    test = load_features(cfg, "test")
    scores = fm.predict(test)
    idx = np.flatnonzero(scores >= fm.threshold)
    idx = rng.choice(idx, size=min(n_alerts, len(idx)), replace=False)
    sample = test.iloc[idx]
    X = fm.transform(sample)
    base_scores = scores[idx]

    exp = Explainer(fm, background, permutations=3)
    vals = exp.values(X)
    names = list(X.columns)
    top = [[names[j] for j in np.argsort(-np.abs(v))[:k]] for v in vals]
    typical = typical_values(background[names] if set(names) <= set(background.columns) else X)

    after_top = deletion_scores(fm, X, top, typical)
    rand_drops = []
    for _ in range(3):
        rnd = [list(rng.choice(names, size=k, replace=False)) for _ in range(len(X))]
        rand_drops.append(base_scores - deletion_scores(fm, X, rnd, typical))
    drop_top = base_scores - after_top
    drop_rand = np.mean(rand_drops, axis=0)

    lo = log_odds(fm.model, X)
    additivity = np.abs(exp.base_value + vals.sum(axis=1) - lo)

    vals2 = Explainer(fm, background, permutations=3, seed=1).values(X)      # a second, independent run
    top2 = [[names[j] for j in np.argsort(-np.abs(v))[:k]] for v in vals2]
    jacc = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(top, top2)]

    counts = pd.Series([c for t in top for c in t]).value_counts()
    out = {
        "explainer": exp.kind, "alerts_sampled": int(len(X)), "top_k": k, "threshold": fm.threshold,
        "deletion": {
            "mean_score_drop_top_k": round(float(drop_top.mean()), 4),
            "mean_score_drop_random_k": round(float(drop_rand.mean()), 4),
            "ratio_top_vs_random": round(float(drop_top.mean() / max(drop_rand.mean(), 1e-9)), 1),
            "share_cleared_top_k": round(float(np.mean(after_top < fm.threshold)), 4),
            "share_cleared_random_k": round(float(np.mean(base_scores - drop_rand < fm.threshold)), 4),
        },
        "additivity": {"median_abs_error_log_odds": round(float(np.median(additivity)), 4),
                       "p95_abs_error_log_odds": round(float(np.quantile(additivity, 0.95)), 4)},
        "stability": {"mean_top_k_jaccard": round(float(np.mean(jacc)), 4),
                      "share_identical_top_k": round(float(np.mean([j == 1.0 for j in jacc])), 4)},
        "most_frequent_top_features": {str(i): int(v) for i, v in counts.head(8).items()},
    }
    d = out["deletion"]
    out["verdict"] = ("faithful" if d["ratio_top_vs_random"] >= 2 and d["share_cleared_top_k"] > d["share_cleared_random_k"]
                      else "weak")
    write_json(out, P["reports"] / "explanation_quality.json")
    log.info("Explanation quality: top-%d deletion drops the score %.3f vs %.3f for random features (%.1fx); "
             "%.0f%% of alerts clear; top-%d stability %.2f", k, d["mean_score_drop_top_k"], d["mean_score_drop_random_k"],
             d["ratio_top_vs_random"], 100 * d["share_cleared_top_k"], k, out["stability"]["mean_top_k_jaccard"])
    return out


if __name__ == "__main__":
    run_explain_quality(load_config())
