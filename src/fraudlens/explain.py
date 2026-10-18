"""Lifecycle stage 6 - Interpretation with SHAP.

Global view: which features drive the champion's scores across the test period
(mean |SHAP| and a beeswarm). Local view: why individual alerts were raised
(waterfall plots), plus the top contributions for every test-period alert,
which the dashboard shows next to each alert.

SHAP values are in log-odds units: positive values push a transaction towards
"fraud", negative values towards "legitimate".

    python -m fraudlens.explain
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .datasets import load_features
from .scoring import FraudModel
from .utils import get_logger, load_config, write_json, write_table

log = get_logger()


def log_odds(model, X) -> np.ndarray:
    """The champion's output on the log-odds scale (SHAP's natural unit for classifiers)."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    p = np.clip(model.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


class Explainer:
    """SHAP explanations for the champion, in log-odds units.

    TreeSHAP (exact and fast) is used when its values add up to the model's
    own output on check rows that include high-scoring transactions. Some
    model types are not fully supported by shap's tree parser (for example
    scikit-learn's HistGradientBoosting with native categorical splits); for
    those the check fails and a permutation explainer on the same log-odds
    output is used instead, which is model-agnostic and additive by design.
    """

    def __init__(self, fm: FraudModel, background: pd.DataFrame | None = None, permutations: int = 3):
        import shap
        self.shap = shap
        self.fm = fm
        self.feature_names: list[str] | None = None
        self.additivity_error = None
        self.kind = None
        try:
            tree = shap.TreeExplainer(fm.model)
            if background is not None:
                sv = tree(background)
                vals = np.asarray(sv.values)
                vals = vals[:, :, 1] if vals.ndim == 3 else vals
                base = np.asarray(sv.base_values)
                base = base[:, 1] if base.ndim == 2 else base
                err = float(np.max(np.abs(vals.sum(axis=1) + base - log_odds(fm.model, background))))
                self.additivity_error = round(err, 6)
                if err > 1e-3:
                    raise ValueError(f"TreeSHAP values do not add up to the model output (max error {err:.3f})")
            self.explainer, self.kind = tree, "tree"
            self.base_value = float(np.ravel(tree.expected_value)[-1])
        except Exception as exc:
            if background is None:
                raise
            log.info("Using a permutation explainer (%s)", exc)
            bg = background.sample(min(len(background), 50), random_state=0)
            masker = shap.maskers.Independent(bg, max_samples=len(bg))
            self.explainer = shap.PermutationExplainer(lambda X: log_odds(fm.model, pd.DataFrame(X, columns=bg.columns)
                                                                          .astype(bg.dtypes.to_dict())),
                                                       masker, seed=0)
            self.max_evals = 2 * permutations * bg.shape[1] + 1
            self.kind = "permutation"
            self.base_value = float(log_odds(fm.model, bg).mean())

    def values(self, X: pd.DataFrame) -> np.ndarray:
        self.feature_names = list(X.columns)
        sv = self.explainer(X, max_evals=self.max_evals, silent=True) if self.kind == "permutation" else self.explainer(X)
        vals = np.asarray(sv.values)
        if vals.ndim == 3:            # (rows, features, classes) -> positive class
            vals = vals[:, :, 1]
        return vals

    def top_contributions(self, X: pd.DataFrame, k: int = 3) -> list[list[dict]]:
        vals = self.values(X)
        out = []
        for i in range(len(X)):
            order = np.argsort(-np.abs(vals[i]))[:k]
            out.append([{"feature": self.feature_names[j], "value": _num(X.iloc[i, j]),
                         "shap": round(float(vals[i, j]), 4)} for j in order])
        return out


def _num(v):
    try:
        f = float(v)
        return None if np.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return str(v)


def run_explain(cfg: dict) -> dict:
    import matplotlib.pyplot as plt
    import shap

    from . import viz          # plotting stack only for the batch stage; serving needs just shap

    figs = cfg["paths"]["figures"]
    fm = FraudModel.load(cfg["paths"]["artifacts"] / "fraudlens_model.joblib")
    test = load_features(cfg, "test")
    X_all = fm.transform(test)
    scores = fm.predict(test)
    alerts = np.flatnonzero(scores >= fm.threshold)

    n = cfg.get("evaluation", {}).get("shap_rows", 5000)
    rng = np.random.default_rng(cfg["model"]["random_state"])
    fraud_idx = np.flatnonzero(test["is_fraud"].to_numpy() == 1)
    legit_idx = np.flatnonzero(test["is_fraud"].to_numpy() == 0)
    n_fraud = min(len(fraud_idx), n // 5)
    sample = np.concatenate([rng.choice(fraud_idx, n_fraud, replace=False),
                             rng.choice(legit_idx, min(n - n_fraud, len(legit_idx)), replace=False)])
    top = np.argsort(-scores)[:50]
    background = X_all.iloc[np.unique(np.concatenate([rng.choice(len(X_all), 200, replace=False), top]))]
    background.to_parquet(cfg["paths"]["artifacts"] / "shap_background.parquet")
    ex = Explainer(fm, background)
    log.info("SHAP explainer: %s (TreeSHAP additivity error %s)", ex.kind, ex.additivity_error)
    Xs = X_all.iloc[sample]
    sv = ex.values(Xs)
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=Xs.columns).sort_values(ascending=False)

    # Global importance (bar) and beeswarm
    top = mean_abs.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(top.index, top.values, color=viz.SERIES[0])
    ax.set(title="Mean |SHAP value| on the test period", xlabel="Average impact on the fraud log-odds")
    fig.tight_layout()
    viz.save(fig, figs / "16_shap_importance")

    explanation = shap.Explanation(values=sv, data=Xs.to_numpy(dtype=float), feature_names=list(Xs.columns))
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    fig.axes[0].set_title("SHAP beeswarm: how each feature's value moves the score", loc="left")
    viz.save(fig, figs / "17_shap_beeswarm")

    # Local explanations: one caught fraud and one false alert
    y = test["is_fraud"].to_numpy()
    local = {}
    for label, cond in (("caught_fraud", y[alerts] == 1), ("false_alert", y[alerts] == 0)):
        idx = alerts[cond]
        if len(idx) == 0:
            continue
        i = int(idx[np.argmax(scores[idx])])
        row = X_all.iloc[[i]]
        v = ex.values(row)[0]
        base = ex.base_value
        e1 = shap.Explanation(values=v, base_values=base, data=row.to_numpy(dtype=float)[0],
                              feature_names=list(row.columns))
        plt.figure()
        shap.plots.waterfall(e1, max_display=10, show=False)
        fig = plt.gcf()
        fig.set_size_inches(8, 5)
        fig.axes[0].set_title(f"Why this {label.replace('_', ' ')} was flagged "
                              f"(score {scores[i]:.3f}, amount ${test['amount'].iloc[i]:,.2f})", loc="left")
        viz.save(fig, figs / f"18_shap_waterfall_{label}")
        local[label] = {"transaction_id": test["transaction_id"].iloc[i], "score": round(float(scores[i]), 4),
                        "top": [{"feature": f, "shap": round(float(s), 4)}
                                for f, s in sorted(zip(row.columns, v), key=lambda t: -abs(t[1]))[:5]]}

    # Top-3 contributions for every alert (shown in the dashboard)
    contrib = ex.top_contributions(X_all.iloc[alerts], k=3)
    rows = [{"transaction_id": test["transaction_id"].iloc[i],
             **{f"reason_{r + 1}": f"{c['feature']} ({c['shap']:+.2f})" for r, c in enumerate(cs)}}
            for i, cs in zip(alerts, contrib)]
    write_table(pd.DataFrame(rows), cfg["paths"]["gold"] / "alert_explanations", cfg["storage"]["format"])

    report = {"explainer": ex.kind, "treeshap_additivity_error": ex.additivity_error, "rows_explained": int(len(sample)), "alerts_explained": int(len(alerts)),
              "mean_abs_shap": mean_abs.round(5).to_dict(), "local_examples": local,
              "note": "SHAP values are in log-odds units; positive pushes towards fraud."}
    write_json(report, cfg["paths"]["reports"] / "explanations.json")
    log.info("SHAP explanations: %s rows, %s alerts", len(sample), len(alerts))
    return report


if __name__ == "__main__":
    run_explain(load_config())
