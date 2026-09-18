"""Train, evaluate and compare fraud models.

Protocol (the part that keeps results honest):
  * Models are fitted on TRAIN only.
  * Model choice and operating thresholds are decided on VALIDATION.
  * TEST is scored once at the end, with thresholds already fixed.

Run:  python -m fraud_pipeline.train
"""
from __future__ import annotations

import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import evaluate as E
from . import models as M
from .datasets import load_analysis_frame, load_features
from .eda import FRAUD_COLOR, LEGIT_COLOR, save_figure
from .utils import get_logger, load_config, read_table, write_json

log = get_logger()
SPLITS = ["train", "val", "test"]


# ------------------------------------------------------------------ loading
def load_view(cfg: dict, view: str) -> dict:
    """Model matrices plus the amount and timestamp needed for business metrics."""
    out = {}
    for split in SPLITS:
        X = read_table(cfg["paths"]["processed"] / f"{view}_{split}")
        # `amount` may already exist as a feature, so the metadata columns are renamed
        meta = (load_features(cfg, split)[["transaction_id", "ts", "amount"]]
                .rename(columns={"ts": "_ts", "amount": "_amount"}))
        X = X.merge(meta, on="transaction_id", how="left", validate="one_to_one")
        y = X.pop("is_fraud").to_numpy()
        ts, amount = X.pop("_ts"), X.pop("_amount").to_numpy()
        X = X.drop(columns=["transaction_id"])
        for c in M.CATEGORICAL_IN_TREE_VIEW:
            if c in X.columns:  # ordinal codes stored as float; HistGB needs integers
                X[c] = X[c].astype("int32")
        out[split] = {"X": X, "y": y, "amount": amount,
                      "days": max((ts.max() - ts.min()).total_seconds() / 86400, 1)}
    return out


def subsample(data: dict, max_rows: int | None, seed: int, keep_all_fraud: bool = True):
    """Stratified subsample of the training split (for quick runs and the neural net)."""
    X, y = data["X"], data["y"]
    if max_rows is None or len(X) <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    fraud_idx = np.flatnonzero(y == 1)
    legit_idx = np.flatnonzero(y == 0)
    n_legit = max(max_rows - len(fraud_idx), 1) if keep_all_fraud else int(max_rows * (1 - y.mean()))
    keep = np.concatenate([fraud_idx, rng.choice(legit_idx, min(n_legit, len(legit_idx)), replace=False)])
    keep.sort()
    return X.iloc[keep], y[keep]


# ------------------------------------------------------------------ training
def train_models(cfg: dict, views: dict) -> dict:
    mcfg = cfg["model"]
    seed = mcfg["random_state"]
    wanted = list(mcfg["models"])
    if "hybrid_gb" in wanted and "isolation_forest" not in wanted:
        wanted.insert(wanted.index("hybrid_gb"), "isolation_forest")  # hybrid needs its score

    fitted: dict = {}
    scores: dict = {}

    for name in wanted:
        view = views[M.VIEW_BY_MODEL[name]]
        t0 = time.perf_counter()

        if name == "isolation_forest":
            legit = view["train"]["y"] == 0
            icfg = mcfg["isolation_forest"]
            model = M.build_isolation_forest(seed, icfg["n_estimators"], icfg["max_samples"])
            model.fit(view["train"]["X"].loc[legit])  # unsupervised: legitimate behaviour only
            scores[name] = {s: M.anomaly_score(model, view[s]["X"]) for s in SPLITS}

        elif name == "hybrid_gb":
            tree = views["tree"]
            Xs = {s: tree[s]["X"].assign(anomaly_score=scores["isolation_forest"][s]) for s in SPLITS}
            X_tr, y_tr = subsample({"X": Xs["train"], "y": tree["train"]["y"]}, mcfg["max_train_rows"], seed)
            model = M.build_hist_gb(seed, _categorical(X_tr))
            model.fit(X_tr, y_tr)
            scores[name] = {s: M.positive_scores(model, Xs[s]) for s in SPLITS}

        else:
            max_rows = mcfg["mlp_max_train_rows"] if name == "mlp" else mcfg["max_train_rows"]
            X_tr, y_tr = subsample(view["train"], max_rows, seed)
            if name == "logistic":
                model = M.build_logistic(seed)
            elif name == "hist_gb":
                model = M.build_hist_gb(seed, _categorical(X_tr))
            elif name == "mlp":
                # MLPClassifier has no class_weight, so fraud is oversampled in its
                # training sample only (never in validation or test).
                X_tr, y_tr = _oversample(X_tr, y_tr, ratio=10, seed=seed)
                model = M.build_mlp(seed)
            else:
                raise ValueError(f"Unknown model {name!r}")
            model.fit(X_tr, y_tr)
            scores[name] = {s: M.positive_scores(model, view[s]["X"]) for s in SPLITS}

        fitted[name] = model
        joblib.dump(model, cfg["paths"]["artifacts"] / f"model_{name}.joblib")
        log.info("Trained %-17s in %5.1fs", name, time.perf_counter() - t0)

    return {"fitted": fitted, "scores": scores}


def _categorical(X: pd.DataFrame) -> list[str]:
    return [c for c in M.CATEGORICAL_IN_TREE_VIEW if c in X.columns]


def _oversample(X: pd.DataFrame, y: np.ndarray, ratio: int, seed: int):
    """Repeat fraud rows until there is 1 fraud per `ratio` legitimate rows."""
    rng = np.random.default_rng(seed)
    fraud_idx = np.flatnonzero(y == 1)
    target = len(y) // ratio
    if len(fraud_idx) >= target or len(fraud_idx) == 0:
        return X, y
    extra = rng.choice(fraud_idx, target - len(fraud_idx), replace=True)
    idx = np.concatenate([np.arange(len(y)), extra])
    rng.shuffle(idx)
    return X.iloc[idx], y[idx]


# ---------------------------------------------------------------- reporting
def evaluate_models(cfg: dict, views: dict, scores: dict) -> dict:
    tcfg = cfg["model"]["threshold"]
    strategies = ["max_f1", "alert_budget", "min_cost"]
    results: dict = {}

    for name, sc in scores.items():
        view = views[M.VIEW_BY_MODEL[name]]
        val, test = view["val"], view["test"]
        thresholds = {
            s: E.choose_threshold(val["y"], sc["val"], val["amount"], val["days"], tcfg, s)
            for s in strategies
        }
        primary = thresholds[tcfg["primary"]]
        results[name] = {
            "view": M.VIEW_BY_MODEL[name],
            "thresholds_from_validation": {k: round(v, 6) for k, v in thresholds.items()},
            "validation": E.evaluate(val["y"], sc["val"], val["amount"], primary, val["days"], tcfg),
            "test": E.evaluate(test["y"], sc["test"], test["amount"], primary, test["days"], tcfg),
            "test_at_other_thresholds": {
                s: E.evaluate(test["y"], sc["test"], test["amount"], thresholds[s], test["days"], tcfg)
                for s in strategies if s != tcfg["primary"]
            },
        }
    return results


def comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        rows.append({
            "model": name,
            "val_pr_auc": r["validation"]["pr_auc"],
            "test_pr_auc": r["test"]["pr_auc"],
            "test_roc_auc": r["test"]["roc_auc"],
            "test_precision": r["test"]["precision"],
            "test_recall": r["test"]["recall"],
            "test_alerts_per_day": r["test"]["alerts_per_day"],
            "test_false_alarms_per_catch": r["test"]["false_alarms_per_catch"],
            "test_net_benefit": r["test"]["net_benefit"],
        })
    return pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).set_index("model")


# ------------------------------------------------------------------ figures
def plot_pr_curves(views: dict, scores: dict, split: str = "test"):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, sc in scores.items():
        d = views[M.VIEW_BY_MODEL[name]][split]
        p, r = E.pr_curve(d["y"], sc[split])
        ax.plot(r, p, label=f"{name} (PR-AUC {E.ranking_metrics(d['y'], sc[split])['pr_auc']:.3f})")
    any_view = views[M.VIEW_BY_MODEL[next(iter(scores))]]
    base = float(np.mean(any_view[split]["y"]))
    ax.axhline(base, ls="--", lw=1, color="grey")
    ax.text(0.02, base, f" random = {base:.3%}", va="bottom", fontsize=8, color="grey")
    ax.set(title=f"Precision-recall ({split})", xlabel="Recall", ylabel="Precision", ylim=(0, 1.02))
    ax.legend(fontsize=8)
    return fig


def plot_threshold_sweep(view_split: dict, sc: np.ndarray, tcfg: dict, chosen: float, model_name: str):
    sweep = E.threshold_sweep(view_split["y"], sc, view_split["amount"], view_split["days"], tcfg)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(sweep["alerts_per_day"], sweep["precision"], label="Precision", color=FRAUD_COLOR)
    axes[0].plot(sweep["alerts_per_day"], sweep["recall"], label="Recall", color=LEGIT_COLOR)
    axes[0].set(title=f"{model_name}: precision vs recall by alert volume",
                xlabel="Alerts per day", ylabel="Rate", xscale="log")
    axes[0].legend()
    axes[1].plot(sweep["alerts_per_day"], sweep["net_benefit"], color=FRAUD_COLOR)
    axes[1].axhline(0, ls="--", lw=1, color="grey")
    chosen_row = E.business_metrics(view_split["y"], sc, view_split["amount"], chosen,
                                    view_split["days"], tcfg["investigation_cost"], tcfg["recovery_rate"])
    axes[1].axvline(chosen_row["alerts_per_day"], ls=":", color="black",
                    label=f"chosen: {chosen_row['alerts_per_day']:.0f}/day")
    axes[1].set(title="Net benefit by alert volume", xlabel="Alerts per day",
                ylabel="Net benefit", xscale="log")
    axes[1].legend()
    fig.tight_layout()
    return fig, sweep


def plot_score_distribution(y, scores, threshold, model_name):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bins = np.linspace(float(np.min(scores)), float(np.max(scores)), 60)
    ax.hist(scores[y == 0], bins=bins, density=True, alpha=0.55, color=LEGIT_COLOR, label="Legit")
    ax.hist(scores[y == 1], bins=bins, density=True, alpha=0.55, color=FRAUD_COLOR, label="Fraud")
    ax.axvline(threshold, ls="--", color="black", lw=1, label="threshold")
    ax.set(title=f"{model_name}: score distribution (test)", xlabel="Fraud score", ylabel="Density", yscale="log")
    ax.legend(fontsize=8)
    return fig


def plot_importance(model, X, y, n_rows: int, seed: int, model_name: str):
    idx = np.random.default_rng(seed).choice(len(X), min(n_rows, len(X)), replace=False)
    imp = permutation_importance(model, X.iloc[idx], y[idx], scoring="average_precision",
                                 n_repeats=3, random_state=seed, n_jobs=-1)
    order = np.argsort(imp.importances_mean)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(np.array(X.columns)[order], imp.importances_mean[order],
            xerr=imp.importances_std[order], color=FRAUD_COLOR)
    ax.set(title=f"{model_name}: permutation importance (drop in PR-AUC)", xlabel="Importance")
    fig.tight_layout()
    table = pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean,
                          "std": imp.importances_std}).sort_values("importance", ascending=False)
    return fig, table.round(5)


# ---------------------------------------------------------- error analysis
AMOUNT_BANDS = [0, 10, 50, 100, 250, 500, 1000, np.inf]
AMOUNT_LABELS = ["<$10", "$10-50", "$50-100", "$100-250", "$250-500", "$500-1k", "$1k+"]


def score_split(cfg: dict, model_name: str, split: str) -> pd.DataFrame:
    """Score one split with a saved model. Returns transaction_id + score."""
    view = M.VIEW_BY_MODEL[model_name]
    X = read_table(cfg["paths"]["processed"] / f"{view}_{split}")
    ids = X["transaction_id"]
    Xf = X.drop(columns=["transaction_id", "is_fraud"])
    for c in M.CATEGORICAL_IN_TREE_VIEW:
        if c in Xf.columns:
            Xf[c] = Xf[c].astype("int32")
    if model_name == "hybrid_gb":
        lin = read_table(cfg["paths"]["processed"] / f"linear_{split}").drop(columns=["transaction_id", "is_fraud"])
        iso = joblib.load(cfg["paths"]["artifacts"] / "model_isolation_forest.joblib")
        Xf = Xf.assign(anomaly_score=M.anomaly_score(iso, lin))
    model = joblib.load(cfg["paths"]["artifacts"] / f"model_{model_name}.joblib")
    score = M.anomaly_score(model, Xf) if model_name == "isolation_forest" else M.positive_scores(model, Xf)
    return pd.DataFrame({"transaction_id": ids, "score": score})


def error_breakdown(cfg: dict, model_name: str, split: str, threshold: float) -> dict[str, pd.DataFrame]:
    """Where the model still fails: missed fraud and false alerts, by segment."""
    df = load_analysis_frame(cfg, split).merge(score_split(cfg, model_name, split), on="transaction_id")
    df["alert"] = df["score"] >= threshold
    df["band"] = pd.cut(df["amount"], AMOUNT_BANDS, labels=AMOUNT_LABELS)
    df["outcome"] = np.where(df["alert"] & (df["is_fraud"] == 1), "caught",
                     np.where(df["alert"], "false_alert",
                      np.where(df["is_fraud"] == 1, "missed", "correctly_ignored")))

    def _segment(by):
        g = df.groupby(by, observed=True)
        t = pd.DataFrame({
            "transactions": g.size(),
            "frauds": g["is_fraud"].sum(),
            "alerts": g["alert"].sum(),
            "caught": g.apply(lambda d: int(((d["outcome"] == "caught")).sum()), include_groups=False),
            "missed": g.apply(lambda d: int(((d["outcome"] == "missed")).sum()), include_groups=False),
            "false_alerts": g.apply(lambda d: int(((d["outcome"] == "false_alert")).sum()), include_groups=False),
        })
        t["precision"] = (t["caught"] / t["alerts"]).round(3)
        t["recall"] = (t["caught"] / t["frauds"]).round(3)
        return t

    missed = df[df["outcome"] == "missed"]
    false_alerts = df[df["outcome"] == "false_alert"]
    return {
        "by_category": _segment("category").sort_values("false_alerts", ascending=False),
        "by_amount_band": _segment("band"),
        "by_hour": _segment("hour"),
        "missed_fraud_summary": pd.DataFrame({
            "count": [len(missed)],
            "amount_total": [round(missed["amount"].sum(), 2)],
            "amount_median": [round(missed["amount"].median(), 2) if len(missed) else None],
            "share_at_night_pct": [round(100 * missed["is_night"].mean(), 1) if len(missed) else None],
        }),
        "false_alert_summary": pd.DataFrame({
            "count": [len(false_alerts)],
            "amount_median": [round(false_alerts["amount"].median(), 2) if len(false_alerts) else None],
            "share_at_night_pct": [round(100 * false_alerts["is_night"].mean(), 1) if len(false_alerts) else None],
            "cards_affected": [int(false_alerts["card_id"].nunique())],
        }),
    }


# ------------------------------------------------------------------- runner
def run_training(cfg: dict) -> dict:
    t0 = time.perf_counter()
    mcfg, tcfg = cfg["model"], cfg["model"]["threshold"]
    needed = {M.VIEW_BY_MODEL[m] for m in mcfg["models"]} | ({"tree", "linear"} if "hybrid_gb" in mcfg["models"] else set())
    views = {v: load_view(cfg, v) for v in needed}
    log.info("Loaded views %s | train rows: %s", sorted(needed), f"{len(views[list(needed)[0]]['train']['X']):,}")

    trained = train_models(cfg, views)
    results = evaluate_models(cfg, views, trained["scores"])
    table = comparison_table(results)

    supervised = [m for m in table.index if m in M.SUPERVISED]
    champion = supervised[0] if supervised else table.index[0]
    log.info("Champion by validation PR-AUC: %s", champion)

    # Figures
    save_figure(plot_pr_curves(views, trained["scores"], "test"), cfg, "10_pr_curves_test")
    champ_view = views[M.VIEW_BY_MODEL[champion]]
    champ_thr = results[champion]["thresholds_from_validation"][tcfg["primary"]]
    fig, sweep = plot_threshold_sweep(champ_view["val"], trained["scores"][champion]["val"],
                                      tcfg, champ_thr, champion)
    save_figure(fig, cfg, "11_threshold_sweep_validation")
    save_figure(plot_score_distribution(champ_view["test"]["y"], trained["scores"][champion]["test"],
                                        champ_thr, champion), cfg, "12_score_distribution_test")
    importance = None
    if champion in {"hist_gb", "hybrid_gb"}:
        X = champ_view["val"]["X"]
        if champion == "hybrid_gb":
            X = X.assign(anomaly_score=trained["scores"]["isolation_forest"]["val"])
        fig, importance = plot_importance(trained["fitted"][champion], X, champ_view["val"]["y"],
                                          mcfg["permutation_importance_rows"], mcfg["random_state"], champion)
        save_figure(fig, cfg, "13_permutation_importance")
    plt.close("all")

    report = {
        "protocol": "fit on train; model and thresholds chosen on validation; test scored once",
        "champion": champion,
        "threshold_strategy": tcfg["primary"],
        "comparison": table.reset_index().to_dict("records"),
        "results": results,
        "champion_importance": importance.to_dict("records") if importance is not None else None,
        "validation_sweep_head": sweep.head(15).to_dict("records"),
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
    write_json(report, cfg["paths"]["reports"] / "model_results.json")
    log.info("Training finished in %.1fs", report["runtime_seconds"])
    return report


if __name__ == "__main__":
    plt.switch_backend("Agg")
    run_training(load_config())
