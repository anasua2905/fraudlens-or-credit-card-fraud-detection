"""Lifecycle stage 5 - Modelling: tune, train, compare and select a champion.

Protocol that keeps the results honest:
  * Tuning uses expanding-window time-series cross-validation INSIDE the
    training period (tuning.py).
  * Final models are fitted on TRAIN only.
  * The champion and the alert threshold are chosen on VALIDATION.
  * TEST is scored once, with everything already fixed.

Every run is tracked in MLflow; the champion is registered as `fraudlens-champion`.

    python -m fraudlens.train
"""
from __future__ import annotations

import datetime as dt
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import evaluate as E
from . import models as M
from . import viz
from .datasets import load_features
from .scoring import FraudModel
from .selection import DEFAULT_COMPLEXITY_ORDER, select_champion
from .tracking import Tracker
from .tuning import tune
from .utils import get_logger, load_config, read_table, write_json

log = get_logger()
SPLITS = ["train", "val", "test"]


# ------------------------------------------------------------------ loading
def load_view(cfg: dict, view: str) -> dict:
    """Model matrices plus the timestamp and amount needed for business metrics."""
    out = {}
    for split in SPLITS:
        X = read_table(cfg["paths"]["processed"] / f"{view}_{split}")
        meta = (load_features(cfg, split)[["transaction_id", "ts", "amount"]]
                .rename(columns={"ts": "_ts", "amount": "_amount"}))
        X = X.merge(meta, on="transaction_id", how="left", validate="one_to_one")
        y = X.pop("is_fraud").to_numpy().astype(int)
        ts, amount = X.pop("_ts"), X.pop("_amount").to_numpy()
        ids = X.pop("transaction_id").to_numpy()
        for c in M.CATEGORICAL_IN_TREE_VIEW:
            if c in X.columns:  # ordinal codes are stored as float
                X[c] = X[c].fillna(-1).astype("int32")
        out[split] = {"X": X, "y": y, "amount": amount, "ts": pd.to_datetime(ts).to_numpy(), "ids": ids,
                      "days": max((ts.max() - ts.min()).total_seconds() / 86400, 1)}
    return out


def subsample(X: pd.DataFrame, y: np.ndarray, max_rows: int | None, seed: int):
    """All fraud plus a random sample of legitimate rows (quick trial runs only)."""
    if max_rows is None or len(X) <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    fraud = np.flatnonzero(y == 1)
    legit = rng.choice(np.flatnonzero(y == 0), max(max_rows - len(fraud), 1), replace=False)
    keep = np.sort(np.concatenate([fraud, legit]))
    return X.iloc[keep], y[keep]


# ----------------------------------------------------------------- training
def train_models(cfg: dict, views: dict, tracker: Tracker) -> dict:
    mcfg = cfg["model"]
    seed = mcfg["random_state"]
    wanted = [m for m in mcfg["models"] if M.available(m)]
    skipped = [m for m in mcfg["models"] if m not in wanted]
    if skipped:
        log.warning("Skipping models whose library is not installed: %s", skipped)
    if "hybrid_gb" in wanted and "isolation_forest" not in wanted:
        wanted.insert(wanted.index("hybrid_gb"), "isolation_forest")
    tuning_cfg = mcfg.get("tuning", {})
    fitted, scores, tuning_results, timings = {}, {}, {}, {}

    for name in wanted:
        view = views[M.VIEW_BY_MODEL[name]]
        t0 = time.perf_counter()
        X_tr, y_tr = subsample(view["train"]["X"], view["train"]["y"], mcfg.get("max_train_rows"), seed)
        categorical = [c for c in M.CATEGORICAL_IN_TREE_VIEW if c in X_tr.columns]
        params = None
        if tuning_cfg.get("enabled") and name in tuning_cfg.get("models", []):
            tuning_results[name] = tune(name, X_tr, y_tr, cfg, tracker)
            params = tuning_results[name]["best_params"]

        if name == "isolation_forest":
            model = M.build(name, seed, iso_cfg=mcfg["isolation_forest"])
            model.fit(X_tr.loc[y_tr == 0])        # unsupervised: legitimate behaviour only
            scores[name] = {s: M.anomaly_score(model, view[s]["X"]) for s in SPLITS}
        elif name == "hybrid_gb":
            Xs = {s: views["tree"][s]["X"].assign(anomaly_score=scores["isolation_forest"][s]) for s in SPLITS}
            Xh, yh = subsample(Xs["train"], views["tree"]["train"]["y"], mcfg.get("max_train_rows"), seed)
            model = M.build(name, seed, categorical=categorical, params=params)
            model.fit(Xh, yh)
            scores[name] = {s: M.positive_scores(model, Xs[s]) for s in SPLITS}
        else:
            spw = float((y_tr == 0).sum() / max(y_tr.sum(), 1))
            model = M.build(name, seed, categorical=categorical, params=params, scale_pos_weight=spw)
            M.fit(model, name, X_tr, y_tr, categorical)
            scores[name] = {s: M.positive_scores(model, view[s]["X"]) for s in SPLITS}

        fitted[name] = model
        joblib.dump(model, cfg["paths"]["artifacts"] / f"model_{name}.joblib", compress=3)
        timings[name] = round(time.perf_counter() - t0, 1)
        log.info("Trained %-17s in %6.1fs", name, timings[name])
    return {"fitted": fitted, "scores": scores, "tuning": tuning_results, "timings": timings, "skipped": skipped}


# --------------------------------------------------------------- evaluation
def evaluate_models(cfg: dict, views: dict, scores: dict) -> dict:
    tcfg = cfg["model"]["threshold"]
    strategies = ["max_f1", "alert_budget", "min_cost"]
    results = {}
    for name, sc in scores.items():
        v = views[M.VIEW_BY_MODEL[name]]
        val, test = v["val"], v["test"]
        thr = {s: E.choose_threshold(val["y"], sc["val"], val["amount"], val["days"], tcfg, s) for s in strategies}
        primary = thr[tcfg["primary"]]
        results[name] = {
            "view": M.VIEW_BY_MODEL[name],
            "library": M.LIBRARY[name],
            "thresholds_from_validation": {k: round(t, 6) for k, t in thr.items()},
            "validation": E.evaluate(val["y"], sc["val"], val["amount"], primary, val["days"], tcfg),
            "test": E.evaluate(test["y"], sc["test"], test["amount"], primary, test["days"], tcfg),
            "test_at_other_thresholds": {
                s: E.evaluate(test["y"], sc["test"], test["amount"], thr[s], test["days"], tcfg)
                for s in strategies if s != tcfg["primary"]},
        }
    return results


def comparison_table(results: dict, tuning: dict) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        t = tuning.get(name, {})
        rows.append({
            "model": name, "library": r["library"],
            "cv_pr_auc": round(t["best_cv_pr_auc_mean"], 4) if t else None,
            "val_pr_auc": r["validation"]["pr_auc"], "test_pr_auc": r["test"]["pr_auc"],
            "test_roc_auc": r["test"]["roc_auc"], "test_precision": r["test"]["precision"],
            "test_recall": r["test"]["recall"], "test_alerts_per_day": r["test"]["alerts_per_day"],
            "test_false_alarms_per_catch": r["test"]["false_alarms_per_catch"],
            "test_net_benefit": r["test"]["net_benefit"],
        })
    return pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).set_index("model")


# ------------------------------------------------------------------ figures
def fig_pr_curves(views, scores, split="test"):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for name in viz.MODEL_ORDER:
        if name not in scores:
            continue
        d = views[M.VIEW_BY_MODEL[name]][split]
        p, r = E.pr_curve(d["y"], scores[name][split])
        ap = E.ranking_metrics(d["y"], scores[name][split])["pr_auc"]
        ax.plot(r, p, color=viz.model_color(name), label=f"{viz.MODEL_LABEL[name]} ({ap:.3f})")
    base = float(np.mean(views["tree"][split]["y"]))
    ax.axhline(base, ls="--", lw=1, color=viz.MUTED)
    ax.set(title=f"Precision-recall on the {split} period (PR-AUC in brackets)", xlabel="Recall",
           ylabel="Precision", ylim=(0, 1.02))
    ax.legend(fontsize=8, loc="lower left")
    return fig


def fig_threshold_sweep(split_data, sc, tcfg, chosen, name):
    sweep = E.threshold_sweep(split_data["y"], sc, split_data["amount"], split_data["days"], tcfg)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(sweep["alerts_per_day"], sweep["precision"], label="Precision", color=viz.SERIES[0])
    axes[0].plot(sweep["alerts_per_day"], sweep["recall"], label="Recall", color=viz.SERIES[1])
    axes[0].set(title="Precision and recall by alert volume (validation)", xlabel="Alerts per day",
                ylabel="Rate", xscale="log")
    axes[0].legend()
    chosen_row = E.business_metrics(split_data["y"], sc, split_data["amount"], chosen, split_data["days"],
                                    tcfg["investigation_cost"], tcfg["recovery_rate"])
    axes[1].plot(sweep["alerts_per_day"], sweep["net_benefit"] / 1000, color=viz.SERIES[0])
    axes[1].axvline(chosen_row["alerts_per_day"], ls=":", color=viz.INK,
                    label=f"chosen: {chosen_row['alerts_per_day']:.0f} alerts/day")
    axes[1].set(title="Net benefit by alert volume (validation, $ thousands)", xlabel="Alerts per day",
                ylabel="$ thousands", xscale="log")
    axes[1].legend()
    fig.suptitle(viz.MODEL_LABEL.get(name, name), x=0.01, ha="left", fontsize=10, color=viz.INK_2)
    fig.tight_layout()
    return fig, sweep


def fig_score_distribution(y, scores, threshold, name):
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 51)
    ax.hist(scores[y == 0], bins=bins, density=True, alpha=0.6, color=viz.LEGIT, label="Legitimate")
    ax.hist(scores[y == 1], bins=bins, density=True, alpha=0.6, color=viz.FRAUD, label="Fraud")
    ax.axvline(threshold, ls="--", color=viz.INK, lw=1, label=f"threshold {threshold:.3f}")
    ax.set(title=f"{viz.MODEL_LABEL.get(name, name)}: score distribution (test)", xlabel="Fraud score",
           ylabel="Density (log)", yscale="log")
    ax.legend(fontsize=8)
    return fig


def fig_importance(model, X, y, n_rows, seed, name):
    idx = np.random.default_rng(seed).choice(len(X), min(n_rows, len(X)), replace=False)
    imp = permutation_importance(model, X.iloc[idx], y[idx], scoring="average_precision", n_repeats=3,
                                 random_state=seed, n_jobs=-1)
    table = pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean, "std": imp.importances_std})
    table = table.sort_values("importance", ascending=False)
    top = table.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color=viz.SERIES[0])
    ax.set(title=f"{viz.MODEL_LABEL.get(name, name)}: permutation importance (validation)",
           xlabel="Drop in PR-AUC when the feature is shuffled")
    fig.tight_layout()
    return fig, table.round(5)


def fig_calibration(cal: pd.DataFrame, name):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=viz.MUTED, label="perfect calibration")
    ax.plot(cal["mean_score"], cal["fraud_rate"], marker="o", color=viz.SERIES[0], label="model")
    ax.set(title=f"Calibration on the test period", xlabel="Mean predicted score",
           ylabel="Observed fraud rate", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    return fig


def fig_monthly(monthly: pd.DataFrame, name):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(monthly["period"], monthly["recall"], marker="o", color=viz.SERIES[0], label="Recall")
    ax.plot(monthly["period"], monthly["precision"], marker="o", color=viz.SERIES[1], label="Precision")
    ax.set(title="Champion performance by month (test period, fixed threshold)", ylim=(0, 1.05), ylabel="Rate")
    ax.legend()
    return fig


# ------------------------------------------------------------------- runner
def run_training(cfg: dict, tracker: Tracker | None = None) -> dict:
    t0 = time.perf_counter()
    tracker = tracker or Tracker(cfg)
    mcfg, tcfg, ecfg = cfg["model"], cfg["model"]["threshold"], cfg.get("evaluation", {})
    figs = cfg["paths"]["figures"]
    views = {v: load_view(cfg, v) for v in ("tree", "linear")}
    log.info("Training rows: %s", f"{len(views['tree']['train']['X']):,}")

    run_name = f"train-{dt.datetime.now():%Y%m%d-%H%M%S}"
    with tracker.run(run_name, tags={"stage": "modelling"}):
        tracker.log_params({"models": mcfg["models"], "threshold_strategy": tcfg["primary"],
                            "investigation_cost": tcfg["investigation_cost"],
                            "recovery_rate": tcfg["recovery_rate"],
                            "tuning_n_iter": mcfg.get("tuning", {}).get("n_iter"),
                            "tuning_cv_folds": mcfg.get("tuning", {}).get("cv_folds")})
        trained = train_models(cfg, views, tracker)
        results = evaluate_models(cfg, views, trained["scores"])
        table = comparison_table(results, trained["tuning"])
        for name, r in results.items():
            tracker.log_metrics({k: r["validation"][k] for k in ("pr_auc", "roc_auc", "precision", "recall")},
                                prefix=f"{name}.val.")
            tracker.log_metrics({k: r["test"][k] for k in ("pr_auc", "roc_auc", "precision", "recall",
                                                          "net_benefit", "alerts_per_day")},
                                prefix=f"{name}.test.")

        sup = [m for m in trained["scores"] if m in M.SUPERVISED]
        y_val = views["tree"]["val"]["y"]
        selection = select_champion(
            y_val, {m: trained["scores"][m]["val"] for m in sup},
            {m: results[m]["validation"]["pr_auc"] for m in sup},
            complexity_order=mcfg.get("complexity_order", DEFAULT_COMPLEXITY_ORDER),
            n_boot=mcfg.get("selection_bootstrap", 500), seed=mcfg["random_state"],
            margin=mcfg.get("selection_margin", 0.01))
        champion = selection["champion"]
        log.info("Champion: %s | %s", champion, selection["reason"])

        # ---- champion deep-dive (all on test, threshold fixed from validation)
        cv = views[M.VIEW_BY_MODEL[champion]]
        thr = results[champion]["thresholds_from_validation"][tcfg["primary"]]
        sc_test = trained["scores"][champion]["test"]
        ci = E.bootstrap_ci(cv["test"]["y"], sc_test, cv["test"]["amount"], thr, cv["test"]["days"], tcfg,
                            n_boot=ecfg.get("bootstrap", 300), seed=mcfg["random_state"])
        monthly = E.performance_by_period(cv["test"]["ts"], cv["test"]["y"], sc_test, cv["test"]["amount"], thr, tcfg)
        cal = E.calibration_table(cv["test"]["y"], sc_test)
        alt = results[champion]["thresholds_from_validation"]["alert_budget"]
        break_even = E.break_even_review_cost(cv["test"]["y"], sc_test, cv["test"]["amount"], alt, thr,
                                              tcfg["recovery_rate"])

        viz.save(fig_pr_curves(views, trained["scores"]), figs / "10_pr_curves_test")
        fig, sweep = fig_threshold_sweep(cv["val"], trained["scores"][champion]["val"], tcfg, thr, champion)
        viz.save(fig, figs / "11_threshold_sweep_validation")
        viz.save(fig_score_distribution(cv["test"]["y"], sc_test, thr, champion), figs / "12_score_distribution_test")
        Xi = cv["val"]["X"]
        if champion == "hybrid_gb":
            Xi = Xi.assign(anomaly_score=trained["scores"]["isolation_forest"]["val"])
        fig, importance = fig_importance(trained["fitted"][champion], Xi, cv["val"]["y"],
                                         ecfg.get("permutation_importance_rows", 50000), mcfg["random_state"], champion)
        viz.save(fig, figs / "13_permutation_importance")
        viz.save(fig_calibration(cal, champion), figs / "14_calibration_test")
        viz.save(fig_monthly(monthly, champion), figs / "15_monthly_performance_test")

        # ---- package the champion for serving
        pre = joblib.load(cfg["paths"]["artifacts"] / f"preprocessor_{M.VIEW_BY_MODEL[champion]}.joblib")
        fm = FraudModel(
            champion=champion, model=trained["fitted"][champion], preprocessor=pre, threshold=float(thr),
            threshold_strategy=tcfg["primary"], view=M.VIEW_BY_MODEL[champion],
            iso_model=trained["fitted"].get("isolation_forest") if champion == "hybrid_gb" else None,
            linear_preprocessor=(joblib.load(cfg["paths"]["artifacts"] / "preprocessor_linear.joblib")
                                 if champion == "hybrid_gb" else None),
            meta={"trained_at": dt.datetime.now().isoformat(timespec="seconds"),
                  "library": M.LIBRARY[champion], "selection_reason": selection["reason"],
                  "test_pr_auc": results[champion]["test"]["pr_auc"], "mlflow_run_id": tracker.run_id})
        fm.save(cfg["paths"]["artifacts"] / "fraudlens_model.joblib")

        report = {
            "protocol": "tuned with time-series CV inside train; fitted on train; champion and threshold "
                        "chosen on validation; test scored once",
            "champion": champion,
            "selection": selection,
            "imbalance_handling": {n: M.imbalance_method(n, m) for n, m in trained["fitted"].items()},
            "threshold_strategy": tcfg["primary"],
            "comparison": table.reset_index().to_dict("records"),
            "results": results,
            "tuning": {n: {k: v for k, v in t.items() if k != "candidates"} | {"candidates": t["candidates"]}
                       for n, t in trained["tuning"].items()},
            "champion_test_ci95": ci,
            "champion_monthly_test": monthly.to_dict("records"),
            "champion_calibration": cal.to_dict("records"),
            "champion_importance": importance.to_dict("records"),
            "break_even_review_cost": break_even,
            "validation_sweep": sweep.to_dict("records"),
            "skipped_models": trained["skipped"],
            "train_seconds": trained["timings"],
            "mlflow": {"tracking_uri": getattr(tracker, "uri", None), "run_id": tracker.run_id},
            "runtime_seconds": round(time.perf_counter() - t0, 1),
        }
        write_json(report, cfg["paths"]["reports"] / "model_results.json")
        try:
            from .interactive import model_explorer
            model_explorer(report, sweep, monthly, cfg)
        except ImportError:
            log.warning("Plotly not installed; skipping the interactive model explorer")
        tracker.log_metrics({"champion.test.pr_auc": results[champion]["test"]["pr_auc"]})
        tracker.log_artifact(cfg["paths"]["reports"] / "model_results.json", "reports")
        for f in sorted(figs.glob("1[0-5]_*.png")):
            tracker.log_artifact(f, "figures")
        reg = cfg["platform"].get("registered_model", "fraudlens-champion")
        tracker.log_model(trained["fitted"][champion], "champion_model", M.LIBRARY[champion], register_as=reg)
        version = tracker.set_champion_alias(reg)
        if version is not None:
            report["mlflow"]["registered_model"] = reg
            report["mlflow"]["champion_version"] = version
            write_json(report, cfg["paths"]["reports"] / "model_results.json")
    log.info("Training finished in %.1fs", report["runtime_seconds"])
    return report


if __name__ == "__main__":
    run_training(load_config())
