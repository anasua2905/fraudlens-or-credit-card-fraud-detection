"""Lifecycle stage 8 - Monitoring and improvement.

Replays the scored validation and test periods month by month, as if the
model were live, and asks two questions each month:

  1. Has the input data drifted away from what the model learned from?
     Evidently AI's DataDriftPreset compares each month with the training
     reference (an HTML report per month); the project's own Population
     Stability Index is computed alongside as a cross-check.
  2. Is the model still performing? Recall, precision, PR-AUC and alert
     volume at the fixed threshold (labels are known in this replay; in
     production they arrive later, from chargebacks and analyst feedback).

A retraining recommendation is raised when the drifted-feature share or the
recall/precision falls past the thresholds in config `monitoring`. With
`--retrain-if-needed` the full training pipeline is re-run automatically.

    python -m fraudlens.monitor [--retrain-if-needed]
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd

from . import evaluate as E
from . import viz
from .datasets import load_features
from .drift import psi, psi_flag
from .preprocess import add_stationary_features
from .utils import get_logger, load_config, read_json, read_table, rel, write_json, write_table

log = get_logger()


def _reference(cfg: dict, features: list[str], n: int = 100_000) -> pd.DataFrame:
    ref = add_stationary_features(load_features(cfg, "train"))
    ref = ref.sample(min(n, len(ref)), random_state=cfg["model"]["random_state"])
    return ref[features]


def evidently_drift(reference: pd.DataFrame, current: pd.DataFrame, numeric: list[str],
                    categorical: list[str], html_path=None) -> dict | None:
    """Run Evidently's DataDriftPreset; return {share, count, columns: {name: drifted}} or None."""
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        return None
    definition = DataDefinition(numerical_columns=numeric, categorical_columns=categorical)
    ref = Dataset.from_pandas(reference, data_definition=definition)
    cur = Dataset.from_pandas(current, data_definition=definition)
    result = Report([DataDriftPreset()]).run(cur, ref)
    if html_path is not None:
        result.save_html(str(html_path))
    return parse_evidently(result.dict())


def parse_evidently(d: dict) -> dict:
    """Pull the drifted-column share and per-column drift flags out of Evidently's result dict."""
    out = {"share": None, "count": None, "columns": {}}
    for m in d.get("metrics", []):
        mid = str(m.get("metric_id") or m.get("metric_name") or m.get("id") or "")
        val = m.get("value")
        if mid.startswith("DriftedColumnsCount") and isinstance(val, dict):
            out["share"] = float(val.get("share")) if val.get("share") is not None else None
            out["count"] = int(val.get("count")) if val.get("count") is not None else None
        elif mid.startswith("ValueDrift") and val is not None:
            col = mid.split("column=", 1)[-1].split(",", 1)[0].rstrip(")")
            thr = None
            if "threshold=" in mid:
                try:
                    thr = float(mid.split("threshold=", 1)[1].split(",", 1)[0].rstrip(")"))
                except ValueError:
                    thr = None
            score = float(val) if not isinstance(val, dict) else float(val.get("value", np.nan))
            method = mid.split("method=", 1)[1].split(",", 1)[0].rstrip(")") if "method=" in mid else ""
            # p-value tests drift when p < threshold; distance tests when distance > threshold
            if thr is None:
                drifted = None
            elif "p_value" in method or "test" in method.lower() or method in ("K-S p_value", "chi-square p_value"):
                drifted = score < thr
            else:
                drifted = score > thr
            out["columns"][col] = {"score": round(score, 5), "method": method, "threshold": thr, "drifted": drifted}
    return out


def run_monitoring(cfg: dict) -> dict:
    mcfg = cfg["monitoring"]
    P = cfg["paths"]
    tcfg = cfg["model"]["threshold"]
    feats_cfg = mcfg["features"]
    categorical = [c for c in feats_cfg if c in ("category", "gender")]
    numeric = [c for c in feats_cfg if c not in categorical]
    reference = _reference(cfg, feats_cfg)
    for c in categorical:
        reference[c] = reference[c].astype(str)

    scored = read_table(P["gold"] / "scored_transactions")
    meta = read_json(P["reports"] / "dashboard_meta.json")
    thr = float(meta["threshold"])
    cur_all = pd.concat([add_stationary_features(load_features(cfg, s)) for s in ("val", "test")], ignore_index=True)
    cur_all = cur_all.merge(scored[["transaction_id", "score"]], on="transaction_id", how="inner")
    cur_all["month"] = pd.to_datetime(cur_all["ts"]).dt.to_period(mcfg.get("window", "M")).astype(str)
    for c in categorical:
        cur_all[c] = cur_all[c].astype(str)

    rows, column_rows = [], []
    used_evidently = False
    for month, g in cur_all.groupby("month"):
        html = P["monitoring"] / f"drift_{month}.html"
        ev = evidently_drift(reference, g[feats_cfg], numeric, categorical, html)
        used_evidently = used_evidently or ev is not None
        psis = {c: psi(reference[c], g[c]) for c in numeric}
        psi_share = float(np.mean([v >= 0.25 for v in psis.values()]))
        y, s, a = g["is_fraud"].to_numpy(), g["score"].to_numpy(), g["amount"].to_numpy()
        days = max((pd.to_datetime(g["ts"]).max() - pd.to_datetime(g["ts"]).min()).total_seconds() / 86400, 1)
        perf = E.confusion_at(y, s, thr)
        perf.update(E.business_metrics(y, s, a, thr, days, tcfg["investigation_cost"], tcfg["recovery_rate"]))
        row = {"month": month, "transactions": len(g), "frauds": int(y.sum()),
               "fraud_rate_pct": round(100 * float(y.mean()), 3),
               "evidently_drift_share": None if ev is None else ev["share"],
               "evidently_drifted_columns": None if ev is None else ev["count"],
               "psi_major_share": round(psi_share, 3),
               "pr_auc": round(float(E.average_precision_score(y, s)), 4) if 0 < y.sum() < len(y) else None,
               "precision": perf["precision"], "recall": perf["recall"], "alerts": perf["alerts"],
               "alerts_per_day": perf["alerts_per_day"], "net_benefit": perf["net_benefit"],
               "report": rel(html, cfg) if ev is not None else None}
        drift_share = row["evidently_drift_share"] if row["evidently_drift_share"] is not None else psi_share
        reasons = []
        if drift_share > mcfg["drift_share_alert"]:
            reasons.append(f"{drift_share:.0%} of monitored features drifted")
        if row["frauds"] and perf["recall"] < mcfg["recall_floor"]:
            reasons.append(f"recall {perf['recall']:.1%} below {mcfg['recall_floor']:.0%}")
        if perf["alerts"] and perf["precision"] < mcfg["precision_floor"]:
            reasons.append(f"precision {perf['precision']:.1%} below {mcfg['precision_floor']:.0%}")
        row["retrain_recommended"] = bool(reasons)
        row["reasons"] = "; ".join(reasons)
        rows.append(row)
        for c, v in psis.items():
            col_ev = (ev or {}).get("columns", {}).get(c, {})
            column_rows.append({"month": month, "feature": c, "psi": round(v, 4), "psi_flag": psi_flag(v),
                                "evidently_score": col_ev.get("score"), "evidently_drifted": col_ev.get("drifted")})
        log.info("Monitoring %s: drift share %.2f, recall %.3f, precision %.3f%s", month, drift_share,
                 perf["recall"], perf["precision"], "  -> RETRAIN" if reasons else "")

    monthly = pd.DataFrame(rows)
    columns = pd.DataFrame(column_rows)
    write_table(monthly, P["monitoring"] / "monthly", "parquet")
    monthly.to_csv(P["monitoring"] / "monthly.csv", index=False)
    columns.to_csv(P["monitoring"] / "feature_drift.csv", index=False)
    latest = rows[-1]
    summary = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "drift_engine": "Evidently DataDriftPreset + PSI cross-check" if used_evidently else "PSI (Evidently not installed)",
        "reference": "training period (100,000-row sample)",
        "threshold": thr,
        "months": len(rows),
        "latest_month": latest["month"],
        "latest": latest,
        "retrain_recommended": any(r["retrain_recommended"] for r in rows),
        "months_flagged": [r["month"] for r in rows if r["retrain_recommended"]],
        "policy": {k: mcfg[k] for k in ("drift_share_alert", "recall_floor", "precision_floor")},
        "monthly": rows,
    }
    write_json(summary, P["monitoring"] / "summary.json")
    _figure(monthly, cfg)
    _log_to_mlflow(cfg, rows, summary)
    return summary


def _log_to_mlflow(cfg: dict, rows: list[dict], summary: dict) -> None:
    """One MLflow run per monitoring pass; each month is a step, so the UI draws the trend."""
    from .tracking import Tracker
    tracker = Tracker(cfg)
    if not tracker.active:
        return
    with tracker.run(f"monitor-{dt.datetime.now():%Y%m%d-%H%M%S}", tags={"stage": "monitoring"}):
        for i, r in enumerate(rows):
            share = r["evidently_drift_share"] if r["evidently_drift_share"] is not None else r["psi_major_share"]
            for k, v in {"drift_share": share, "recall": r["recall"], "precision": r["precision"],
                         "alerts_per_day": r["alerts_per_day"], "pr_auc": r["pr_auc"]}.items():
                if v is not None:
                    tracker.mlflow.log_metric(k, float(v), step=i)
        tracker.mlflow.set_tag("retrain_recommended", str(summary["retrain_recommended"]))
        tracker.log_artifact(cfg["paths"]["monitoring"] / "summary.json", "monitoring")


def _figure(monthly: pd.DataFrame, cfg: dict) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(monthly["month"], monthly["recall"], marker="o", color=viz.SERIES[0], label="Recall")
    axes[0].plot(monthly["month"], monthly["precision"], marker="o", color=viz.SERIES[1], label="Precision")
    axes[0].axhline(cfg["monitoring"]["recall_floor"], ls="--", lw=1, color=viz.MUTED)
    axes[0].set(title="Monthly performance at the fixed threshold", ylim=(0, 1.05))
    axes[0].legend()
    share = monthly["evidently_drift_share"].fillna(monthly["psi_major_share"])
    axes[1].bar(monthly["month"], share, color=viz.SERIES[0])
    axes[1].axhline(cfg["monitoring"]["drift_share_alert"], ls="--", lw=1, color=viz.MUTED)
    axes[1].set(title="Share of monitored features drifting vs training", ylim=(0, 1))
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    viz.save(fig, cfg["paths"]["figures"] / "19_monitoring_monthly")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 8: drift and performance monitoring")
    ap.add_argument("--retrain-if-needed", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    summary = run_monitoring(cfg)
    if a.retrain_if_needed and summary["retrain_recommended"]:
        log.warning("Retraining triggered for months %s", summary["months_flagged"])
        from .run_all import run_all
        run_all(cfg, start="prepare", stop="verify")


if __name__ == "__main__":
    main()
