"""Evaluation for a heavily imbalanced problem.

Accuracy is useless here: labelling everything legitimate scores 99.4%. These
metrics answer the questions a bank actually asks — how much fraud do we catch,
how many good customers do we bother, and is the review queue affordable?

Thresholds are chosen on VALIDATION and applied once to test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def ranking_metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    """Threshold-free quality. PR-AUC is compared with its random baseline (the fraud rate)."""
    base = float(np.mean(y))
    ap = float(average_precision_score(y, scores))
    return {
        "pr_auc": round(ap, 4),
        "pr_auc_baseline": round(base, 5),
        "pr_auc_lift": round(ap / base, 1) if base else None,
        "roc_auc": round(float(roc_auc_score(y, scores)), 4),
    }


def confusion_at(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    flagged = scores >= threshold
    tp = int(np.sum(flagged & (y == 1)))
    fp = int(np.sum(flagged & (y == 0)))
    fn = int(np.sum(~flagged & (y == 1)))
    tn = int(np.sum(~flagged & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": round(float(threshold), 6),
        "alerts": tp + fp, "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "false_positive_rate": round(fp / (fp + tn), 5) if fp + tn else 0.0,
        "alert_rate_pct": round(100 * (tp + fp) / len(y), 3),
        # How many good transactions are questioned for each fraud caught:
        "false_alarms_per_catch": round(fp / tp, 1) if tp else None,
    }


def business_metrics(y, scores, amounts, threshold, days, investigation_cost, recovery_rate) -> dict:
    flagged = scores >= threshold
    caught = float(np.sum(amounts[flagged & (y == 1)]))
    missed = float(np.sum(amounts[~flagged & (y == 1)]))
    alerts = int(flagged.sum())
    return {
        "alerts_per_day": round(alerts / days, 1),
        "fraud_amount_caught": round(recovery_rate * caught, 2),
        "fraud_amount_missed": round(missed, 2),
        "review_cost": round(alerts * investigation_cost, 2),
        "net_benefit": round(recovery_rate * caught - alerts * investigation_cost, 2),
    }


def evaluate(y, scores, amounts, threshold, days, cfg_thr: dict) -> dict:
    out = ranking_metrics(y, scores)
    out.update(confusion_at(y, scores, threshold))
    out.update(business_metrics(y, scores, amounts, threshold, days,
                                cfg_thr["investigation_cost"], cfg_thr["recovery_rate"]))
    return out


def _candidate_thresholds(scores: np.ndarray, n: int = 400) -> np.ndarray:
    qs = np.linspace(0.5, 0.99999, n)
    return np.unique(np.quantile(scores, qs))


def choose_threshold(y, scores, amounts, days, cfg_thr: dict, strategy: str) -> float:
    """Pick an operating point on the validation split."""
    cand = _candidate_thresholds(scores)
    if strategy == "alert_budget":
        wanted = cfg_thr["alerts_per_day"] * days
        return float(np.quantile(scores, max(0.0, 1 - wanted / len(scores))))
    if strategy == "max_f1":
        f1 = [confusion_at(y, scores, t)["f1"] for t in cand]
        return float(cand[int(np.argmax(f1))])
    if strategy == "min_cost":  # maximise net benefit
        net = [cfg_thr["recovery_rate"] * amounts[(scores >= t) & (y == 1)].sum()
               - (scores >= t).sum() * cfg_thr["investigation_cost"] for t in cand]
        return float(cand[int(np.argmax(net))])
    raise ValueError(f"Unknown threshold strategy {strategy!r}")


def threshold_sweep(y, scores, amounts, days, cfg_thr: dict, points: int = 60) -> pd.DataFrame:
    """Precision, recall and net benefit across alert volumes — the false-positive trade-off."""
    rows = []
    for q in np.linspace(0.90, 0.9999, points):
        t = float(np.quantile(scores, q))
        r = confusion_at(y, scores, t)
        r.update(business_metrics(y, scores, amounts, t, days,
                                  cfg_thr["investigation_cost"], cfg_thr["recovery_rate"]))
        rows.append(r)
    return pd.DataFrame(rows).drop_duplicates(subset="threshold")


def pr_curve(y, scores, max_points: int = 2000):
    precision, recall, _ = precision_recall_curve(y, scores)
    step = max(1, len(precision) // max_points)
    return precision[::step], recall[::step]
