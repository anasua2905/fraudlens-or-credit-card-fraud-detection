"""How far do the test results carry? Monthly stability and sensitivity to real-world conditions.

Two questions the simulated test period cannot answer directly, made explicit:

  1. Is performance stable month to month? For each test month: recall and precision at the
     deployed threshold with Wilson 95% intervals, the lowest month against the policy floor,
     and a chi-square test of whether the months differ more than chance allows.
  2. What would precision and alert volume look like if real data had a different fraud rate,
     or the model raised more false alarms on real customers than on simulated ones? Precision
     depends on prevalence: precision = TPR p / (TPR p + FPR (1 - p)). The recall (TPR) and
     false-positive rate measured on the test period are carried over and stressed.

    python -m fraudlens.stability        # writes reports/stability.json
"""
from __future__ import annotations

from scipy.stats import chi2_contingency

from .casework import wilson
from .utils import get_logger, load_config, read_json, write_json

log = get_logger()

PREVALENCES = [0.0005, 0.001, 0.002, None, 0.005, 0.01]     # None = the test period's own fraud rate
FPR_FACTORS = [1, 2, 5]


def monthly_stability(results: dict, recall_floor: float, precision_floor: float) -> dict:
    months = results["champion_monthly_test"]
    rows = []
    for m in months:
        tp, fn, fp = m["true_positives"], m["false_negatives"], m["false_positives"]
        r_lo, r_hi = wilson(tp, tp + fn)
        p_lo, p_hi = wilson(tp, tp + fp)
        rows.append({"month": m["period"], "transactions": m["transactions"], "frauds": m["frauds"],
                     "alerts": m["alerts"], "recall": m["recall"], "recall_ci95": [r_lo, r_hi],
                     "precision": m["precision"], "precision_ci95": [p_lo, p_hi],
                     "alerts_per_day": m["alerts_per_day"], "pr_auc": m.get("pr_auc")})
    p_recall = float(chi2_contingency([[m["true_positives"], m["false_negatives"]] for m in months])[1])
    p_prec = float(chi2_contingency([[m["true_positives"], m["false_positives"]] for m in months])[1])
    rec = [r["recall"] for r in rows]
    prec = [r["precision"] for r in rows]
    checks = [
        {"check": f"every month's recall >= the {recall_floor:.0%} floor", "status": "PASS" if min(rec) >= recall_floor else "FAIL",
         "detail": f"lowest {min(rec):.3f} ({rows[rec.index(min(rec))]['month']})"},
        {"check": f"every month's precision >= the {precision_floor:.0%} floor",
         "status": "PASS" if min(prec) >= precision_floor else "FAIL",
         "detail": f"lowest {min(prec):.3f} ({rows[prec.index(min(prec))]['month']})"},
        {"check": "monthly recall differs no more than chance (chi-square p >= 0.05)",
         "status": "PASS" if p_recall >= 0.05 else "WARN", "detail": f"p = {p_recall:.3f}"},
        {"check": "monthly precision differs no more than chance (chi-square p >= 0.05)",
         "status": "PASS" if p_prec >= 0.05 else "WARN", "detail": f"p = {p_prec:.3f}"},
    ]
    return {"months": rows, "recall_range": [min(rec), max(rec)], "precision_range": [min(prec), max(prec)],
            "chi2_p_recall": round(p_recall, 4), "chi2_p_precision": round(p_prec, 4), "checks": checks,
            "stable": all(c["status"] == "PASS" for c in checks)}


def prevalence_sensitivity(test: dict, days: float) -> dict:
    """Precision and alerts/day at other fraud rates and higher false-alarm rates (same recall)."""
    tp, fn, fp, tn = (test[k] for k in ("true_positives", "false_negatives", "false_positives", "true_negatives"))
    n = tp + fn + fp + tn
    tpr, fpr, base = tp / (tp + fn), fp / (fp + tn), (tp + fn) / n
    per_day = n / days
    rows = []
    for p in PREVALENCES:
        prev = base if p is None else p
        row = {"fraud_rate_pct": round(100 * prev, 3), "is_test_rate": p is None}
        for f in FPR_FACTORS:
            alerts = tpr * prev + min(fpr * f, 1) * (1 - prev)
            row[f"precision_fpr_x{f}"] = round(tpr * prev / alerts, 4)
            row[f"alerts_per_day_fpr_x{f}"] = round(alerts * per_day, 1)
        rows.append(row)
    return {"recall": round(tpr, 4), "false_positive_rate": round(fpr, 6), "test_fraud_rate_pct": round(100 * base, 3),
            "transactions_per_day": round(per_day, 1), "fpr_factors": FPR_FACTORS, "rows": rows,
            "note": "Recall is held at the test value; real data may also lower it. Precision falls with rarer "
                    "fraud and with more false alarms, which is why a shadow-mode trial on real transactions "
                    "comes before any savings claim."}


def run_stability(cfg: dict) -> dict:
    P = cfg["paths"]
    res = read_json(P["reports"] / "model_results.json")
    test = res["results"][res["champion"]]["test"]
    mon = cfg["monitoring"]
    days = sum(m["alerts"] / m["alerts_per_day"] for m in res["champion_monthly_test"]   # days in each month
               if m["alerts_per_day"]) or 1.0
    out = {"threshold": test["threshold"], "monthly": monthly_stability(res, mon["recall_floor"], mon["precision_floor"]),
           "prevalence": prevalence_sensitivity(test, days)}
    write_json(out, P["reports"] / "stability.json")
    log.info("Stability: recall %.3f-%.3f by month (chi-square p %.2f); %s", *out["monthly"]["recall_range"],
             out["monthly"]["chi2_p_recall"], "stable" if out["monthly"]["stable"] else "NOT stable")
    return out


if __name__ == "__main__":
    run_stability(load_config())
