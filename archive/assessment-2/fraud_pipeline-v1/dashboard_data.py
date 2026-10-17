"""Data layer for the FraudLens monitoring dashboard.

Every number the dashboard shows is computed here, with plain pandas, so it
can be tested without a browser. The Streamlit app only arranges the output.

Note on outcomes: this is a historical replay, so the true label of every
alert is known. In live monitoring, recent alerts would be unresolved.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import evaluate as E
from .utils import read_table

AMOUNT_BANDS = [0, 10, 50, 100, 250, 500, 1000, np.inf]
AMOUNT_LABELS = ["<$10", "$10-50", "$50-100", "$100-250", "$250-500", "$500-1k", "$1k+"]
OUTCOME_ORDER = ["caught", "false_alert", "missed", "cleared"]


# ----------------------------------------------------------------- loading
def _json(path):
    return json.loads(path.read_text()) if path.exists() else None


def load_dashboard(cfg: dict) -> dict:
    P = cfg["paths"]
    df = read_table(P["gold"] / "scored_transactions")
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"])
    for c in ["category", "state", "gender", "merchant", "split"]:
        df[c] = df[c].astype(str)
    eda = _json(P["reports"] / "eda_summary.json") or {}
    category_risk = {k: v["fraud_rate_pct"] for k, v in eda.get("category", {}).items()}
    return {
        "df": df,
        "meta": _json(P["reports"] / "dashboard_meta.json"),
        "results": _json(P["reports"] / "model_results.json"),
        "preprocessing": _json(P["reports"] / "preprocessing_report.json"),
        "verification": _json(P["reports"] / "verification_report.json"),
        "category_risk_train": category_risk,  # fraud rates from the TRAINING period
    }


# ------------------------------------------------------------- filtering
def apply_filters(df: pd.DataFrame, split: str, start=None, end=None,
                  categories: list[str] | None = None, states: list[str] | None = None) -> pd.DataFrame:
    out = df[df["split"] == split]
    if start is not None:
        out = out[out["ts"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["ts"] < pd.Timestamp(end) + pd.Timedelta(days=1)]
    if categories:
        out = out[out["category"].isin(categories)]
    if states:
        out = out[out["state"].isin(states)]
    return out


def period_days(df: pd.DataFrame) -> float:
    if df.empty:
        return 1.0
    return max((df["ts"].max() - df["ts"].min()).total_seconds() / 86400, 1.0)


def outcomes(df: pd.DataFrame, threshold: float) -> pd.Series:
    alert = df["score"] >= threshold
    fraud = df["is_fraud"] == 1
    return pd.Series(np.select([alert & fraud, alert & ~fraud, ~alert & fraud],
                               ["caught", "false_alert", "missed"], "cleared"), index=df.index)


# ------------------------------------------------------------------- KPIs
def kpis(df: pd.DataFrame, threshold: float, cost: float, recovery: float) -> dict:
    if df.empty:
        return {}
    y, s, a = df["is_fraud"].to_numpy(), df["score"].to_numpy(), df["amount"].to_numpy()
    days = period_days(df)
    k = E.confusion_at(y, s, threshold)
    k.update(E.business_metrics(y, s, a, threshold, days, cost, recovery))
    alerted = df[df["score"] >= threshold]
    k.update({
        "transactions": len(df),
        "frauds": int(y.sum()),
        "days": round(days, 1),
        "cards_to_review": int(alerted["card_id"].nunique()),
        "alerts_per_card": round(len(alerted) / max(alerted["card_id"].nunique(), 1), 1),
    })
    if y.sum() and len(np.unique(y)) == 2:
        k.update(E.ranking_metrics(y, s))
    return k


def daily(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    d = df.assign(outcome=outcomes(df, threshold), date=df["ts"].dt.floor("D"))
    t = d.pivot_table(index="date", columns="outcome", values="transaction_id", aggfunc="count", fill_value=0)
    for col in OUTCOME_ORDER:
        if col not in t:
            t[col] = 0
    t["transactions"] = t[OUTCOME_ORDER].sum(axis=1)
    t["alerts"] = t["caught"] + t["false_alert"]
    return t[["transactions", "alerts", *OUTCOME_ORDER]].reset_index()


# ------------------------------------------------------------- alert queue
def card_queue(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """One row per card with alerts: fraud on a card arrives in ~45-hour bursts,
    so reviewing by card is far less work than reviewing every transaction."""
    al = df[df["score"] >= threshold]
    if al.empty:
        return pd.DataFrame(columns=["card_id", "alerts", "first_alert", "last_alert", "window_hours",
                                     "alerted_amount", "max_score", "top_category", "confirmed_fraud",
                                     "false_alerts"])
    g = al.groupby("card_id")
    q = pd.DataFrame({
        "alerts": g.size(),
        "first_alert": g["ts"].min(),
        "last_alert": g["ts"].max(),
        "alerted_amount": g["amount"].sum().round(2),
        "max_score": g["score"].max().round(4),
        "top_category": g["category"].agg(lambda s: s.value_counts().index[0]),
        "confirmed_fraud": g["is_fraud"].sum(),
    })
    q["false_alerts"] = q["alerts"] - q["confirmed_fraud"]
    q["window_hours"] = ((q["last_alert"] - q["first_alert"]).dt.total_seconds() / 3600).round(1)
    q = q.sort_values(["max_score", "alerted_amount"], ascending=False).reset_index()
    return q[["card_id", "alerts", "first_alert", "last_alert", "window_hours", "alerted_amount",
              "max_score", "top_category", "confirmed_fraud", "false_alerts"]]


def alert_reasons(row: pd.Series, category_risk: dict[str, float] | None = None) -> list[str]:
    """Plain-language evidence for one transaction, from the engineered features."""
    reasons = []
    ratio = row.get("amt_to_card_mean")
    if pd.notna(ratio) and ratio >= 2:
        reasons.append(f"Amount is {ratio:.1f}x this card's usual spend")
    if row.get("is_night") == 1:
        reasons.append(f"Night-time transaction ({int(row['hour']):02d}:00)")
    if category_risk:
        top = sorted(category_risk, key=category_risk.get, reverse=True)[:3]
        if row.get("category") in top:
            reasons.append(f"High-risk category ({row['category']}, "
                           f"{category_risk[row['category']]:.2f}% fraud rate in training)")
    n24 = row.get("txn_count_24h")
    if pd.notna(n24) and n24 >= 3:
        reasons.append(f"{int(n24)} earlier transactions in 24h totalling ${row.get('amt_sum_24h', 0):,.0f}")
    gap = row.get("secs_since_last_txn")
    if pd.notna(gap) and gap < 3600:
        reasons.append(f"Only {gap / 60:.0f} min since the card's previous transaction")
    return reasons or ["No single strong indicator; flagged by the combined pattern"]


def card_timeline(df: pd.DataFrame, card_id: str, threshold: float,
                  category_risk: dict[str, float] | None = None) -> pd.DataFrame:
    t = df[df["card_id"] == card_id].sort_values("ts").copy()
    t["outcome"] = outcomes(t, threshold)
    t["reasons"] = [("; ".join(alert_reasons(r, category_risk)) if r["score"] >= threshold else "")
                    for _, r in t.iterrows()]
    return t


# ------------------------------------------------------ threshold & segments
def sweep(df: pd.DataFrame, cost: float, recovery: float, points: int = 80) -> pd.DataFrame:
    if df.empty or df["is_fraud"].sum() == 0:
        return pd.DataFrame()
    cfg_thr = {"investigation_cost": cost, "recovery_rate": recovery}
    return E.threshold_sweep(df["is_fraud"].to_numpy(), df["score"].to_numpy(), df["amount"].to_numpy(),
                             period_days(df), cfg_thr, points=points)


def break_even_cost(df: pd.DataFrame, thr_low: float, thr_high: float, recovery: float) -> float | None:
    """Review cost at which lowering the threshold from thr_high to thr_low pays for itself."""
    y, s, a = df["is_fraud"].to_numpy(), df["score"].to_numpy(), df["amount"].to_numpy()
    extra_alerts = int(((s >= thr_low) & (s < thr_high)).sum())
    extra_value = recovery * float(a[(s >= thr_low) & (s < thr_high) & (y == 1)].sum())
    return round(extra_value / extra_alerts, 2) if extra_alerts else None


def segment(df: pd.DataFrame, threshold: float, by: str) -> pd.DataFrame:
    d = df.assign(outcome=outcomes(df, threshold))
    if by == "amount_band":
        d["amount_band"] = pd.cut(d["amount"], AMOUNT_BANDS, labels=AMOUNT_LABELS)
    t = d.pivot_table(index=by, columns="outcome", values="transaction_id", aggfunc="count",
                      fill_value=0, observed=True)
    for col in OUTCOME_ORDER:
        if col not in t:
            t[col] = 0
    t["transactions"] = t[OUTCOME_ORDER].sum(axis=1)
    t["frauds"] = t["caught"] + t["missed"]
    t["alerts"] = t["caught"] + t["false_alert"]
    t["fraud_rate_pct"] = (100 * t["frauds"] / t["transactions"]).round(3)
    t["precision"] = (t["caught"] / t["alerts"].replace(0, np.nan)).round(3)
    t["recall"] = (t["caught"] / t["frauds"].replace(0, np.nan)).round(3)
    return t[["transactions", "frauds", "fraud_rate_pct", "alerts", "caught", "false_alert",
              "missed", "precision", "recall"]].reset_index()


# --------------------------------------------------------------- reporting
def summary_markdown(k: dict, filters: dict, meta: dict, top_categories: pd.DataFrame) -> str:
    lines = [
        "# FraudLens monitoring summary",
        "",
        f"**Period:** {filters['split']} split, {filters['start']} to {filters['end']}  ",
        f"**Model:** {meta['champion']}  |  **Threshold:** {filters['threshold']:.4f}  ",
        f"**Filters:** categories = {filters['categories'] or 'all'}; states = {filters['states'] or 'all'}",
        "",
        "## Headline",
        "",
        "| Measure | Value |", "|---|---|",
        f"| Transactions | {k['transactions']:,} |",
        f"| Frauds | {k['frauds']:,} |",
        f"| Alerts | {k['alerts']:,} ({k['alerts_per_day']} per day) |",
        f"| Cards to review | {k['cards_to_review']:,} |",
        f"| Precision | {k['precision']:.1%} |",
        f"| Recall | {k['recall']:.1%} |",
        f"| False alarms per catch | {k['false_alarms_per_catch']} |",
        f"| Fraud value recovered | ${k['fraud_amount_caught']:,.0f} |",
        f"| Fraud value missed | ${k['fraud_amount_missed']:,.0f} |",
        f"| Review cost | ${k['review_cost']:,.0f} |",
        f"| Net benefit | ${k['net_benefit']:,.0f} |",
        "",
        "## Categories with the most false alerts",
        "",
        "| Category | Alerts | Caught | False alerts | Precision |", "|---|---|---|---|---|",
    ]
    for _, r in top_categories.head(5).iterrows():
        prec = "n/a" if pd.isna(r["precision"]) else f"{r['precision']:.1%}"
        lines.append(f"| {r['category']} | {int(r['alerts'])} | {int(r['caught'])} | {int(r['false_alert'])} | {prec} |")
    lines += ["", f"_Assumptions: review cost ${filters['cost']} per alert, recovery rate {filters['recovery']:.0%}._"]
    return "\n".join(lines)
