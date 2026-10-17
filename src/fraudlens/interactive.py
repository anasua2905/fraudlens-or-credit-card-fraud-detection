"""Interactive Plotly exports (stages 4 and 6): self-contained HTML pages in reports/interactive/."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import viz
from .utils import get_logger

log = get_logger()


def _page(title: str, intro: str, figs: list[go.Figure], path: Path) -> Path:
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>"
             "<meta name='viewport' content='width=device-width, initial-scale=1'>"
             f"<style>body{{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;background:{viz.SURFACE};"
             f"color:{viz.INK};max-width:1100px;margin:24px auto;padding:0 16px}}"
             f"p{{color:{viz.INK_2};line-height:1.5}}</style></head><body>",
             f"<h1>{title}</h1><p>{intro}</p>"]
    for i, f in enumerate(figs):
        f.update_layout(template=viz.plotly_template())
        parts.append(f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False))
    parts.append("</body></html>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def eda_explorer(df: pd.DataFrame, summary: dict, cfg: dict) -> Path:
    hour = pd.DataFrame(summary["time"]["hour"]).T.reset_index(names="hour")
    hour["hour"] = hour["hour"].astype(int)
    f1 = px.bar(hour, x="hour", y="fraud_rate_pct", title="Fraud rate by hour of day (training data)",
                labels={"fraud_rate_pct": "Fraud rate (%)", "hour": "Hour"},
                hover_data={"transactions": ":,", "frauds": ":,"}, color_discrete_sequence=[viz.SERIES[0]])

    cat = pd.DataFrame(summary["category"]).T.reset_index(names="category").sort_values("fraud_rate_pct")
    f2 = px.bar(cat, x="fraud_rate_pct", y="category", orientation="h",
                title="Fraud rate by merchant category", labels={"fraud_rate_pct": "Fraud rate (%)", "category": ""},
                hover_data={"frauds": ":,", "share_of_frauds_pct": ":.1f", "fraud_amount": ":,.0f"},
                color_discrete_sequence=[viz.SERIES[0]])

    rng = np.random.default_rng(0)
    legit = df[df["is_fraud"] == 0]
    s = pd.concat([df[df["is_fraud"] == 1], legit.iloc[rng.choice(len(legit), min(60_000, len(legit)), replace=False)]])
    s = s.assign(Class=s["is_fraud"].map({0: "Legitimate", 1: "Fraud"}), log10_amount=np.log10(s["amount"]))
    f3 = px.histogram(s, x="log10_amount", color="Class", histnorm="probability density", barmode="overlay",
                      nbins=80, opacity=0.6, title="Amount distribution by class (log10 dollars)",
                      color_discrete_map={"Legitimate": viz.LEGIT, "Fraud": viz.FRAUD},
                      labels={"log10_amount": "log10(amount $)"})

    st = df.groupby("state", observed=True).agg(transactions=("is_fraud", "size"), frauds=("is_fraud", "sum"))
    st = st[st["transactions"] >= 1000].assign(fraud_rate_pct=lambda t: 100 * t["frauds"] / t["transactions"])
    f4 = px.choropleth(st.reset_index(), locations="state", locationmode="USA-states", scope="usa",
                       color="fraud_rate_pct", color_continuous_scale=viz.BLUE_RAMP,
                       title="Fraud rate by cardholder state (states with 1,000+ transactions)",
                       labels={"fraud_rate_pct": "Fraud rate (%)"}, hover_data={"transactions": ":,", "frauds": ":,"})

    ratio = pd.DataFrame(summary["behaviour"]["amount_vs_card_mean"]).T.reset_index(names="band")
    f5 = px.bar(ratio, x="band", y="fraud_rate_pct", title="Fraud rate by amount relative to the card's usual spend",
                labels={"fraud_rate_pct": "Fraud rate (%)", "band": "Amount / card's historical mean"},
                hover_data={"transactions": ":,", "frauds": ":,"}, color_discrete_sequence=[viz.SERIES[0]])
    out = _page("FraudLens - interactive EDA explorer",
                "Training period only (Jan 2019 to May 2020). Hover any bar for counts.",
                [f1, f2, f3, f4, f5], cfg["paths"]["interactive"] / "eda_explorer.html")
    log.info("Interactive EDA explorer written to %s", out)
    return out


def model_explorer(results: dict, sweep: pd.DataFrame, monthly: pd.DataFrame, cfg: dict) -> Path:
    comp = pd.DataFrame(results["comparison"])
    comp["label"] = comp["model"].map(viz.MODEL_LABEL)
    f1 = px.bar(comp.sort_values("test_pr_auc"), x="test_pr_auc", y="label", orientation="h",
                title="Test PR-AUC by model (higher is better)", labels={"test_pr_auc": "PR-AUC", "label": ""},
                color="model", color_discrete_map={m: viz.model_color(m) for m in comp["model"]},
                hover_data={"val_pr_auc": ":.4f", "test_precision": ":.3f", "test_recall": ":.3f", "model": False})
    f1.update_layout(showlegend=False)
    f2 = go.Figure()
    f2.add_scatter(x=sweep["alerts_per_day"], y=sweep["precision"], name="Precision", line=dict(color=viz.SERIES[0]))
    f2.add_scatter(x=sweep["alerts_per_day"], y=sweep["recall"], name="Recall", line=dict(color=viz.SERIES[1]))
    f2.update_layout(title="Champion: precision and recall by alert volume (validation)", xaxis_type="log",
                     xaxis_title="Alerts per day", yaxis_title="Rate")
    f3 = go.Figure()
    f3.add_scatter(x=monthly["period"], y=monthly["recall"], name="Recall", mode="lines+markers",
                   line=dict(color=viz.SERIES[0]))
    f3.add_scatter(x=monthly["period"], y=monthly["precision"], name="Precision", mode="lines+markers",
                   line=dict(color=viz.SERIES[1]))
    f3.update_layout(title="Champion performance by month (test period)", yaxis_range=[0, 1.05])
    return _page("FraudLens - model explorer", f"Champion: {viz.MODEL_LABEL.get(results['champion'])}.",
                 [f1, f2, f3], cfg["paths"]["interactive"] / "model_explorer.html")
