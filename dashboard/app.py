"""FraudLens transaction monitoring dashboard.

Run from the project root:   streamlit run dashboard/app.py
Requires the pipeline outputs (python -m fraud_pipeline.run_all).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from fraud_pipeline import dashboard_data as D  # noqa: E402
from fraud_pipeline.utils import load_config  # noqa: E402

COLORS = {"caught": "#2E7D32", "false_alert": "#EF6C00", "missed": "#C62828", "cleared": "#90A4AE"}
LABELS = {"caught": "Fraud caught", "false_alert": "False alert", "missed": "Fraud missed", "cleared": "Cleared"}

st.set_page_config(page_title="FraudLens", page_icon="🛡️", layout="wide")


# ------------------------------------------------------------------ helpers
def show_chart(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:  # older Streamlit versions
        st.plotly_chart(fig, use_container_width=True)


def show_table(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except Exception:  # older Streamlit versions
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def money(x):
    return f"${x:,.0f}"


@st.cache_data(show_spinner="Loading scored transactions...")
def get_data(root: str) -> dict:
    return D.load_dashboard(load_config(root=Path(root)))


# --------------------------------------------------------------------- data
try:
    data = get_data(str(ROOT))
except FileNotFoundError:
    st.error("Dashboard data not found. From the project root, run:  `python -m fraud_pipeline.run_all`")
    st.stop()

df_all, meta, results = data["df"], data["meta"], data["results"]
risk = data["category_risk_train"]
thr_opts = meta["thresholds_from_validation"]

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🛡️ FraudLens")
st.sidebar.caption(f"Model: **{meta['champion']}**")

period = st.sidebar.radio("Monitoring period", ["Test (Jun to Dec 2020)", "Validation (May to Jun 2020)"])
split = "test" if period.startswith("Test") else "val"
base = df_all[df_all["split"] == split]
dmin, dmax = base["ts"].min().date(), base["ts"].max().date()
dates = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
start, end = dates if isinstance(dates, (list, tuple)) and len(dates) == 2 else (dmin, dmax)
categories = st.sidebar.multiselect("Merchant category", sorted(base["category"].unique()))
states = st.sidebar.multiselect("State", sorted(base["state"].unique()))

st.sidebar.divider()
st.sidebar.subheader("Operating point")
presets = {
    f"Model default ({meta['threshold_strategy']})": meta["threshold"],
    "Maximise F1": thr_opts["max_f1"],
    f"Alert budget ({meta['alerts_per_day_budget']}/day)": thr_opts["alert_budget"],
    "Custom": None,
}
choice = st.sidebar.selectbox("Threshold preset", list(presets))
if presets[choice] is None:
    threshold = st.sidebar.slider("Custom threshold", 0.0, 1.0, float(meta["threshold"]), 0.001, format="%.3f")
else:
    threshold = float(presets[choice])
    st.sidebar.caption(f"Threshold = {threshold:.4f} (chosen on validation)")
cost = st.sidebar.number_input("Review cost per alert ($)", 0.0, 1000.0, float(meta["investigation_cost"]), 1.0)
recovery = st.sidebar.slider("Recovery rate on caught fraud", 0.0, 1.0, float(meta["recovery_rate"]), 0.05)

df = D.apply_filters(df_all, split, start, end, categories, states)
k = D.kpis(df, threshold, cost, recovery)

# ------------------------------------------------------------------- header
st.title("Transaction Monitoring")
st.caption(f"{period.split(' (')[0]} period, {start} to {end} · {len(df):,} transactions · threshold {threshold:.4f} · "
           "historical replay: the true outcome of every alert is known")
if df.empty or not k:
    st.warning("No transactions match the current filters.")
    st.stop()

tabs = st.tabs(["Overview", "Alert queue", "Threshold & cost", "Patterns", "Lookup", "Model & data health", "Report"])

# ----------------------------------------------------------------- overview
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Transactions", f"{k['transactions']:,}")
    c[1].metric("Alerts", f"{k['alerts']:,}", help=f"{k['alerts_per_day']} per day")
    c[2].metric("Cards to review", f"{k['cards_to_review']:,}", help=f"{k['alerts_per_card']} alerts per card on average")
    c[3].metric("Frauds in period", f"{k['frauds']:,}")
    c = st.columns(4)
    c[0].metric("Precision", f"{k['precision']:.1%}", help="Share of alerts that are fraud")
    c[1].metric("Recall", f"{k['recall']:.1%}", help="Share of fraud that raised an alert")
    c[2].metric("False alarms per catch", k["false_alarms_per_catch"] if k["false_alarms_per_catch"] is not None else "n/a")
    c[3].metric("Net benefit", money(k["net_benefit"]), help="Recovered fraud minus review cost")
    c = st.columns(3)
    c[0].metric("Fraud value recovered", money(k["fraud_amount_caught"]))
    c[1].metric("Fraud value missed", money(k["fraud_amount_missed"]))
    c[2].metric("Review cost", money(k["review_cost"]))

    d = D.daily(df, threshold)
    fig = go.Figure()
    for col in ["caught", "false_alert"]:
        fig.add_bar(x=d["date"], y=d[col], name=LABELS[col], marker_color=COLORS[col])
    fig.add_scatter(x=d["date"], y=d["missed"], name=LABELS["missed"], mode="lines",
                    line=dict(color=COLORS["missed"], width=2))
    fig.update_layout(barmode="stack", title="Daily alerts and missed fraud", height=380,
                      legend=dict(orientation="h", y=-0.2), margin=dict(t=50, b=10))
    show_chart(fig)

# -------------------------------------------------------------- alert queue
with tabs[1]:
    q = D.card_queue(df, threshold)
    st.markdown(f"**{len(q):,} cards** account for **{k['alerts']:,} alerts**. Fraud on a card arrives in bursts "
                f"of about two days, so reviewing by card cuts the queue by "
                f"**{1 - len(q) / max(k['alerts'], 1):.0%}**.")
    show_table(q, column_config={
        "max_score": st.column_config.ProgressColumn("Max score", min_value=0.0, max_value=1.0, format="%.3f"),
        "alerted_amount": st.column_config.NumberColumn("Alerted amount", format="$%.2f"),
        "first_alert": st.column_config.DatetimeColumn("First alert", format="YYYY-MM-DD HH:mm"),
        "last_alert": st.column_config.DatetimeColumn("Last alert", format="YYYY-MM-DD HH:mm"),
    }, height=320)

    if len(q):
        card = st.selectbox("Investigate a card", q["card_id"].head(500), key="queue_card")
        tl = D.card_timeline(df, card, threshold, risk)
        fig = px.scatter(tl, x="ts", y="score", color="outcome", size="amount", size_max=18,
                         color_discrete_map=COLORS, hover_data=["merchant", "category", "amount"],
                         title=f"Card {card}: every transaction in the period")
        fig.add_hline(y=threshold, line_dash="dash", annotation_text="threshold")
        fig.update_layout(height=340, margin=dict(t=50, b=10))
        show_chart(fig)
        show_table(tl.loc[tl["score"] >= threshold,
                          ["ts", "merchant", "category", "amount", "score", "outcome", "reasons"]])

        alerts_csv = df[df["score"] >= threshold].assign(outcome=lambda x: D.outcomes(x, threshold))
        st.download_button("Download all alerts (CSV)", alerts_csv.to_csv(index=False).encode(),
                           file_name=f"fraudlens_alerts_{split}.csv", mime="text/csv")

# --------------------------------------------------------- threshold & cost
with tabs[2]:
    sw = D.sweep(df, cost, recovery)
    if sw.empty:
        st.info("No fraud in the filtered data, so the trade-off cannot be drawn.")
    else:
        c = st.columns(2)
        fig = go.Figure()
        fig.add_scatter(x=sw["alerts_per_day"], y=sw["precision"], name="Precision", line=dict(color="#C62828"))
        fig.add_scatter(x=sw["alerts_per_day"], y=sw["recall"], name="Recall", line=dict(color="#1565C0"))
        # on a log axis Plotly places shapes in log10 units
        current_x = math.log10(max(k["alerts_per_day"], 1e-3))
        fig.add_vline(x=current_x, line_dash="dot", annotation_text="current")
        fig.update_layout(title="Precision and recall by alert volume", xaxis_type="log",
                          xaxis_title="Alerts per day", height=360, margin=dict(t=50, b=10))
        with c[0]:
            show_chart(fig)
        fig = px.line(sw, x="alerts_per_day", y="net_benefit", log_x=True, title="Net benefit by alert volume",
                      labels={"alerts_per_day": "Alerts per day", "net_benefit": "Net benefit ($)"})
        fig.add_vline(x=current_x, line_dash="dot", annotation_text="current")
        fig.add_hline(y=0, line_color="grey")
        fig.update_layout(height=360, margin=dict(t=50, b=10))
        with c[1]:
            show_chart(fig)

    rows = []
    for name, t in [(f"Model default ({meta['threshold_strategy']})", meta["threshold"]),
                    ("Maximise F1", thr_opts["max_f1"]), ("Alert budget", thr_opts["alert_budget"])]:
        kk = D.kpis(df, t, cost, recovery)
        rows.append({"Operating point": name, "Threshold": round(t, 4), "Alerts/day": kk["alerts_per_day"],
                     "Precision": f"{kk['precision']:.1%}", "Recall": f"{kk['recall']:.1%}",
                     "False alarms per catch": kk["false_alarms_per_catch"],
                     "Fraud missed": money(kk["fraud_amount_missed"]), "Net benefit": money(kk["net_benefit"])})
    show_table(pd.DataFrame(rows))
    be = D.break_even_cost(df, thr_opts["alert_budget"], meta["threshold"], recovery)
    if be is not None:
        st.info(f"Moving from the model default to the alert-budget threshold pays for itself only if a review "
                f"costs less than **${be:,.2f}**. Current assumption: ${cost:,.2f}.")

# ----------------------------------------------------------------- patterns
with tabs[3]:
    seg = D.segment(df, threshold, "category").sort_values("alerts")
    fig = go.Figure()
    for col in ["caught", "false_alert", "missed"]:
        fig.add_bar(y=seg["category"], x=seg[col], name=LABELS[col], orientation="h", marker_color=COLORS[col])
    fig.update_layout(barmode="stack", title="Outcomes by merchant category", height=460,
                      legend=dict(orientation="h", y=-0.15), margin=dict(t=50, b=10))
    show_chart(fig)

    c = st.columns(2)
    hr = D.segment(df, threshold, "hour")
    fig = go.Figure()
    fig.add_bar(x=hr["hour"], y=hr["alerts"], name="Alerts", marker_color="#90A4AE")
    fig.add_scatter(x=hr["hour"], y=hr["fraud_rate_pct"], name="Fraud rate (%)", yaxis="y2",
                    line=dict(color=COLORS["missed"]))
    fig.update_layout(title="Alerts and fraud rate by hour", height=360, margin=dict(t=50, b=10),
                      yaxis2=dict(overlaying="y", side="right", title="Fraud rate (%)"),
                      legend=dict(orientation="h", y=-0.2))
    with c[0]:
        show_chart(fig)
    amt = D.segment(df, threshold, "amount_band")
    fig = px.bar(amt, x="amount_band", y=["caught", "false_alert", "missed"], title="Outcomes by amount",
                 color_discrete_map=COLORS, labels={"value": "Transactions", "amount_band": "Amount"})
    fig.update_layout(height=360, margin=dict(t=50, b=10), legend=dict(orientation="h", y=-0.2))
    with c[1]:
        show_chart(fig)

    alerted = df[df["score"] >= threshold].assign(outcome=lambda x: D.outcomes(x, threshold))
    if len(alerted):
        fig = px.scatter_geo(alerted.head(5000), lat="lat", lon="lon", color="outcome", scope="usa",
                             color_discrete_map=COLORS, hover_data=["card_id", "amount", "category"],
                             title="Location of alerted cardholders")
        fig.update_layout(height=420, margin=dict(t=50, b=10))
        show_chart(fig)

# ------------------------------------------------------------------- lookup
with tabs[4]:
    query = st.text_input("Transaction ID or card ID", placeholder="Paste an ID from the alert queue")
    if query:
        q = query.strip()
        if q in set(df["transaction_id"]):
            row = df[df["transaction_id"] == q].iloc[0]
            c = st.columns(4)
            c[0].metric("Score", f"{row['score']:.4f}")
            c[1].metric("Amount", f"${row['amount']:,.2f}")
            c[2].metric("Decision", "ALERT" if row["score"] >= threshold else "Cleared")
            c[3].metric("True label", "Fraud" if row["is_fraud"] == 1 else "Legitimate")
            st.markdown(f"**{row['merchant']}** · {row['category']} · {row['ts']} · card `{row['card_id']}`")
            for r in D.alert_reasons(row, risk):
                st.markdown(f"- {r}")
            q = row["card_id"]
        if q in set(df["card_id"]):
            show_table(D.card_timeline(df, q, threshold, risk)[
                ["ts", "merchant", "category", "amount", "score", "outcome", "reasons"]])
        elif q not in set(df["transaction_id"]):
            st.warning("No transaction or card with that ID in the filtered period.")

# ----------------------------------------------------- model & data health
with tabs[5]:
    sel = results.get("selection", {})
    st.subheader(f"Champion: {meta['champion']}")
    if sel:
        st.info(sel["reason"])
        comp = pd.DataFrame(sel["comparisons"]).T.reset_index().rename(columns={"index": "model"})
        st.caption(f"Rule: {sel['rule']}. Margin = {sel.get('margin')}.")
        show_table(comp)
    show_table(pd.DataFrame(results["comparison"]))
    if results.get("imbalance_handling"):
        st.markdown("**Class imbalance handling (read from the fitted models)**")
        show_table(pd.DataFrame(list(results["imbalance_handling"].items()), columns=["model", "method"]))

    pre = data.get("preprocessing") or {}
    if pre.get("validation_drift_psi"):
        st.markdown("**Feature drift, training to validation (PSI)**")
        drift = pd.DataFrame(pre["validation_drift_psi"]).T.reset_index().rename(columns={"index": "feature"})
        show_table(drift.sort_values("psi", ascending=False))

    ver = data.get("verification")
    if ver:
        (st.success if ver["status"] == "PASS" else st.error)(
            f"End-to-end verification: {ver['status']} ({ver['passed']} passed, {ver['failed']} failed)")
        with st.expander("All verification checks"):
            show_table(pd.DataFrame(ver["checks"])[["stage", "check", "status"]])

# ------------------------------------------------------------------- report
with tabs[6]:
    filters = {"split": split, "start": start, "end": end, "threshold": threshold,
               "categories": categories, "states": states, "cost": cost, "recovery": recovery}
    cat_seg = D.segment(df, threshold, "category").sort_values("false_alert", ascending=False)
    md = D.summary_markdown(k, filters, meta, cat_seg)
    st.markdown(md)
    c = st.columns(2)
    c[0].download_button("Download summary (Markdown)", md.encode(), file_name="fraudlens_summary.md",
                         mime="text/markdown")
    c[1].download_button("Download category breakdown (CSV)", cat_seg.to_csv(index=False).encode(),
                         file_name="fraudlens_categories.csv", mime="text/csv")
