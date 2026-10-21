"""FraudLens - transaction monitoring and fraud scoring dashboard (Streamlit).

Local (full pipeline output):   streamlit run dashboard/app.py
Hosted (Streamlit Community Cloud): reads the compact deploy bundle in dashboard/bundle,
built with `python -m fraudlens.bundle`.
"""
from __future__ import annotations

import copy
import datetime as dt
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from fraudlens import dashboard_data as D  # noqa: E402
from fraudlens import security  # noqa: E402
from fraudlens.casework import get_store  # noqa: E402
from fraudlens.palette import (FRAUD, GRID, INK, INK_2, LEGIT, MODEL_LABEL, MUTED, OUTCOME_LABEL, SERIES,  # noqa: E402
                               STATUS, SURFACE, feature_label)

st.set_page_config(page_title="FraudLens", page_icon=":shield:", layout="wide")


# ------------------------------------------------------------ access control
def _configured_users() -> dict:
    spec = os.environ.get("FRAUDLENS_DASHBOARD_USERS")
    if not spec:
        try:
            spec = dict(st.secrets["users"]) if "users" in st.secrets else None
        except Exception:      # no secrets file
            spec = None
    return security.parse_users(spec)


USERS = _configured_users()
if USERS and not st.session_state.get("user"):
    st.title("FraudLens")
    st.caption("Sign in to review transactions. Viewers can explore; analysts can also record decisions.")
    with st.form("login"):
        name = st.text_input("User name")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary"):
            role = security.authenticate(USERS, name, password)
            if role:
                st.session_state.update(user=name, role=role)
                st.rerun()
            st.error("Unknown user name or wrong password.")
    st.stop()
USER = st.session_state.get("user") or "analyst-1"
ROLE = st.session_state.get("role") or ("analyst" if not USERS else "viewer")
CAN_DECIDE = ROLE in ("analyst", "admin")

STATUS_ICON = {"caught": "✅", "false_alert": "⚠️", "missed": "❌", "cleared": "▫️"}
CATEGORIES = ["entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos", "health_fitness",
              "home", "kids_pets", "misc_net", "misc_pos", "personal_care", "shopping_net", "shopping_pos", "travel"]


# ------------------------------------------------------------------ helpers
def layout(fig: go.Figure, title: str | None = None, height: int = 360, legend_bottom: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=INK)) if title else None,
        height=height, margin=dict(t=50 if title else 20, b=10, l=10, r=10),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=INK_2, size=12),
        colorway=SERIES, hoverlabel=dict(bgcolor="white"),
        legend=dict(orientation="h", y=-0.2, x=0) if legend_bottom else None)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def chart(fig):
    try:
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def table(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def image(path: Path, caption: str | None = None):
    if path.exists():
        try:
            st.image(str(path), caption=caption, width="stretch")
        except TypeError:
            st.image(str(path), caption=caption, use_container_width=True)


def money(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"${x:,.0f}"


def md_money(x) -> str:
    """money() for st.markdown, where a bare $ starts LaTeX maths."""
    return money(x).replace("$", "\\$")


def pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1%}"


@st.cache_data(show_spinner="Loading scored transactions...")
def get_data(root: str, prefer: str | None) -> dict:
    return D.load_dashboard(Path(root), prefer)


@st.cache_resource(show_spinner="Loading the champion model and card history...")
def get_runtime(source: str):
    from fraudlens.runtime import load_runtime
    return load_runtime(Path(source))


def session_scorer(rt):
    """Each browser session scores against its own copy of the card store, so one user's
    test transactions never change another user's card history."""
    if "store" not in st.session_state:
        st.session_state["store"] = copy.deepcopy(rt.store)
    from fraudlens.scoring import FraudScorer
    return FraudScorer(rt.model, st.session_state["store"], rt.scorer.category_risk, rt.scorer.explainer)


# --------------------------------------------------------------------- data
try:
    data = get_data(str(ROOT), os.environ.get("FRAUDLENS_DASHBOARD_SOURCE"))
except FileNotFoundError:
    st.error("No scored transactions found. Run `python -m fraudlens.run_all` (full pipeline) "
             "or `python -m fraudlens.bundle` (deploy bundle) from the project root.")
    st.stop()

df_all, meta, results = data["df"], data["meta"] or {}, data["results"] or {}
risk = data["category_risk_train"]
thr_opts = meta.get("thresholds_from_validation", {})
champion = meta.get("champion", "?")

# ------------------------------------------------------------------ sidebar
st.sidebar.title("FraudLens")
st.sidebar.caption(f"Champion model: **{MODEL_LABEL.get(champion, champion)}**  \n"
                   f"Data source: {'deploy bundle' if data['kind'] == 'bundle' else 'full pipeline output'}")
if USERS:
    st.sidebar.caption(f"Signed in as **{USER}** ({ROLE})")
    if st.sidebar.button("Sign out"):
        for kk in ("user", "role"):
            st.session_state.pop(kk, None)
        st.rerun()
else:
    st.sidebar.caption("Access control is off (development). Set FRAUDLENS_DASHBOARD_USERS to require sign-in.")

split = st.sidebar.radio("Period", ["test", "val"], format_func=lambda s: {
    "test": "Test (21 Jun to 31 Dec 2020)", "val": "Validation (11 May to 20 Jun 2020)"}[s])
base = df_all[df_all["split"] == split]
dmin, dmax = base["ts"].min().date(), base["ts"].max().date()
dates = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
start, end = dates if isinstance(dates, (list, tuple)) and len(dates) == 2 else (dmin, dmax)
categories = st.sidebar.multiselect("Merchant category", sorted(base["category"].unique()))
states = st.sidebar.multiselect("Cardholder state", sorted(base["state"].unique()))

st.sidebar.divider()
st.sidebar.subheader("Operating point")
presets = {f"Model default ({meta.get('threshold_strategy', 'min_cost')})": meta.get("threshold", 0.5)}
if "max_f1" in thr_opts:
    presets["Maximise F1"] = thr_opts["max_f1"]
if "alert_budget" in thr_opts:
    presets[f"Alert budget ({meta.get('alerts_per_day_budget', 50)}/day)"] = thr_opts["alert_budget"]
presets["Custom"] = None
choice = st.sidebar.selectbox("Threshold preset", list(presets))
if presets[choice] is None:
    threshold = st.sidebar.slider("Custom threshold", 0.0, 1.0, float(meta.get("threshold", 0.5)), 0.001,
                                  format="%.3f")
else:
    threshold = float(presets[choice])
    st.sidebar.caption(f"Threshold {threshold:.4f}, chosen on the validation period")
st.sidebar.divider()
st.sidebar.subheader("Cost assumptions")
st.sidebar.caption("Used only for the estimated financial figures; they are assumptions, not measurements.")
cost = st.sidebar.number_input("Review cost per alert ($)", 0.0, 1000.0, float(meta.get("investigation_cost", 15)), 1.0)
recovery = st.sidebar.slider("Share of caught fraud recovered", 0.0, 1.0, float(meta.get("recovery_rate", 0.9)), 0.05)

df = D.apply_filters(df_all, split, start, end, categories, states)
k = D.kpis(df, threshold, cost, recovery)

# ------------------------------------------------------------------- header
st.title("Transaction monitoring")
st.caption(f"{'Test' if split == 'test' else 'Validation'} period {start} to {end} · {len(df):,} transactions · "
           f"threshold {threshold:.4f} · historical replay, so the true outcome of every alert is known")
if df.empty or not k:
    st.warning("No transactions match the current filters.")
    st.stop()

tabs = st.tabs(["Overview", "Live transaction", "Case review", "Alert history", "Threshold & cost", "Patterns",
                "Monitoring", "Model", "Data pipeline", "Report"])
CASES = get_store(os.environ.get("FRAUDLENS_CASE_DB"))

# ================================================================= overview
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Transactions", f"{k['transactions']:,}")
    c[1].metric("Alerts raised", f"{k['alerts']:,}", help=f"{k['alerts_per_day']} per day")
    c[2].metric("Cards to review", f"{k['cards_to_review']:,}",
                help=f"{k['alerts_per_card']} alerts per card on average")
    c[3].metric("Frauds in period", f"{k['frauds']:,}")
    c = st.columns(4)
    c[0].metric("Precision", pct(k["precision"]), help="Share of alerts that were fraud")
    c[1].metric("Recall", pct(k["recall"]), help="Share of frauds that raised an alert")
    c[2].metric("False alarms per catch", k["false_alarms_per_catch"] if k["false_alarms_per_catch"] is not None else "n/a")
    c[3].metric("Estimated net benefit", money(k["net_benefit"]),
                help=f"An estimate, not realised savings: value of caught fraud x {recovery:.0%} assumed recovered, "
                     f"minus {k['alerts']:,} alerts x ${cost:,.2f} assumed review cost")
    st.caption(f"Financial figures are **estimates** under the sidebar assumptions (review cost \\${cost:,.2f} per "
               f"alert, {recovery:.0%} of caught fraud value recovered) on simulated transactions; real savings "
               "would have to be measured in a trial on real transactions.")

    fraud_value = float(df.loc[df["is_fraud"] == 1, "amount"].sum())
    caught_value = float(df.loc[(df["is_fraud"] == 1) & (df["score"] >= threshold), "amount"].sum())
    caught_share = caught_value / max(fraud_value, 1)
    seg_cat = D.segment(df, threshold, "category")
    worst = seg_cat.sort_values("false_alert", ascending=False).iloc[0]
    hour_seg = D.segment(df, threshold, "hour")
    peak = hour_seg.sort_values("fraud_rate_pct", ascending=False).head(3)["hour"].astype(int).tolist()
    st.markdown(
        f"**What this period says.** The model flagged **{k['alerts_per_day']} transactions a day** and "
        f"caught **{caught_share:.1%} of the fraud value** ({md_money(caught_value)} of {md_money(fraud_value)}; "
        f"if {recovery:.0%} of it is recovered, {md_money(k['fraud_amount_caught'])}). At the assumed review cost the "
        f"reviews would cost {md_money(k['review_cost'])}, so the **estimated** net benefit is "
        f"**{md_money(k['net_benefit'])}**. Most false alerts come from **{worst['category']}** "
        f"({int(worst['false_alert'])}); fraud is most concentrated at **{', '.join(f'{h:02d}:00' for h in peak)}**.")

    d = D.daily(df, threshold)
    fig = go.Figure()
    for col in ["caught", "false_alert"]:
        fig.add_bar(x=d["date"], y=d[col], name=OUTCOME_LABEL[col], marker_color=STATUS[col],
                    marker_line=dict(color=SURFACE, width=1))
    fig.add_scatter(x=d["date"], y=d["missed"], name=OUTCOME_LABEL["missed"], mode="lines+markers",
                    line=dict(color=STATUS["missed"], width=2), marker=dict(size=5))
    fig.update_layout(barmode="stack", yaxis_title="Transactions")
    chart(layout(fig, "Daily alerts (bars) and missed fraud (line)", 380))

# ============================================================ alert history
with tabs[3]:
    st.caption("Historical replay of the test period: every alert's true outcome is known. Decisions recorded "
               "here go to the same case store as live alerts (marked as replay).")
    q = D.card_queue(df, threshold)
    st.markdown(f"**{len(q):,} cards** account for **{k['alerts']:,} alerts**. Fraud on a card arrives in short "
                f"bursts, so reviewing by card instead of by transaction cuts the queue by "
                f"**{1 - len(q) / max(k['alerts'], 1):.0%}**.")
    table(q, column_config={
        "max_score": st.column_config.ProgressColumn("Max score", min_value=0.0, max_value=1.0, format="%.3f"),
        "alerted_amount": st.column_config.NumberColumn("Alerted amount", format="$%.2f"),
        "first_alert": st.column_config.DatetimeColumn("First alert", format="YYYY-MM-DD HH:mm"),
        "last_alert": st.column_config.DatetimeColumn("Last alert", format="YYYY-MM-DD HH:mm"),
    }, height=300)

    if len(q):
        card = st.selectbox("Investigate a card", q["card_id"].head(500), key="queue_card")
        tl = D.card_timeline(df, card, threshold, risk)
        tl["outcome_label"] = tl["outcome"].map(OUTCOME_LABEL)
        fig = px.scatter(tl, x="ts", y="score", color="outcome_label", size="amount", size_max=18,
                         color_discrete_map={OUTCOME_LABEL[o]: STATUS[o] for o in STATUS},
                         hover_data={"merchant": True, "category": True, "amount": ":$,.2f", "outcome_label": False},
                         labels={"ts": "", "score": "Fraud score", "outcome_label": "Outcome"})
        fig.add_hline(y=threshold, line_dash="dash", line_color=MUTED, annotation_text="threshold")
        chart(layout(fig, f"Card {card}: every transaction in the period", 340))
        cols = ["ts", "merchant", "category", "amount", "score", "outcome", "reasons"]
        if "shap_reasons" in tl:
            cols.append("shap_reasons")
        alerted = tl.loc[tl["score"] >= threshold, cols].copy()
        alerted["outcome"] = alerted["outcome"].map(lambda o: f"{STATUS_ICON[o]} {OUTCOME_LABEL[o]}")
        st.markdown("**Alerts on this card**: evidence from the card's history, and the model's top SHAP "
                    "contributions (log-odds; positive pushes towards fraud)")
        table(alerted.rename(columns={"shap_reasons": "SHAP top contributions", "reasons": "Evidence"}),
              column_config={"amount": st.column_config.NumberColumn(format="$%.2f"),
                             "score": st.column_config.NumberColumn(format="%.3f")})

        with st.form("feedback"):
            st.markdown("**Analyst decision on this card's alerts** (stored in the case store with your name)")
            fc = st.columns([2, 4, 1])
            verdict = fc[0].radio("Verdict", ["Confirmed fraud", "Legitimate"], horizontal=True)
            note = fc[1].text_input("Note", placeholder="e.g. customer confirmed by phone")
            if fc[2].form_submit_button("Record", disabled=not CAN_DECIDE):
                label = "fraud" if verdict == "Confirmed fraud" else "legitimate"
                n = 0
                for _, r in tl[tl["score"] >= threshold].iterrows():
                    CASES.decide_transaction(str(r["transaction_id"]), label, USER, note or None, role=ROLE, extra={
                        "source": "replay", "card_id": card, "score": round(float(r["score"]), 6), "threshold": threshold,
                        "ts": str(r["ts"]), "amount": float(r["amount"]), "category": r["category"],
                        "reasons": [x for x in str(r.get("reasons", "")).split("; ") if x], "model": champion})
                    n += 1
                st.success(f"Recorded {verdict.lower()} for {n} alerts on card {card} as {USER}.")
        if not CAN_DECIDE:
            st.caption("Signed in as a viewer: decisions are recorded by analysts only.")

        alerts_csv = df[df["score"] >= threshold].assign(outcome=lambda x: D.outcomes(x, threshold))
        st.download_button("Download every alert in the period (CSV)", alerts_csv.to_csv(index=False).encode(),
                           file_name=f"fraudlens_alerts_{split}.csv", mime="text/csv")

# ========================================================= live transaction
with tabs[1]:
    st.markdown("**A new transaction, end to end.** Enter a transaction as the payment system would send it. "
                "FraudLens then (1) receives it, (2) generates its features from the card's history in the online "
                "feature store, (3) scores it with the champion model and explains the score, and (4) routes it: "
                "an alert opens a case in the **Case review** queue, anything else is approved. This is the same "
                "code path as the scoring API (`POST /score`). Card-history changes stay in your session.")
    try:
        rt = get_runtime(str(data["source"]))
    except Exception as exc:  # pragma: no cover - shown in the UI
        st.error(f"Live scoring unavailable: {exc}")
        rt = None
    if rt is not None:
        scorer = session_scorer(rt)
        store = st.session_state["store"]
        known = sorted(set(q["card_id"].head(30)) | set(list(store.cards)[:30])) if len(q) else list(store.cards)[:60]
        lc = st.columns([2, 1, 1])
        card_choice = lc[0].selectbox("Card", ["(new card, no history)"] + known,
                                      help="Cards from the alert queue first, then others from the store")
        card_id = "new-card-0001" if card_choice.startswith("(new") else card_choice
        stc = store.cards.get(card_id)
        last = pd.Timestamp(stc.last_t, unit="s") if stc and stc.last_t else pd.Timestamp("2020-12-31 12:00")
        if stc:
            lc[1].metric("Card's transactions so far", f"{stc.n:,}")
            lc[2].metric("Card's average amount", f"${stc.total / max(stc.n, 1):,.2f}")
        with st.form("score"):
            fc = st.columns(4)
            amount = fc[0].number_input("Amount ($)", 0.01, 50_000.0, 875.0, 1.0)
            category = fc[1].selectbox("Category", CATEGORIES, index=CATEGORIES.index("shopping_net"))
            merchant = fc[2].text_input("Merchant", "Example Store")
            minutes = fc[3].number_input("Minutes after the card's last transaction", 0, 100_000, 30, 5)
            add = st.checkbox("Add this transaction to the card's history after scoring", value=True)
            submitted = st.form_submit_button("Score transaction", type="primary")
        burst = st.button("Simulate a card-testing burst (6 rising amounts, 3 minutes apart)")

        def run_one(amt, cat, mer, ts, update):
            txn = {"card_id": card_id, "ts": ts.isoformat(), "amount": float(amt), "category": cat, "merchant": mer,
                   "transaction_id": f"live-{dt.datetime.now():%H%M%S%f}"}
            prior = store.cards.get(card_id)
            history = {"prior_txns": prior.n if prior else 0,
                       "card_mean": (prior.total / prior.n) if prior and prior.n else None}
            out = scorer.score_transaction(txn, update_state=update)
            out["transaction_id"] = txn["transaction_id"]
            out["case_id"] = CASES.record_scored(out, txn, source="dashboard", actor=USER, role=ROLE)
            out["_txn"], out["_history"] = txn, history
            return out

        if submitted:
            ts = last + pd.Timedelta(minutes=int(minutes))
            res = run_one(amount, category, merchant, ts, add)
            st.session_state["last_score"] = res
        if burst:
            rows = []
            t0 = (pd.Timestamp(store.cards[card_id].last_t, unit="s") if card_id in store.cards and store.cards[card_id].last_t
                  else last)
            for i, amt in enumerate([2.0, 5.0, 25.0, 180.0, 650.0, 1200.0]):
                r = run_one(amt, "shopping_net" if i > 2 else "misc_net", "Burst Merchant", t0 + pd.Timedelta(minutes=3 * (i + 1)), True)
                rows.append({"step": i + 1, "amount": amt, "score": r["score"], "alert": r["alert"],
                             "txns past 1h": r["features"]["txn_count_1h"],
                             "spend past 24h": r["features"]["amt_sum_24h"]})
            st.session_state["burst"] = pd.DataFrame(rows)
            st.session_state["last_score"] = r

        res = st.session_state.get("last_score")
        if res:
            txn, hist = res["_txn"], res["_history"]
            f = res["features"]
            s1, s2 = st.columns(2)
            with s1.container(border=True):
                st.markdown("**1 · Transaction received**")
                table(pd.DataFrame({"field": list(txn), "value": [str(v) for v in txn.values()]}))
            with s2.container(border=True):
                st.markdown("**2 · Features generated from the card's history** (online feature store)")
                mins = f.get("secs_since_last_txn")
                feat_rows = [
                    ("Card's earlier transactions", f"{hist['prior_txns']:,}"),
                    ("Card's average amount", money(hist["card_mean"]) if hist["card_mean"] else "no history"),
                    ("Amount vs the card's average", f"{f['amt_to_card_mean']:.1f}x" if f.get("amt_to_card_mean") else "n/a"),
                    ("Amount z-score for this card", f"{f['amt_zscore_card']:.1f}" if f.get("amt_zscore_card") else "n/a"),
                    ("Minutes since the card's last transaction", f"{mins / 60:,.0f}" if mins is not None else "first"),
                    ("Transactions in the past 1 h / 24 h / 7 d",
                     f"{f['txn_count_1h']} / {f['txn_count_24h']} / {f['txn_count_7d']}"),
                    ("Spend in the past 24 h", money(f["amt_sum_24h"])),
                    ("New merchant for this card", "yes" if f.get("is_new_merchant_for_card") else "no"),
                    ("Night-time (22:00-03:59)", "yes" if f.get("is_night") else "no"),
                ]
                table(pd.DataFrame(feat_rows, columns=["feature", "value"]))
            s3, s4 = st.columns(2)
            with s3.container(border=True):
                st.markdown("**3 · Model inference and explanation**")
                rc = st.columns(3)
                rc[0].metric("Fraud score", f"{res['score']:.4f}")
                rc[1].metric("Threshold", f"{res['threshold']:.4f}")
                rc[2].metric("Card known", "Yes" if res["known_card"] else "No")
                contrib = res.get("contributions")
                if contrib:
                    cdf = pd.DataFrame(contrib)
                    cdf["label"] = cdf["feature"].map(feature_label)
                    cdf["direction"] = np.where(cdf["shap"] > 0, "Towards fraud", "Towards legitimate")
                    fig = px.bar(cdf.iloc[::-1], x="shap", y="label", orientation="h", color="direction",
                                 color_discrete_map={"Towards fraud": FRAUD, "Towards legitimate": LEGIT},
                                 hover_data={"value": True, "direction": False},
                                 labels={"shap": "SHAP contribution (log-odds)", "label": ""})
                    fig.update_layout(legend=dict(orientation="h", y=1.18, x=0, title=None))
                    chart(layout(fig, "Top contributions to this score", 280, legend_bottom=False))
                if res["alert"]:
                    cb = res.get("clear_below_amount")
                    st.caption("Counterfactual: " + (f"with the same card history and time, this transaction would "
                               f"have been approved at **\\${cb:,.2f}**." if cb else
                               "even at \\$1 it would still alert, so the pattern (timing, velocity, category), "
                               "not the amount, drives this alert."))
            with s4.container(border=True):
                st.markdown("**4 · Decision and routing**")
                if res["alert"]:
                    st.error(f"**ALERT: sent to review.** Case **{res['case_id']}** is open in the **Case review** tab.")
                    st.markdown("**Evidence for the analyst:** " + "; ".join(res["reasons"]))
                else:
                    st.success(f"**Approved** (score below the threshold). Logged as case {res['case_id']}; if the "
                               "customer later disputes it, the case can still be labelled as fraud.")
                st.caption("Every scored transaction is stored with its features, score, threshold and explanation, "
                           "so the analyst sees exactly what the model saw, and the decision becomes a label.")
            with st.expander("All model inputs for this transaction"):
                feats = {feature_label(kk): v for kk, v in res["features"].items()}
                table(pd.DataFrame({"feature": list(feats), "value": [str(v) for v in feats.values()]}))
        if "burst" in st.session_state:
            b = st.session_state["burst"]
            fig = px.line(b, x="step", y="score", markers=True, labels={"step": "Transaction in the burst",
                                                                        "score": "Fraud score"})
            fig.update_traces(line_color=SERIES[0], marker=dict(size=9))
            fig.add_hline(y=rt.model.threshold, line_dash="dash", line_color=MUTED, annotation_text="threshold")
            chart(layout(fig, "Score as the burst builds up (velocity features rise)", 300, legend_bottom=False))
            table(b, column_config={"amount": st.column_config.NumberColumn(format="$%.2f"),
                                    "spend past 24h": st.column_config.NumberColumn(format="$%.2f")})
        if st.button("Reset my session's card history"):
            st.session_state.pop("store", None)
            st.session_state.pop("last_score", None)
            st.session_state.pop("burst", None)
            st.rerun()

# ============================================================== case review
with tabs[2]:
    st.markdown("**The analyst workflow.** Alerts from new transactions (this dashboard's *Live transaction* tab, "
                "or the scoring API) wait here as open cases. The analyst opens a case, sees what the model saw, "
                "and records a decision. Each decision closes the case, is written to the audit log with the "
                "analyst's name, and becomes a label that monitoring and the next retraining use.")
    fb = CASES.feedback_summary(precision_floor=float((data.get("monitoring") or {}).get("policy", {})
                                                      .get("precision_floor", 0.4)))
    m = st.columns(5)
    m[0].metric("Open alerts", fb["open_alerts"],
                help="Oldest waiting " + (f"{fb['oldest_open_hours']:.1f} h" if fb["oldest_open_hours"] is not None else "-"))
    m[1].metric("Reviewed alerts", fb["reviewed_alerts"])
    lo, hi = fb["live_precision_ci95"]
    m[2].metric("Live precision", pct(fb["live_precision"]),
                help="Share of reviewed alerts the analysts confirmed as fraud" +
                     (f" (95% CI {lo:.0%}-{hi:.0%})" if lo is not None else ""))
    m[3].metric("Median time to decision", f"{fb['median_hours_to_decision']:.1f} h"
                if fb["median_hours_to_decision"] is not None else "-")
    m[4].metric("New labelled examples", fb["labelled_examples"])

    with st.expander("No live alerts yet? Generate a batch of incoming transactions (demo)"):
        st.caption("Scores 25 plausible next transactions for random known cards (a few with unusual amounts, "
                   "times or bursts), through the same online feature store and model, in your session.")
        if st.button("Generate incoming transactions"):
            from fraudlens.runtime import example_transaction  # noqa: F401  (same runtime as the API)
            try:
                rt_demo = get_runtime(str(data["source"]))
                sc = session_scorer(rt_demo)
                dstore = st.session_state["store"]
                rng = np.random.default_rng(int(dt.datetime.now().timestamp()))
                cards = [c for c, v in dstore.cards.items() if v.last_t is not None and v.n > 0]
                made = 0
                for i in range(25):
                    cid = cards[int(rng.integers(len(cards)))]
                    stt = dstore.cards[cid]
                    mean = stt.total / max(stt.n, 1)
                    unusual = i % 5 == 0
                    amt = round(float(mean * (rng.uniform(8, 20) if unusual else rng.lognormal(0, 0.5))), 2)
                    ts = pd.Timestamp(stt.last_t, unit="s") + pd.Timedelta(minutes=int(rng.integers(5, 60 * 30)))
                    if unusual:
                        ts = ts.normalize() + pd.Timedelta(hours=int(rng.choice([22, 23, 0, 1, 2])))
                    cat = "shopping_net" if unusual else CATEGORIES[int(rng.integers(len(CATEGORIES)))]
                    txn = {"card_id": cid, "ts": ts.isoformat(), "amount": max(amt, 1.0), "category": cat,
                           "merchant": "Demo Merchant", "transaction_id": f"demo-{i}-{rng.integers(1e9)}"}
                    out = sc.score_transaction(txn, update_state=True)
                    out["transaction_id"] = txn["transaction_id"]
                    CASES.record_scored(out, txn, source="dashboard-demo", actor=USER, role=ROLE)
                    made += int(out["alert"])
                st.session_state["demo_msg"] = f"Scored 25 transactions: {made} alerts opened."
            except Exception as exc:  # pragma: no cover - shown in the UI
                st.error(f"Could not score: {exc}")
            else:
                st.rerun()
        if st.session_state.get("demo_msg"):
            st.success(st.session_state.pop("demo_msg"))

    open_cases = CASES.cases("open", 200)
    if not open_cases:
        st.info("No open alerts. Score a transaction in **Live transaction** or generate a demo batch above.")
    else:
        qdf = pd.DataFrame([{"case": c["case_id"], "received": c["received_at"], "card": c["card_id"],
                             "amount": c["amount"], "category": c["category"], "score": c["score"],
                             "source": c["source"], "evidence": "; ".join(c["reasons"][:2])} for c in open_cases])
        table(qdf, column_config={"score": st.column_config.ProgressColumn("Score", min_value=0.0, max_value=1.0,
                                                                            format="%.3f"),
                                  "amount": st.column_config.NumberColumn(format="$%.2f")}, height=260)
        pick = st.selectbox("Open a case", qdf["case"], format_func=lambda cid: f"{cid} - card "
                            f"{qdf.set_index('case').loc[cid, 'card']}, ${qdf.set_index('case').loc[cid, 'amount']:,.2f}")
        case = CASES.get(pick)
        cc = st.columns([3, 2])
        with cc[0].container(border=True):
            st.markdown(f"**Case {case['case_id']}** · card {case['card_id']} · {case['category']} · "
                        f"{md_money(case['amount'])} · {case['ts']}")
            st.markdown(f"Score **{case['score']:.4f}** vs threshold {case['threshold']:.4f} (model {case['model']})")
            st.markdown("**Evidence:** " + ("; ".join(case["reasons"]) or "-"))
            if case.get("clear_below_amount"):
                st.caption(f"Counterfactual: would have been approved at \\${case['clear_below_amount']:,.2f} (same history and time).")
            if case["contributions"]:
                cdf = pd.DataFrame(case["contributions"])
                cdf["label"] = cdf["feature"].map(feature_label)
                fig = px.bar(cdf.iloc[::-1], x="shap", y="label", orientation="h",
                             color=np.where(cdf.iloc[::-1]["shap"] > 0, "Towards fraud", "Towards legitimate"),
                             color_discrete_map={"Towards fraud": FRAUD, "Towards legitimate": LEGIT},
                             labels={"shap": "SHAP contribution (log-odds)", "label": "", "color": ""})
                fig.update_layout(legend=dict(orientation="h", y=1.18, x=0, title=None))
                chart(layout(fig, "Why the model flagged it", 260, legend_bottom=False))
            with st.expander("Model inputs at scoring time"):
                table(pd.DataFrame({"feature": [feature_label(x) for x in case["features"]],
                                    "value": [str(v) for v in case["features"].values()]}))
        with cc[1].container(border=True):
            st.markdown("**Record the decision**")
            with st.form("case_decision"):
                verdict = st.radio("Verdict", ["Confirmed fraud", "Legitimate"], horizontal=True)
                note = st.text_input("Note", placeholder="e.g. card blocked after call with customer")
                st.caption(f"Recorded as **{USER}** ({ROLE}).")
                if st.form_submit_button("Record decision", type="primary", disabled=not CAN_DECIDE):
                    CASES.decide(pick, "fraud" if verdict == "Confirmed fraud" else "legitimate", USER,
                                 note or None, role=ROLE)
                    st.success(f"Case {pick} closed as {verdict.lower()}.")
                    st.rerun()
            if not CAN_DECIDE:
                st.caption("Viewers cannot record decisions.")

    st.markdown("#### How these decisions inform monitoring")
    verdict_box = {"ok": st.success, "review_threshold": st.error, "insufficient": st.info}[fb["verdict"]]
    verdict_box(f"**Live check: {fb['verdict'].replace('_', ' ')}.** {fb['reason']}.")
    st.markdown(
        f"1. **Live precision.** Confirmed fraud ÷ reviewed alerts is tracked with a 95% interval against the "
        f"policy floor ({fb['policy']['precision_floor']:.0%}). If the whole interval falls below the floor (with "
        f"at least {fb['policy']['min_decisions']} decisions), monitoring raises a **threshold/model review**: it is "
        f"added to the monitoring summary, exported to Prometheus (`fraudlens_live_precision`) and alerted in Grafana.\n"
        f"2. **Missed fraud.** A fraud label on an *approved* transaction (e.g. a customer dispute) counts as a miss "
        f"({fb['fraud_reported_on_approved']} so far), the live counterpart of recall.\n"
        f"3. **Workload.** The open backlog and time to decision show whether the alert volume is manageable.\n"
        f"4. **Retraining data.** Every decided case keeps the exact model inputs, so decisions become labelled "
        f"training examples ({fb['labelled_examples']} so far) for the next champion/challenger comparison.")
    decided = CASES.frame()
    decided = decided[decided["decision"].notna()]
    if len(decided):
        dd = decided.assign(day=pd.to_datetime(decided["decided_at"]).dt.date)
        daily_d = dd.groupby(["day", "decision"]).size().reset_index(name="decisions")
        fig = px.bar(daily_d, x="day", y="decisions", color="decision",
                     color_discrete_map={"fraud": FRAUD, "legitimate": LEGIT}, labels={"day": "", "decision": ""})
        chart(layout(fig, "Decisions per day", 260))
        show = decided.sort_values("decided_at", ascending=False)[["case_id", "decided_at", "analyst", "decision",
                                                                   "score", "amount", "source", "note"]]
        table(show.head(50))
        c2 = st.columns(2)
        c2[0].download_button("Download labelled examples for retraining (CSV)",
                              CASES.labelled_examples().to_csv(index=False).encode(),
                              file_name="fraudlens_labelled_feedback.csv", mime="text/csv")
        c2[1].download_button("Download the audit log (CSV)", CASES.audit_log(5000).to_csv(index=False).encode(),
                              file_name="fraudlens_audit_log.csv", mime="text/csv")

# ========================================================= threshold & cost
with tabs[4]:
    sw = D.sweep(df, cost, recovery)
    if sw.empty:
        st.info("No fraud in the filtered data, so the trade-off cannot be drawn.")
    else:
        current_x = max(k["alerts_per_day"], 1e-3)

        def mark_current(fig, y0, y1):
            # a dashed two-point trace instead of add_vline: vline shapes on log axes land in the wrong place
            fig.add_scatter(x=[current_x, current_x], y=[y0, y1], mode="lines+text", text=["", "current"],
                            textposition="top center", line=dict(color=MUTED, dash="dot", width=1.5),
                            showlegend=False, hoverinfo="skip")
        c = st.columns(2)
        fig = go.Figure()
        fig.add_scatter(x=sw["alerts_per_day"], y=sw["precision"], name="Precision", line=dict(color=SERIES[0], width=2))
        fig.add_scatter(x=sw["alerts_per_day"], y=sw["recall"], name="Recall", line=dict(color=SERIES[1], width=2))
        mark_current(fig, 0, 1)
        fig.update_layout(xaxis_type="log", xaxis_title="Alerts per day", yaxis_title="Rate", yaxis_range=[0, 1.05])
        with c[0]:
            chart(layout(fig, "Precision and recall by alert volume"))
        fig = go.Figure()
        fig.add_scatter(x=sw["alerts_per_day"], y=sw["net_benefit"], name="Estimated net benefit",
                        line=dict(color=SERIES[0], width=2))
        mark_current(fig, min(0, float(sw["net_benefit"].min())), float(sw["net_benefit"].max()))
        fig.add_hline(y=0, line_color=MUTED)
        fig.update_layout(xaxis_type="log", xaxis_title="Alerts per day", yaxis_title="Estimated net benefit ($)",
                          showlegend=False)
        with c[1]:
            chart(layout(fig, f"Estimated net benefit by alert volume (\\${cost:,.0f}/review, {recovery:.0%} recovered)"))

    rows = []
    for name, t in [(n, v) for n, v in presets.items() if v is not None]:
        kk = D.kpis(df, t, cost, recovery)
        rows.append({"Operating point": name, "Threshold": round(t, 4), "Alerts/day": kk["alerts_per_day"],
                     "Precision": pct(kk["precision"]), "Recall": pct(kk["recall"]),
                     "False alarms per catch": kk["false_alarms_per_catch"],
                     "Fraud missed": money(kk["fraud_amount_missed"]), "Estimated net benefit": money(kk["net_benefit"])})
    table(pd.DataFrame(rows))
    st.caption(f"Estimated net benefit = value of caught fraud x {recovery:.0%} assumed recovery - alerts x "
               f"\\${cost:,.2f} assumed review cost. Both assumptions are set in the sidebar; the result is an "
               "estimate for comparing operating points, not demonstrated savings.")
    if "alert_budget" in thr_opts:
        be = D.break_even_cost(df, thr_opts["alert_budget"], meta["threshold"], recovery)
        if be is not None:
            st.info(f"Lowering the threshold from the model default to the alert-budget point pays for itself only "
                    f"if a review costs less than **\\${be:,.2f}**. Current assumption: \\${cost:,.2f} per review.")

# ================================================================= patterns
with tabs[5]:
    seg = seg_cat.sort_values("alerts")
    fig = go.Figure()
    for col in ["caught", "false_alert", "missed"]:
        fig.add_bar(y=seg["category"], x=seg[col], name=OUTCOME_LABEL[col], orientation="h", marker_color=STATUS[col],
                    marker_line=dict(color=SURFACE, width=1))
    fig.update_layout(barmode="stack", xaxis_title="Transactions")
    chart(layout(fig, "Alert outcomes by merchant category", 460))

    c = st.columns(2)
    fig = px.bar(hour_seg, x="hour", y="fraud_rate_pct", labels={"hour": "Hour of day", "fraud_rate_pct": "Fraud rate (%)"})
    fig.update_traces(marker_color=SERIES[0])
    with c[0]:
        chart(layout(fig, "Fraud rate by hour of day", legend_bottom=False))
    fig = go.Figure()
    for col in ["caught", "false_alert"]:
        fig.add_bar(x=hour_seg["hour"], y=hour_seg[col], name=OUTCOME_LABEL[col], marker_color=STATUS[col])
    fig.update_layout(barmode="stack", xaxis_title="Hour of day", yaxis_title="Alerts")
    with c[1]:
        chart(layout(fig, "Alerts by hour of day"))

    amt = D.segment(df, threshold, "amount_band")
    fig = go.Figure()
    for col in ["caught", "false_alert", "missed"]:
        fig.add_bar(x=amt["amount_band"].astype(str), y=amt[col], name=OUTCOME_LABEL[col], marker_color=STATUS[col])
    fig.update_layout(barmode="group", xaxis_title="Amount", yaxis_title="Transactions")
    chart(layout(fig, "Alert outcomes by transaction amount"))

    st_seg = (df.groupby("state").agg(transactions=("is_fraud", "size"), frauds=("is_fraud", "sum"))
              .query("transactions >= 500").assign(fraud_rate_pct=lambda t: 100 * t["frauds"] / t["transactions"])
              .reset_index())
    if len(st_seg):
        fig = px.choropleth(st_seg, locations="state", locationmode="USA-states", scope="usa", color="fraud_rate_pct",
                            color_continuous_scale=["#cde2fb", "#6da7ec", "#2a78d6", "#104281"],
                            hover_data={"transactions": ":,", "frauds": ":,"}, labels={"fraud_rate_pct": "Fraud rate (%)"})
        chart(layout(fig, "Fraud rate by cardholder state (states with 500+ transactions)", 420, legend_bottom=False))

# =============================================================== monitoring
with tabs[6]:
    mon = data.get("monitoring")
    if not mon:
        st.info("Run `python -m fraudlens.monitor` to generate the monitoring report.")
    else:
        m = pd.DataFrame(mon["monthly"])
        flagged = mon.get("months_flagged", [])
        (st.warning if mon["retrain_recommended"] else st.success)(
            f"**{'Retraining recommended' if mon['retrain_recommended'] else 'No retraining needed'}.** "
            + (f"Months flagged: {', '.join(flagged)}. " if flagged else "")
            + f"Policy: retrain if more than {mon['policy']['drift_share_alert']:.0%} of monitored features drift, "
              f"recall falls below {mon['policy']['recall_floor']:.0%} or precision below "
              f"{mon['policy']['precision_floor']:.0%}. Drift engine: {mon['drift_engine']}.")
        c = st.columns(2)
        fig = go.Figure()
        fig.add_scatter(x=m["month"], y=m["recall"], name="Recall", mode="lines+markers", line=dict(color=SERIES[0], width=2))
        fig.add_scatter(x=m["month"], y=m["precision"], name="Precision", mode="lines+markers",
                        line=dict(color=SERIES[1], width=2))
        fig.add_hline(y=mon["policy"]["recall_floor"], line_dash="dash", line_color=MUTED, annotation_text="recall floor")
        fig.update_layout(yaxis_range=[0, 1.05], yaxis_title="Rate")
        with c[0]:
            chart(layout(fig, "Monthly performance at the fixed threshold"))
        share = m["evidently_drift_share"].fillna(m["psi_major_share"])
        fig = go.Figure(go.Bar(x=m["month"], y=share, marker_color=SERIES[0], name="Drifted share"))
        fig.add_hline(y=mon["policy"]["drift_share_alert"], line_dash="dash", line_color=MUTED,
                      annotation_text="retrain trigger")
        fig.update_layout(yaxis_range=[0, 1], yaxis_title="Share of monitored features", yaxis_tickformat=".0%")
        with c[1]:
            chart(layout(fig, "Data drift vs the training period", legend_bottom=False))
        show = m[["month", "transactions", "frauds", "fraud_rate_pct", "evidently_drift_share", "psi_major_share",
                  "pr_auc", "precision", "recall", "alerts_per_day", "net_benefit", "retrain_recommended", "reasons"]]
        table(show, column_config={"net_benefit": st.column_config.NumberColumn("Est. net benefit", format="$%.0f"),
                                   "evidently_drift_share": st.column_config.NumberColumn("Evidently drift share", format="%.2f"),
                                   "psi_major_share": st.column_config.NumberColumn("PSI major share", format="%.2f")})
        fd = data.get("feature_drift")
        if fd is not None and len(fd):
            piv = fd.pivot_table(index="feature", columns="month", values="psi").round(3)
            fig = px.imshow(piv, color_continuous_scale=["#f4f3ef", "#9ec5f4", "#2a78d6", "#0d366b"], aspect="auto",
                            labels={"color": "PSI"}, text_auto=".2f")
            chart(layout(fig, "Population Stability Index per feature and month (0.25+ is a major shift)",
                         60 + 28 * len(piv), legend_bottom=False))
        st.markdown("#### Monthly stability of the champion (test period, deployed threshold)")
        from fraudlens.stability import monthly_stability
        try:
            stab = monthly_stability(results, mon["policy"]["recall_floor"], mon["policy"]["precision_floor"])
        except Exception:
            stab = None
        if stab:
            sm = pd.DataFrame(stab["months"])
            fig = go.Figure()
            for name, col, color in [("Recall", "recall", SERIES[0]), ("Precision", "precision", SERIES[1])]:
                ci = np.array(sm[f"{col}_ci95"].tolist(), dtype=float)
                fig.add_scatter(x=sm["month"], y=sm[col], name=name, mode="markers+lines",
                                line=dict(color=color, width=2), marker=dict(size=8),
                                error_y=dict(type="data", symmetric=False, array=ci[:, 1] - sm[col],
                                             arrayminus=sm[col] - ci[:, 0], color=color, thickness=1.2, width=4))
            fig.add_hline(y=mon["policy"]["recall_floor"], line_dash="dash", line_color=MUTED,
                          annotation_text="recall floor")
            fig.add_hline(y=mon["policy"]["precision_floor"], line_dash="dot", line_color=MUTED,
                          annotation_text="precision floor")
            fig.update_layout(yaxis_range=[0.3, 1.02], yaxis_title="Rate (95% Wilson interval)")
            chart(layout(fig, "Recall and precision by month, with 95% intervals", 340))
            (st.success if stab["stable"] else st.warning)(
                f"**{'Stable' if stab['stable'] else 'Not stable'} month to month.** Recall ranged "
                f"{stab['recall_range'][0]:.1%}-{stab['recall_range'][1]:.1%} and precision "
                f"{stab['precision_range'][0]:.1%}-{stab['precision_range'][1]:.1%}; a chi-square test finds no "
                f"more variation between months than chance (recall p = {stab['chi2_p_recall']:.2f}, precision "
                f"p = {stab['chi2_p_precision']:.2f}).")
            table(pd.DataFrame(stab["checks"]))
        fbm = CASES.feedback_summary(precision_floor=mon["policy"]["precision_floor"])
        st.markdown("#### Live feedback from analyst decisions")
        st.markdown(f"{fbm['reviewed_alerts']} reviewed alerts, live precision "
                    f"{pct(fbm['live_precision'])}, {fbm['open_alerts']} open. Verdict: **{fbm['verdict'].replace('_', ' ')}** "
                    f"({fbm['reason']}). Details and the decision log are in **Case review**; `python -m fraudlens.monitor` "
                    "adds this block to the monitoring summary and exports the labelled examples.")
        st.caption("Evidently HTML reports (one per month) are in reports/monitoring/ of the full pipeline output. "
                   "In production, labels arrive later (chargebacks, analyst decisions), so the performance "
                   "panel lags the drift panel.")

# ==================================================================== model
with tabs[7]:
    sel = results.get("selection", {})
    st.subheader(f"Champion: {MODEL_LABEL.get(champion, champion)}")
    if sel:
        st.info(sel["reason"])
    comp = pd.DataFrame(results.get("comparison", []))
    if len(comp):
        comp.insert(0, "Model", comp["model"].map(lambda x: MODEL_LABEL.get(x, x)))
        table(comp.drop(columns=["model"]))
        fig = px.bar(comp.sort_values("test_pr_auc"), x="test_pr_auc", y="Model", orientation="h",
                     labels={"test_pr_auc": "Test PR-AUC", "Model": ""})
        fig.update_traces(marker_color=[FRAUD if m == champion else SERIES[0] for m in comp.sort_values("test_pr_auc")["model"]])
        chart(layout(fig, "Test PR-AUC by model (champion in orange)", 300, legend_bottom=False))
    ci = results.get("champion_test_ci95")
    if ci:
        st.markdown("**Champion on the unseen test period, with bootstrap 95% confidence intervals**")
        est = results.get("results", {}).get(champion, {}).get("test", {})
        table(pd.DataFrame([{"metric": kk, "test value": est.get(kk), "95% CI low": v.get("low"),
                             "95% CI high": v.get("high")} for kk, v in ci.items() if isinstance(v, dict)]))
    ex = data.get("explanations")
    if ex:
        imp = pd.Series(ex["mean_abs_shap"]).head(15).rename(index=feature_label).iloc[::-1]
        fig = px.bar(x=imp.values, y=imp.index, orientation="h", labels={"x": "Mean |SHAP| (log-odds)", "y": ""})
        fig.update_traces(marker_color=SERIES[0])
        chart(layout(fig, f"What drives the score: mean |SHAP| ({ex['explainer']} explainer)", 420, legend_bottom=False))
    with st.expander("Evaluation and explanation figures"):
        figs = data["figures"]
        for name, cap in [("10_pr_curves_test", "Precision-recall curves, test"),
                          ("12_score_distribution_test", "Score distribution by class, test"),
                          ("14_calibration_test", "Calibration, test"),
                          ("15_monthly_performance_test", "Monthly performance, test"),
                          ("17_shap_beeswarm", "SHAP beeswarm"),
                          ("18_shap_waterfall_caught_fraud", "SHAP waterfall: a caught fraud"),
                          ("18_shap_waterfall_false_alert", "SHAP waterfall: a false alert")]:
            image(figs / f"{name}.png", cap)
    if results.get("imbalance_handling"):
        st.markdown("**Class imbalance handling (read from the fitted models)**")
        table(pd.DataFrame(list(results["imbalance_handling"].items()), columns=["model", "method"]))

    eq = data.get("explanation_quality")
    st.markdown("#### Are the explanations faithful to the model?")
    if eq:
        d = eq["deletion"]
        e = st.columns(4)
        e[0].metric("Score drop, top-3 features removed", f"{d['mean_score_drop_top_k']:.3f}",
                    help="Replace each alert's top-3 SHAP features with typical values and re-score")
        e[1].metric("Score drop, 3 random features removed", f"{d['mean_score_drop_random_k']:.3f}")
        e[2].metric("Alerts cleared (top-3 vs random)", f"{d['share_cleared_top_k']:.0%} vs {d['share_cleared_random_k']:.0%}")
        e[3].metric("Top-3 stability (Jaccard)", f"{eq['stability']['mean_top_k_jaccard']:.2f}",
                    help="Same alerts explained twice with different random seeds")
        st.markdown(f"On {eq['alerts_sampled']} test-period alerts, removing the three features SHAP names lowers the "
                    f"score **{d['ratio_top_vs_random']}x** more than removing three random features, so the reasons "
                    f"shown to analysts point at what the model actually used (verdict: **{eq['verdict']}**). "
                    f"Additivity error of the {eq['explainer']} explainer: median "
                    f"{eq['additivity']['median_abs_error_log_odds']} log-odds. Every live alert also carries a "
                    "counterfactual (the amount at which it would have been approved).")
    else:
        st.caption("Run `python -m fraudlens.explain_quality` to add the faithfulness checks.")

    st.markdown("#### How far do these results transfer to real transactions?")
    st.markdown("The data are **simulated** (Sparkov), so the test metrics are an upper-bound guide. Precision depends "
                "on how common fraud is and on how many false alarms real customers trigger; the table carries the "
                "test recall and false-positive rate over to other conditions.")
    from fraudlens.stability import prevalence_sensitivity
    test_res = results.get("results", {}).get(champion, {}).get("test")
    if test_res:
        days = float((df_all[df_all["split"] == "test"]["ts"].max() - df_all[df_all["split"] == "test"]["ts"].min())
                     .total_seconds() / 86400) or 1.0
        prev = prevalence_sensitivity(test_res, days)
        pv = pd.DataFrame(prev["rows"])
        pv.insert(0, "Fraud rate", pv.apply(lambda r: f"{r['fraud_rate_pct']:.2f}%" + (" (test)" if r["is_test_rate"] else ""), axis=1))
        pv = pv.drop(columns=["fraud_rate_pct", "is_test_rate"]).rename(columns={
            "precision_fpr_x1": "Precision (same false-alarm rate)", "alerts_per_day_fpr_x1": "Alerts/day",
            "precision_fpr_x2": "Precision (2x false alarms)", "alerts_per_day_fpr_x2": "Alerts/day (2x)",
            "precision_fpr_x5": "Precision (5x false alarms)", "alerts_per_day_fpr_x5": "Alerts/day (5x)"})
        table(pv, column_config={c: st.column_config.NumberColumn(format="percent") for c in pv.columns
                                 if c.startswith("Precision")})
        st.caption(f"Recall {prev['recall']:.1%} and false-positive rate {prev['false_positive_rate']:.4%} from the "
                   f"test period, at {prev['transactions_per_day']:,.0f} transactions a day. {prev['note']}")
    st.markdown("**Before relying on these numbers with real customers:** (1) run in shadow mode on real "
                "transactions (score, do not block) and compare with chargebacks; (2) re-estimate the threshold "
                "on real validation data; (3) check recall and precision per segment and month with the stability "
                "checks above; (4) measure real review cost and recovery rate before quoting savings.")

# ============================================================ data pipeline
with tabs[8]:
    st.markdown("From the open dataset to this page: every stage, the open-source tool that runs it, and the "
                "evidence it left behind.")
    try:
        st.graphviz_chart("""
digraph G { rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fillcolor="#f4f3ef",
  color="#c3c2b7", fontname="Helvetica", fontsize=11]; edge [color="#898781"];
  kaggle [label="Kaggle\\nSparkov CSVs"]; raw [label="MinIO raw\\n(bronze)"]; prep [label="Pandas\\nvalidate + clean"];
  silver [label="MinIO silver\\n(Parquet)"]; feat [label="Features\\n(card history)"]; gold [label="MinIO gold"];
  hms [label="Hive Metastore\\n(crawler)"]; trino [label="Trino SQL\\n+ access rules"]; superset [label="Superset BI"];
  train [label="scikit-learn / XGBoost\\nLightGBM + MLflow"]; shap [label="SHAP\\nexplanations"];
  api [label="FastAPI\\nscoring API"]; dash [label="Streamlit\\ndashboard"]; mon [label="Evidently drift\\nPrometheus/Grafana"];
  kaggle -> raw -> prep -> silver -> feat -> gold -> hms -> trino -> superset;
  feat -> train -> shap -> dash; train -> api -> mon; gold -> dash; train -> mon; }""")
    except Exception:
        pass
    acq = data.get("acquisition") or {}
    files = acq.get("files", {})
    entries = list(files.values()) if isinstance(files, dict) else files
    if entries:
        st.markdown(f"**2. Acquisition** · {acq.get('dataset')} · licence {acq.get('licence')} · via {acq.get('acquired_via')}")
        table(pd.DataFrame([{"file": f["file"], "rows": f["rows"], "MB": round(f.get("bytes", 0) / 1e6, 1),
                             "sha256": f["sha256"][:16] + "..."} for f in entries]))
    val = data.get("validation")
    if val:
        st.markdown("**3. Validation (data contract)**")
        table(pd.DataFrame([{"file": v.get("dataset"), "status": v.get("status"),
                             "rows": (v.get("profile") or {}).get("rows"), "checks": len(v.get("checks", [])),
                             "warnings": sum(ch.get("status") == "WARN" for ch in v.get("checks", []))} for v in val]))
    pipe = data.get("pipeline") or {}
    if pipe.get("split"):
        st.markdown("**3. Chronological split**")
        sp = pipe["split"].get("splits", pipe["split"])
        table(pd.DataFrame(sp).T.reset_index(names="split"))
    cat = data.get("catalog") or {}
    st.markdown(f"**Data catalog** (Hive Metastore via the crawler): status **{cat.get('status', 'not run')}**")
    if cat.get("tables"):
        table(pd.DataFrame([{"table": t["table"], "format": t.get("format"), "rows": t.get("rows"),
                             "partitions": t.get("partitions"), "location": t.get("location"),
                             "description": t.get("description")} for t in cat["tables"]]))
    elif cat.get("reason"):
        st.caption(f"Not registered in this build: {cat['reason']}. Start the Docker platform and run "
                   "`python -m fraudlens.run_all --only publish crawl`.")
    ver = data.get("verification")
    if ver:
        (st.success if ver["status"] == "PASS" else st.error)(
            f"End-to-end verification: {ver['status']} ({ver['passed']} checks passed, {ver['failed']} failed)")
        if ver.get("success_criteria"):
            st.markdown("**Success criteria (stage 1)**")
            table(pd.DataFrame(ver["success_criteria"]))
        with st.expander("All verification checks"):
            table(pd.DataFrame(ver["checks"])[["stage", "check", "status"]])

# =================================================================== report
with tabs[9]:
    filters = {"split": split, "start": start, "end": end, "threshold": threshold,
               "categories": categories, "states": states, "cost": cost, "recovery": recovery}
    cat_seg = seg_cat.sort_values("false_alert", ascending=False)
    md = D.summary_markdown(k, filters, meta, cat_seg)
    st.markdown(md.replace("$", "\\$"))      # a bare $ would start LaTeX maths in st.markdown
    c = st.columns(2)
    c[0].download_button("Download summary (Markdown)", md.encode(), file_name="fraudlens_summary.md",
                         mime="text/markdown")
    c[1].download_button("Download category breakdown (CSV)", cat_seg.to_csv(index=False).encode(),
                         file_name="fraudlens_categories.csv", mime="text/csv")
