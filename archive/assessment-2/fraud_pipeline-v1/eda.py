"""Exploratory data analysis on the TRAINING split.

Each analysis function returns a table (DataFrame) and, where useful, a
matplotlib Figure. `run_eda` runs them all, saves the charts to
reports/figures/ and a machine-readable summary to reports/eda_summary.json.

Only the drift section touches validation/test data, and it uses features
only (no labels).
"""
from __future__ import annotations

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from . import features as F  # noqa: E402
from .datasets import load_analysis_frame, load_features  # noqa: E402
from .drift import psi, psi_flag  # noqa: E402
from .utils import get_logger, write_json  # noqa: E402

log = get_logger()

LEGIT_COLOR, FRAUD_COLOR = "#4C72B0", "#C44E52"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 12,
})


# ----------------------------------------------------------------- helpers
def rate_table(df: pd.DataFrame, by: str | list[str], min_count: int = 0) -> pd.DataFrame:
    t = (df.assign(_fraud_amt=df["amount"] * df["is_fraud"])
           .groupby(by, observed=True)
           .agg(transactions=("is_fraud", "size"), frauds=("is_fraud", "sum"),
                fraud_amount=("_fraud_amt", "sum"))
           .assign(fraud_rate_pct=lambda t: 100 * t["frauds"] / t["transactions"]))
    t["share_of_frauds_pct"] = 100 * t["frauds"] / t["frauds"].sum()
    return t[t["transactions"] >= min_count].round(3)


def save_figure(fig, cfg, name):
    if fig is not None and cfg is not None:
        fig.savefig(cfg["paths"]["figures"] / f"{name}.png", bbox_inches="tight")


def _base_line(ax, rate_pct):
    ax.axhline(rate_pct, ls="--", lw=1, color="grey")
    ax.text(ax.get_xlim()[1], rate_pct, " overall", va="center", fontsize=8, color="grey")


# ---------------------------------------------------------------- analyses
def class_balance(cfg: dict) -> tuple[pd.DataFrame, plt.Figure]:
    rows = []
    for s in ["train", "val", "test"]:
        y = load_features(cfg, s)["is_fraud"]
        rows.append({"split": s, "transactions": len(y), "frauds": int(y.sum()),
                     "fraud_rate_pct": round(100 * y.mean(), 3),
                     "legit_per_fraud": round((len(y) - y.sum()) / max(y.sum(), 1), 1)})
    t = pd.DataFrame(rows).set_index("split")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(t.index, t["fraud_rate_pct"], color=FRAUD_COLOR)
    for i, v in enumerate(t["fraud_rate_pct"]):
        ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom")
    ax.set(title="Fraud rate by split", ylabel="Fraud rate (%)")
    return t, fig


def amount_profile(df: pd.DataFrame) -> tuple[pd.DataFrame, plt.Figure]:
    q = [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    t = df.groupby("is_fraud")["amount"].quantile(q).unstack()
    t.columns = [f"p{int(c * 100)}" for c in t.columns]
    t.insert(0, "mean", df.groupby("is_fraud")["amount"].mean())
    t.index = t.index.map({0: "legit", 1: "fraud"})
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bins = np.logspace(0, np.log10(df["amount"].max() + 1), 60)
    for lab, col, name in [(0, LEGIT_COLOR, "Legit"), (1, FRAUD_COLOR, "Fraud")]:
        ax.hist(df.loc[df.is_fraud == lab, "amount"], bins=bins, density=True, alpha=0.55, color=col, label=name)
    ax.set_xscale("log")
    ax.set(title="Transaction amount by class (density, log scale)", xlabel="Amount ($)", ylabel="Density")
    ax.legend()
    return t.round(2), fig


def fraud_by_category(df: pd.DataFrame) -> tuple[pd.DataFrame, plt.Figure]:
    t = rate_table(df, "category").sort_values("fraud_rate_pct")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(t.index.astype(str), t["fraud_rate_pct"], color=FRAUD_COLOR)
    ax.axvline(100 * df["is_fraud"].mean(), ls="--", lw=1, color="grey")
    ax.set(title="Fraud rate by merchant category", xlabel="Fraud rate (%)")
    return t.sort_values("fraud_rate_pct", ascending=False), fig


def fraud_by_time(df: pd.DataFrame) -> tuple[dict, plt.Figure]:
    base = 100 * df["is_fraud"].mean()
    hour = rate_table(df, "hour")
    dow = rate_table(df, "day_of_week")
    dow.index = [DAYS[i] for i in dow.index]
    month = rate_table(df.assign(month=df["ts"].dt.to_period("M").astype(str)), "month")
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), gridspec_kw={"width_ratios": [2, 1, 2]})
    axes[0].plot(hour.index, hour["fraud_rate_pct"], marker="o", color=FRAUD_COLOR)
    axes[0].set(title="Fraud rate by hour of day", xlabel="Hour", ylabel="Fraud rate (%)", xticks=range(0, 24, 2))
    axes[1].bar(dow.index, dow["fraud_rate_pct"], color=FRAUD_COLOR)
    axes[1].set(title="By day of week")
    axes[2].plot(month.index, month["fraud_rate_pct"], marker="o", color=FRAUD_COLOR)
    axes[2].set(title="By month (training period)")
    axes[2].tick_params(axis="x", rotation=90)
    for ax in axes:
        ax.set_ylim(bottom=0)  # rates start at zero so differences are not exaggerated
        _base_line(ax, base)
    fig.tight_layout()
    return {"hour": hour, "day_of_week": dow, "month": month}, fig


def fraud_by_demographics(df: pd.DataFrame, min_state_txns: int) -> tuple[dict, plt.Figure]:
    bands = pd.cut(df["age"], [0, 25, 35, 45, 55, 65, 75, 120],
                   labels=["<25", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"], right=False)
    age = rate_table(df.assign(age_band=bands), "age_band")
    gender = rate_table(df, "gender")
    state = rate_table(df, "state", min_count=min_state_txns).sort_values("fraud_rate_pct", ascending=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), gridspec_kw={"width_ratios": [2, 1, 2]})
    axes[0].bar(age.index.astype(str), age["fraud_rate_pct"], color=FRAUD_COLOR)
    axes[0].set(title="Fraud rate by age band", ylabel="Fraud rate (%)")
    axes[1].bar(gender.index.astype(str), gender["fraud_rate_pct"], color=FRAUD_COLOR)
    axes[1].set(title="By gender")
    top = state.head(10)
    axes[2].bar(top.index.astype(str), top["fraud_rate_pct"], color=FRAUD_COLOR)
    axes[2].set(title=f"Top 10 states (≥{min_state_txns:,} txns)")
    for ax in axes:
        _base_line(ax, 100 * df["is_fraud"].mean())
    fig.tight_layout()
    return {"age_band": age, "gender": gender, "state": state}, fig


def fraud_by_behaviour(df: pd.DataFrame) -> tuple[dict, plt.Figure]:
    """Card-history and velocity signals engineered in Step 1."""
    base = 100 * df["is_fraud"].mean()
    count_bins = [-1, 0, 1, 2, 4, 9, np.inf]
    count_lbl = ["0", "1", "2", "3-4", "5-9", "10+"]
    ratio_bins = [0, 0.5, 1, 2, 5, 10, np.inf]
    ratio_lbl = ["<0.5x", "0.5-1x", "1-2x", "2-5x", "5-10x", "10x+"]
    d = df.assign(
        txn_24h_bucket=pd.cut(df["txn_count_24h"], count_bins, labels=count_lbl),
        amt_ratio_bucket=pd.cut(df["amt_to_card_mean"], ratio_bins, labels=ratio_lbl),
        night=df["is_night"].map({0: "day", 1: "night (22-04)"}),
        new_merchant=df["is_new_merchant_for_card"].map({0: "seen before", 1: "new"}),
    )
    tables = {
        "txn_count_24h": rate_table(d, "txn_24h_bucket"),
        "amount_vs_card_mean": rate_table(d, "amt_ratio_bucket"),
        "night": rate_table(d, "night"),
        "new_merchant": rate_table(d, "new_merchant"),
        "distance_km_by_class": df.groupby("is_fraud")["distance_km"].describe().round(1),
    }
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, (key, title) in zip(axes, [("txn_count_24h", "Prior txns in last 24h"),
                                       ("amount_vs_card_mean", "Amount vs card's usual"),
                                       ("night", "Time of day"), ("new_merchant", "Merchant new to card")]):
        t = tables[key]
        ax.bar(t.index.astype(str), t["fraud_rate_pct"], color=FRAUD_COLOR)
        ax.set(title=title, ylabel="Fraud rate (%)" if ax is axes[0] else None)
        ax.tick_params(axis="x", rotation=30)
        _base_line(ax, base)
    fig.tight_layout()
    return tables, fig


def card_level_patterns(df: pd.DataFrame) -> dict:
    fr = df[df["is_fraud"] == 1]
    per_card = fr.groupby("card_id", observed=True).agg(
        frauds=("is_fraud", "size"), first=("ts", "min"), last=("ts", "max"), amount=("amount", "sum"))
    span_h = (per_card["last"] - per_card["first"]).dt.total_seconds() / 3600
    return {
        "cards_total": int(df["card_id"].nunique()),
        "cards_with_fraud": int(len(per_card)),
        "frauds_per_fraud_card_median": float(per_card["frauds"].median()),
        "frauds_per_fraud_card_max": int(per_card["frauds"].max()),
        "fraud_span_hours_median": round(float(span_h.median()), 1),
        "fraud_span_hours_p90": round(float(span_h.quantile(0.9)), 1),
        "fraud_amount_per_card_median": round(float(per_card["amount"].median()), 2),
    }


def feature_signal(df: pd.DataFrame, numeric: list[str], sample_rows: int, seed: int = 0) -> tuple[pd.DataFrame, plt.Figure]:
    """Single-feature ROC AUC (direction-free) and class means, on a stratified sample."""
    s = _stratified_sample(df, sample_rows, seed)
    rows = []
    for c in numeric:
        ok = s[c].notna()
        auc = roc_auc_score(s.loc[ok, "is_fraud"], s.loc[ok, c]) if s.loc[ok, "is_fraud"].nunique() == 2 else np.nan
        rows.append({"feature": c, "auc": max(auc, 1 - auc), "direction": "higher=riskier" if auc >= 0.5 else "lower=riskier",
                     "mean_legit": df.loc[df.is_fraud == 0, c].mean(), "mean_fraud": df.loc[df.is_fraud == 1, c].mean()})
    t = pd.DataFrame(rows).set_index("feature").sort_values("auc", ascending=False).round(4)
    fig, ax = plt.subplots(figsize=(7, 6))
    tt = t.sort_values("auc")
    ax.barh(tt.index, tt["auc"], color=[FRAUD_COLOR if a >= 0.7 else LEGIT_COLOR for a in tt["auc"]])
    ax.axvline(0.5, ls="--", lw=1, color="grey")
    ax.set(title="Single-feature ROC AUC (0.5 = no signal)", xlabel="AUC", xlim=(0.45, 1))
    return t, fig


def correlations(df: pd.DataFrame, numeric: list[str], sample_rows: int, threshold: float, seed: int = 0):
    s = df[numeric].sample(min(sample_rows, len(df)), random_state=seed)
    corr = s.corr(method="spearman")
    pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
                 .rename("spearman_rho").reset_index())
    pairs = pairs[pairs["spearman_rho"].abs() >= threshold].sort_values("spearman_rho", key=abs, ascending=False)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric)), numeric, rotation=90, fontsize=8)
    ax.set_yticks(range(len(numeric)), numeric, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman rho")
    ax.set_title("Feature correlation (Spearman)")
    return pairs.round(3).reset_index(drop=True), fig


def drift_check(cfg: dict, train: pd.DataFrame, numeric: list[str]) -> tuple[pd.DataFrame, dict]:
    """Label-free comparison of train vs validation/test feature distributions.

    Includes the time-stable replacement features from preprocessing, so the
    report shows both the problem (running totals drift) and the fix.
    """
    from .preprocess import DERIVED, add_stationary_features
    train = add_stationary_features(train.copy())
    numeric = list(numeric) + DERIVED
    out, extra = {}, {}
    train_cards = set(train["card_id"].astype(str))
    for s in ["val", "test"]:
        other = add_stationary_features(load_features(cfg, s))
        out[s] = {c: psi(train[c], other[c]) for c in numeric}
        cards = set(other["card_id"].astype(str))
        new = cards - train_cards
        extra[s] = {"cards": len(cards), "cards_not_in_train": len(new),
                    "transactions_on_new_cards": int(other["card_id"].astype(str).isin(new).sum())}
    t = pd.DataFrame(out).round(4)
    t["flag_val"] = t["val"].map(psi_flag)
    t["flag_test"] = t["test"].map(psi_flag)
    return t.sort_values("val", ascending=False), extra


def _stratified_sample(df, n, seed):
    if len(df) <= n:
        return df
    frac = n / len(df)
    return df.groupby("is_fraud", group_keys=False).sample(frac=frac, random_state=seed)


# ------------------------------------------------------------------ runner
def run_eda(cfg: dict) -> dict:
    ecfg = cfg["eda"]
    numeric = F.numeric_features(cfg["features"]["velocity_windows"])
    df = load_analysis_frame(cfg, "train")
    log.info("EDA on %s training transactions", f"{len(df):,}")
    summary: dict = {}

    t, fig = class_balance(cfg); save_figure(fig, cfg, "01_class_balance"); summary["class_balance"] = t.to_dict("index")
    t, fig = amount_profile(df); save_figure(fig, cfg, "02_amount_by_class"); summary["amount_by_class"] = t.to_dict("index")
    t, fig = fraud_by_category(df); save_figure(fig, cfg, "03_fraud_by_category"); summary["category"] = t.to_dict("index")
    tabs, fig = fraud_by_time(df); save_figure(fig, cfg, "04_fraud_by_time")
    summary["time"] = {k: v.to_dict("index") for k, v in tabs.items()}
    tabs, fig = fraud_by_demographics(df, ecfg["min_state_txns"]); save_figure(fig, cfg, "05_fraud_by_demographics")
    summary["demographics"] = {k: v.to_dict("index") for k, v in tabs.items()}
    tabs, fig = fraud_by_behaviour(df); save_figure(fig, cfg, "06_fraud_by_behaviour")
    summary["behaviour"] = {k: v.to_dict("index") for k, v in tabs.items()}
    summary["card_level"] = card_level_patterns(df)
    t, fig = feature_signal(df, numeric, ecfg["sample_rows"]); save_figure(fig, cfg, "07_feature_signal")
    summary["feature_signal"] = t.to_dict("index")
    pairs, fig = correlations(df, numeric, ecfg["sample_rows"], ecfg["corr_threshold"]); save_figure(fig, cfg, "08_correlation")
    summary["high_correlation_pairs"] = pairs.to_dict("records")
    t, extra = drift_check(cfg, df, numeric)
    summary["drift_psi"] = t.to_dict("index")
    summary["new_cards"] = extra
    plt.close("all")

    write_json(_stringify_keys(summary), cfg["paths"]["reports"] / "eda_summary.json")
    log.info("EDA finished: figures in %s", cfg["paths"]["figures"])
    return summary


def _stringify_keys(o):
    if isinstance(o, dict):
        return {str(k): _stringify_keys(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_stringify_keys(v) for v in o]
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


if __name__ == "__main__":
    from .utils import load_config
    plt.switch_backend("Agg")  # save figures without opening windows
    run_eda(load_config())
