FILENAME = "04_eda.ipynb"

CELLS = [
("md", r"""
# 4 · Exploratory data analysis

| | |
|---|---|
| Lifecycle stage | 4 of 8: *EDA* (understand patterns, test the stage-1 hypotheses) |
| Tools | pandas, Seaborn, Matplotlib, Plotly |
| Code | `src/fraudlens/eda.py`, `src/fraudlens/interactive.py` |
| Data | **training period only** (Jan 2019 to May 2020), so nothing learned here leaks from validation or test |
| Outputs | `reports/figures/01-09_*.png`, `reports/eda_summary.json`, `reports/interactive/eda_explorer.html` |
"""),
("setup", r"""
from fraudlens import eda
from fraudlens import features as F
from fraudlens.datasets import load_analysis_frame
df = load_analysis_frame(cfg, "train")
numeric = F.numeric_features(cfg["features"]["velocity_windows"])
print(f"{len(df):,} training transactions, {int(df['is_fraud'].sum()):,} frauds ({df['is_fraud'].mean():.3%})")"""),
("md", r"""
## 4.1 Class balance: rare, and rarer in the test period

About one transaction in 170 is fraud during training, and the test period is even more imbalanced.
Precision at a fixed threshold will therefore be lower on test than on validation, for reasons that have
nothing to do with the model.
"""),
("code", r"""
t, fig = eda.class_balance(cfg)
plt.show()
t
"""),
("md", r"""
## 4.2 H1: fraudulent transactions are much larger
"""),
("code", r"""
t, fig = eda.amount_profile(df)
plt.show()
display(t)
fig = eda.amount_by_category_seaborn(df)
plt.show()
"""),
("code", r"""
top5 = df["amount"] >= df["amount"].quantile(0.95)
print(f"The top 5% of amounts contain {df.loc[top5, 'is_fraud'].sum() / df['is_fraud'].sum():.0%} of all fraud "
      f"(fraud rate {df.loc[top5, 'is_fraud'].mean():.2%} vs {df['is_fraud'].mean():.2%} overall).")
"""),
("md", r"""
## 4.3 H4: merchant category matters
"""),
("code", r"""
t, fig = eda.fraud_by_category(df)
plt.show()
t.head(8)
"""),
("md", r"""
## 4.4 H2: fraud happens at night
"""),
("code", r"""
tabs, fig = eda.fraud_by_time(df)
plt.show()
night = df["is_night"] == 1
print(f"Night hours (22:00-03:59): {night.mean():.1%} of transactions but {df.loc[night, 'is_fraud'].sum() / df['is_fraud'].sum():.1%} of fraud; "
      f"fraud rate {df.loc[night, 'is_fraud'].mean():.2%} at night vs {df.loc[~night, 'is_fraud'].mean():.2%} by day.")
"""),
("md", r"""
## 4.5 H5: demographics carry little signal
"""),
("code", r"""
tabs, fig = eda.fraud_by_demographics(df, cfg["eda"]["min_state_txns"])
plt.show()
pd.concat({k: v["fraud_rate_pct"] for k, v in tabs.items() if k != "state"}, axis=0).to_frame("fraud_rate_pct").T
"""),
("md", r"""
## 4.6 H3: fraud comes in bursts on a compromised card
"""),
("code", r"""
tabs, fig = eda.fraud_by_behaviour(df)
plt.show()
cards = eda.card_level_patterns(df)
pd.Series(cards).to_frame("value").T
"""),
("md", r"""
## 4.7 Which single features separate fraud best?

Single-feature ROC AUC (0.5 = no signal) on a stratified sample, and the Spearman correlation between
features: highly correlated pairs (e.g. `amount` and `log_amount`) are fine for trees but are reduced for
the linear model.
"""),
("code", r"""
t, fig = eda.feature_signal(df, numeric, cfg["eda"]["sample_rows"])
plt.show()
t.head(12)
"""),
("code", r"""
pairs, fig = eda.correlations(df, numeric, cfg["eda"]["sample_rows"], cfg["eda"]["corr_threshold"])
plt.show()
pairs.head(8)
"""),
("md", r"""
## 4.8 Interactive views (Plotly)

The same analysis as interactive charts: fraud rate by category × hour of day, and the amount distribution
by class. The full interactive explorer is `reports/interactive/eda_explorer.html` (open it in a browser).
"""),
("code", r"""
import plotly.express as px
import plotly.io as pio
pio.templates["fraudlens"] = viz.plotly_template()
pio.templates.default = "fraudlens"

heat = (df.groupby(["category", "hour"], observed=True)["is_fraud"].mean().mul(100).rename("fraud_rate_pct")
          .reset_index().pivot(index="category", columns="hour", values="fraud_rate_pct"))
fig = px.imshow(heat, aspect="auto", color_continuous_scale=viz.BLUE_RAMP,
                labels={"color": "Fraud rate (%)", "x": "Hour of day", "y": ""},
                title="Fraud rate by merchant category and hour (training period)")
fig.show()
"""),
("code", r"""
rng = np.random.default_rng(0)
legit = df[df["is_fraud"] == 0]
s = pd.concat([df[df["is_fraud"] == 1], legit.iloc[rng.choice(len(legit), min(60_000, len(legit)), replace=False)]])
s = s.assign(Class=s["is_fraud"].map({0: "Legitimate", 1: "Fraud"}), log10_amount=np.log10(s["amount"]))
fig = px.histogram(s, x="log10_amount", color="Class", histnorm="probability density", barmode="overlay", nbins=80,
                   opacity=0.6, color_discrete_map={"Legitimate": viz.LEGIT, "Fraud": viz.FRAUD},
                   title="Amount by class (log10 $): fraud clusters at $200-$1,200")
fig.show()
"""),
("md", r"""
## 4.9 Drift between periods (a preview of stage 8)

Population Stability Index of each feature between training and the later periods. Running totals such
as `card_prior_txn_count` drift by construction (they only grow) and were excluded from modelling; the
behavioural ratios and windows are stable.
"""),
("code", r"""
summary = read_json(P["reports"] / "eda_summary.json")
drift = pd.DataFrame(summary["drift_psi"]).T.sort_values("test", ascending=False)
display(drift.head(10))
pd.DataFrame(summary["new_cards"]).T
"""),
("md", r"""
## 4.10 What EDA tells the modelling stage

| Hypothesis | Verdict | Consequence for modelling |
|---|---|---|
| H1 large amounts | **Strongly supported**: median fraud ≈ $390 vs ≈ $47 legitimate; most fraud is in the top 5% of amounts | Keep outliers; use amount and amount relative to the card's history |
| H2 night | **Strongly supported**: about 85% of training fraud happens between 22:00 and 03:59, where the fraud rate is ~18× the daytime rate | `hour`, `is_night` |
| H3 bursts | **Supported**: fraud cards see ~10 frauds within ~2 days; 24-hour spend is the single strongest feature | velocity windows 1 h / 24 h / 7 d |
| H4 category | **Supported**: online shopping, misc. online and grocery POS have 2-3× the base rate | `category` (native categorical in trees) |
| H5 demographics | **Weak**: age bands differ modestly, gender barely | keep `age`, `gender` but monitor fairness (stage 6) |

Other findings: fraud is rarer in the test period; a few cards appear only after training (cold start, handled
by the online feature store); feature pairs with |ρ| > 0.85 are kept for trees and handled for the linear model.

**Next:** [05 · Modelling](05_modelling.ipynb)
"""),
]
