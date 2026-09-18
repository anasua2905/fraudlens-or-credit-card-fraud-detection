"""Builds notebooks/01_cleaning_eda_preprocessing.ipynb (run from project root)."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# FraudLens — Data Cleaning, Preprocessing and EDA

**Dataset:** Sparkov Credit Card Transactions Fraud Detection (Kaggle, ~1.85M simulated transactions, 2019–2020)

This notebook covers three stages of the project:

| Part | Goal | Data used |
|---|---|---|
| A. Data cleaning | Audit data quality and record a decision for every issue | All transactions (labels: training split only) |
| B. Exploratory data analysis | Understand how fraud differs from legitimate activity | Training split only |
| C. Preprocessing | Build model-ready matrices, fitted on training data only | Train → applied to validation and test |

**Why training data only?** Anything learned from the validation or test periods would leak into later
modelling decisions and make results look better than they really are. The one exception is the drift check
in B.10, which compares feature distributions (never labels) across periods.

**Prerequisite:** run the Step 1 pipeline first (`python -m fraud_pipeline.pipeline`).
""")

code("""
import sys
from pathlib import Path

# Locate the project root (the folder containing pyproject.toml)
ROOT = Path.cwd()
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

import json
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from fraud_pipeline import eda, preprocess, quality
from fraud_pipeline import features as F
from fraud_pipeline.datasets import load_analysis_frame
from fraud_pipeline.utils import load_config

%matplotlib inline
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", "{:,.3f}".format)

cfg = load_config(root=ROOT)
NUMERIC = F.numeric_features(cfg["features"]["velocity_windows"])
print("Project root:", ROOT)
""")

# ------------------------------------------------------------------ Part A
md("""
---
# Part A — Data cleaning

## A.1 Recap of Step 1 cleaning

The ingestion pipeline already validated both raw files against a data contract and removed structural problems:
parsed dates, removed duplicate transaction ids and invalid rows, stripped the `fraud_` prefix that the data
generator adds to every merchant name, and replaced card numbers with salted hashes while dropping names and
street addresses.
""")

code("""
validation = json.loads((cfg["paths"]["reports"] / "validation_report.json").read_text())
run_log = json.loads((cfg["paths"]["reports"] / "pipeline_run.json").read_text())

checks = pd.DataFrame(
    {r["dataset"]: {c["check"]: c["status"] for c in r["checks"]} for r in validation}
)
display(checks)
display(pd.DataFrame(run_log["cleaning"]).set_index("source_split"))
""")

md("""
All checks pass except `unix_time_consistent_with_timestamp`: that column is offset by about 7 years from the
readable timestamp, so the pipeline ignores it and derives time from `trans_date_trans_time`.

## A.2 Quality audit of the cleaned data

The audit looks for subtler issues: column types and cardinality, consistency (does each card keep one profile?
does each merchant belong to one category?), implausible ages, outliers and near-duplicate transactions.
""")

code("""
audit = quality.run_quality_audit(cfg)

profile = pd.DataFrame(audit["structural_profile"]["columns_detail"]).T
print(f"{audit['structural_profile']['rows']:,} rows | "
      f"{audit['structural_profile']['columns']} columns | "
      f"{audit['structural_profile']['memory_mb']:,} MB in memory")
display(profile)
""")

code("""
display(pd.Series(audit["consistency"], name="value").to_frame())
""")

md("""
## A.3 Outliers

Outliers are flagged with the 1.5 × IQR rule on the training split. The key question is whether outliers are
errors to remove or signal to keep, so the table compares the fraud rate inside the outliers with the overall rate.
""")

code("""
outliers = pd.DataFrame(audit["outliers_train"]).T
display(outliers)
""")

md("""
## A.4 Near-duplicates and missing values
""")

code("""
display(pd.Series(audit["near_duplicates_train"], name="value").to_frame())
display(pd.Series(audit["missing_train_pct"], name="missing %").to_frame())
""")

md("""
Missing values appear only in the card-history features, on a card's first one or two transactions, where no
history exists yet. They are expected, not errors.

## A.5 Cleaning decisions
""")

code("""
pd.set_option("display.max_colwidth", None)
display(pd.DataFrame(audit["decisions"]))
pd.reset_option("display.max_colwidth")
""")

# ------------------------------------------------------------------ Part B
md("""
---
# Part B — Exploratory data analysis (training split)

Charts are also saved to `reports/figures/` for the report and dashboard.
""")

code("""
df = load_analysis_frame(cfg, "train")
print(f"Training transactions: {len(df):,}   frauds: {int(df.is_fraud.sum()):,}   "
      f"fraud rate: {df.is_fraud.mean():.3%}")
df.head()
""")

md("""
## B.1 Class balance

Fraud is rare in every split. A model that labels everything "legitimate" would already be more than 99%
accurate, so accuracy is not a useful metric; later steps use precision, recall and PR-AUC.
""")

code("""
t, fig = eda.class_balance(cfg)
eda.save_figure(fig, cfg, "01_class_balance")
display(t); plt.show()
""")

md("""
## B.2 Transaction amount
""")

code("""
t, fig = eda.amount_profile(df)
eda.save_figure(fig, cfg, "02_amount_by_class")
display(t); plt.show()
""")

md("""
## B.3 Merchant category

`share_of_frauds_pct` shows where investigators' attention would be spent; `fraud_rate_pct` shows how risky
each category is. The dashed line is the overall fraud rate.
""")

code("""
t, fig = eda.fraud_by_category(df)
eda.save_figure(fig, cfg, "03_fraud_by_category")
display(t); plt.show()
""")

md("""
## B.4 Time patterns
""")

code("""
tabs, fig = eda.fraud_by_time(df)
eda.save_figure(fig, cfg, "04_fraud_by_time")
plt.show()
display(tabs["hour"])
""")

code("""
display(tabs["day_of_week"])
display(tabs["month"])
""")

md("""
## B.5 Cardholder demographics and location
""")

code("""
tabs, fig = eda.fraud_by_demographics(df, cfg["eda"]["min_state_txns"])
eda.save_figure(fig, cfg, "05_fraud_by_demographics")
plt.show()
display(tabs["age_band"])
display(tabs["gender"])
display(tabs["state"].head(10))
""")

md("""
## B.6 Behavioural signals (engineered in Step 1)

These features compare each transaction with the card's own history. They are the main tool for reducing false
positives: a $900 purchase is normal for some cardholders and highly unusual for others.
""")

code("""
tabs, fig = eda.fraud_by_behaviour(df)
eda.save_figure(fig, cfg, "06_fraud_by_behaviour")
plt.show()
for name in ["txn_count_24h", "amount_vs_card_mean", "night", "new_merchant", "distance_km_by_class"]:
    print(name); display(tabs[name])
""")

md("""
## B.7 Card-level fraud patterns

Does fraud arrive as a single event or as a burst on a compromised card? Short bursts favour velocity features
and suggest the dashboard should group alerts by card.
""")

code("""
display(pd.Series(eda.card_level_patterns(df), name="value").to_frame())
""")

md("""
## B.8 Single-feature signal

ROC AUC of each feature used on its own (0.5 = no signal, 1.0 = perfect separation), computed on a stratified
sample. This is a screening tool only: features can be weak alone and strong in combination.
""")

code("""
t, fig = eda.feature_signal(df, NUMERIC, cfg["eda"]["sample_rows"])
eda.save_figure(fig, cfg, "07_feature_signal")
display(t); plt.show()
""")

md("""
## B.9 Correlation between features

Highly correlated pairs carry overlapping information. Tree models tolerate this; linear models can become
unstable, which is why the linear view in Part C keeps only one of each obvious pair (e.g. `log_amount` instead
of `amount`).
""")

code("""
pairs, fig = eda.correlations(df, NUMERIC, cfg["eda"]["sample_rows"], cfg["eda"]["corr_threshold"])
eda.save_figure(fig, cfg, "08_correlation")
display(pairs); plt.show()
""")

md("""
## B.10 Distribution drift across periods

The Population Stability Index (PSI) compares each feature's distribution in validation and test with training.
Rule of thumb: below 0.10 stable, 0.10–0.25 moderate, above 0.25 major. **Only the validation column is used
for decisions**; the test column is shown for information. No labels are used here.

The Step 1 running totals (`card_prior_txn_count`, `card_prior_category_count`) only ever increase, so they are
expected to drift. `category_share_card` is the bounded replacement created in preprocessing.
""")

code("""
drift, new_cards = eda.drift_check(cfg, df, NUMERIC)
display(drift)
display(pd.DataFrame(new_cards).T)
""")

code("""
# Save the full EDA summary (tables + charts) for the report
summary = eda.run_eda(cfg)
print("Saved:", cfg["paths"]["reports"] / "eda_summary.json")
""")

# ------------------------------------------------------------------ Part C
md("""
---
# Part C — Preprocessing

Two model-ready views are built. Every transformer is **fitted on the training split only** and then applied
unchanged to validation and test.

| Step | Linear view (logistic regression, neural nets, autoencoders) | Tree view (Random Forest, XGBoost, LightGBM) |
|---|---|---|
| Skewed features | log(1 + x) | unchanged |
| Extreme values | clipped at training 0.1% / 99.9% quantiles | unchanged |
| Missing history | median + missing-indicator column | kept as NaN (handled natively) |
| Scale | standardised (mean 0, std 1) | unchanged |
| Hour, day of week | sine/cosine encoding (23:00 is next to 00:00) | integer |
| Category, gender | one-hot | ordinal codes |
| Running totals | excluded (config `exclude_features`) | excluded |

**Class imbalance is deliberately not handled here.** Oversampling or class weights must be applied inside model
training on the training split only; applying them earlier would distort validation and test results.
""")

code("""
report = preprocess.run_preprocessing(cfg)

print("Excluded:", report["excluded_features"], "| Added:", report["added_features"])
print("Training class counts:", report["class_counts_train"],
      "| suggested scale_pos_weight:", report["suggested_scale_pos_weight"])
for name, info in report["views"].items():
    print(f"\\n{name} view: {info['n_features']} features")
    display(pd.DataFrame(info["splits"]).T)
""")

md("""
## C.1 Checks on the linear view

After scaling, continuous training features should have mean ≈ 0 and standard deviation ≈ 1, and no split
should contain missing values.
""")

code("""
lin = report["views"]["linear"]
print("Max |mean| of scaled training features:", lin["train_scaled_mean_abs_max"])
print("Std range of scaled training features:", lin["train_scaled_std_range"])
print("\\nFeatures:", lin["feature_names"])
""")

md("""
## C.2 Drift of the final model inputs (train → validation)

Features flagged **major** should be reviewed before modelling. To exclude one, add it to
`preprocess.exclude_features` in `config/pipeline.yaml` and re-run this part.
""")

code("""
vd = pd.DataFrame(report["validation_drift_psi"]).T.sort_values("psi", ascending=False)
display(vd)
""")

code("""
from fraud_pipeline.utils import read_table
X_train = read_table(cfg["paths"]["processed"] / "linear_train")
X_train.describe().T.head(15)
""")

md("""
---
# Summary

**Outputs of this notebook**

| File | Contents |
|---|---|
| `reports/cleaning_audit.json` | Quality audit and cleaning decisions |
| `reports/eda_summary.json` | All EDA tables |
| `reports/figures/*.png` | EDA charts |
| `reports/preprocessing_report.json` | Feature lists, checks, validation drift |
| `artifacts/preprocessor_{linear,tree}.joblib` | Fitted preprocessors (reused for scoring and the dashboard) |
| `data/processed/{linear,tree}_{train,val,test}.parquet` | Model-ready matrices |

**Key findings** *(to be written once the notebook has been run on the full dataset)*

1. …
2. …
3. …
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}
out = Path("notebooks/01_cleaning_eda_preprocessing.ipynb")
out.write_text(json.dumps(nb, indent=1))
print("Wrote", out, "with", len(cells), "cells")
