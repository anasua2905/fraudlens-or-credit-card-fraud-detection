"""Builds notebooks/02_modelling.ipynb (run from project root)."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# FraudLens — Machine Learning Model Development

**Prerequisites:** run the Step 1 pipeline and the cleaning/EDA/preprocessing notebook first.

## Protocol

| Stage | Data | Purpose |
|---|---|---|
| Fit | Training split (Jan 2019 – May 2020) | Learn model parameters |
| Select | Validation split (May – Jun 2020) | Compare models, choose the alert threshold |
| Report | Test split (Jun – Dec 2020) | Scored **once**, with everything already fixed |

Test data never influences a choice made here. That is what makes the final numbers an estimate of
future performance rather than a description of the past.

## Models

| Model | Type | Input |
|---|---|---|
| `logistic` | Logistic regression, class-weighted | linear view |
| `hist_gb` | Histogram gradient boosting | tree view |
| `isolation_forest` | Unsupervised anomaly detection, fitted on legitimate transactions only | linear view |
| `hybrid_gb` | Gradient boosting **plus the anomaly score as a feature** | tree view |
| `mlp` | Neural network (optional, off by default: slow) | linear view |

## Why not accuracy

Fraud is 0.58% of training transactions. Predicting "legitimate" for everything scores 99.4% accuracy and
catches nothing. This notebook reports **PR-AUC** (with its random baseline), precision, recall, alerts per
day, false alarms per catch, and net benefit in dollars.
""")

code("""
import sys
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

import json
import pandas as pd
from IPython.display import Image, display

from fraud_pipeline import train as T
from fraud_pipeline.utils import load_config

pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", "{:,.4f}".format)

cfg = load_config(root=ROOT)
FIGURES = cfg["paths"]["figures"]
print("Models to train:", cfg["model"]["models"])
print("Threshold strategy:", cfg["model"]["threshold"]["primary"])
""")

md("""
## 1. Train and evaluate

This trains every model, picks thresholds on validation, scores the test split once, and saves models,
figures and `reports/model_results.json`. Expect roughly 3 to 6 minutes on the full dataset.

For a quick trial, set `model.max_train_rows` in `config/pipeline.yaml` (e.g. 200000) first.
""")

code("""
report = T.run_training(cfg)
print(f"Champion: {report['champion']}   |   runtime: {report['runtime_seconds']}s")
""")

md("""
## 2. Model comparison

Ranked by **validation** PR-AUC, which is the number that decided the champion. The test columns are
reported at the threshold chosen on validation.
""")

code("""
comparison = pd.DataFrame(report["comparison"]).set_index("model")
display(comparison)
""")

code("""
display(Image(filename=str(FIGURES / "10_pr_curves_test.png")))
""")

md("""
## 3. Choosing the operating threshold

Three strategies are computed on validation:

* `max_f1` — balances precision and recall with no business input
* `alert_budget` — as many alerts as the review team can handle per day
* `min_cost` — maximises recovered fraud minus review cost

The active one is `model.threshold.primary` in the config; costs are `investigation_cost` and `recovery_rate`.
""")

code("""
champion = report["champion"]
res = report["results"][champion]
display(pd.Series(res["thresholds_from_validation"], name="threshold").to_frame())
display(pd.DataFrame({"validation": res["validation"], "test": res["test"]}))
""")

code("""
display(Image(filename=str(FIGURES / "11_threshold_sweep_validation.png")))
display(pd.DataFrame(report["validation_sweep_head"])[
    ["threshold", "alerts_per_day", "precision", "recall", "false_alarms_per_catch", "net_benefit"]])
""")

md("""
The left panel is the false-positive trade-off in operational terms: more alerts per day means more fraud
caught and lower precision. The right panel shows where the extra alerts stop paying for themselves.
""")

md("""
## 4. What the champion learned
""")

code("""
display(Image(filename=str(FIGURES / "12_score_distribution_test.png")))
""")

code("""
if report["champion_importance"]:
    display(Image(filename=str(FIGURES / "13_permutation_importance.png")))
    display(pd.DataFrame(report["champion_importance"]).head(15))
else:
    print("Permutation importance is produced for gradient boosting champions only.")
""")

md("""
## 5. Error analysis

Which frauds still slip through, and which customers get bothered without cause. This is the evidence for
the "minimise false positives" objective, and it tells the dashboard which segments need attention.
""")

code("""
threshold = res["thresholds_from_validation"][report["threshold_strategy"]]
errors = T.error_breakdown(cfg, champion, "test", threshold)

print("Missed fraud (test)")
display(errors["missed_fraud_summary"])
print("False alerts (test)")
display(errors["false_alert_summary"])
""")

code("""
display(errors["by_category"])
""")

code("""
display(errors["by_amount_band"])
""")

code("""
display(errors["by_hour"])
""")

md("""
## 6. Comparison against the other thresholds

Same model, different operating points, all chosen on validation. Useful for the report: it shows the
recall-versus-workload decision explicitly instead of hiding it in one number.
""")

code("""
rows = {report["threshold_strategy"]: res["test"], **res["test_at_other_thresholds"]}
display(pd.DataFrame(rows).T[
    ["precision", "recall", "f1", "alerts_per_day", "false_alarms_per_catch",
     "fraud_amount_caught", "fraud_amount_missed", "net_benefit"]])
""")

md("""
---
# Summary

**Outputs**

| File | Contents |
|---|---|
| `reports/model_results.json` | All metrics, thresholds and the comparison table |
| `reports/figures/10-13_*.png` | PR curves, threshold sweep, score distribution, feature importance |
| `artifacts/model_*.joblib` | Fitted models, ready for the dashboard |

**Findings** *(fill in after running on the full dataset)*

1. …
2. …
3. …

**Caution for the report.** Sparkov generates fraud with strong, simple rules (a night-time window and a
capped amount range), so scores here will be far above what a real fraud model achieves. The value of these
results is the method: honest time-based evaluation, thresholds tied to review capacity, and error analysis.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
out = Path("notebooks/02_modelling.ipynb")
out.write_text(json.dumps(nb, indent=1))
print("Wrote", out, "with", len(cells), "cells")
