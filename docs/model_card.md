# Model card: `fraudlens-champion`

| | |
|---|---|
| Model | `HistGradientBoostingClassifier` (scikit-learn 1.9.1), native categorical splits, `class_weight="balanced"` |
| Hyper-parameters | learning_rate 0.1, max_leaf_nodes 15, min_samples_leaf 50, l2_regularization 1.0, early stopping (tuned by 3-fold time-series CV) |
| Input | 20 features from the transaction and the card's earlier history (see the data dictionary) |
| Output | a fraud score in [0, 1] and an alert when score ≥ **0.7397** |
| Artefact | `artifacts/fraudlens_model.joblib` / `dashboard/bundle/model.joblib` (preprocessor + model + threshold); MLflow registered model `fraudlens-champion`, alias `champion` |
| Owner | FraudLens project team (PRT661) |

## Intended use

Ranking card transactions by fraud risk so a fraud-operations team reviews the riskiest ones first; each alert
comes with its top three SHAP reasons. It is a **decision-support** tool: an analyst decides, the model does not
block cards on its own.

Not intended for: decisions about individual customers beyond transaction review, credit decisions, or use on
real card data without retraining and re-validation.

## Training data and protocol

Sparkov simulated transactions (CC0). Chronological split:

| Split | Period | Transactions | Fraud |
|---|---|---|---|
| train | 1 Jan 2019 - 10 May 2020 | 1,185,094 | 6,837 (0.58%) |
| validation | 10 May - 21 Jun 2020 | 111,581 | 669 (0.60%) |
| test | 21 Jun - 31 Dec 2020 | 555,719 | 2,145 (0.39%) |

Tuning used time-series CV inside train; the champion and the threshold were chosen on validation; the test
period was scored once.

## Selection

Six candidates (logistic regression, HistGradientBoosting, XGBoost, LightGBM, Isolation Forest, and a hybrid of
HistGradientBoosting with the Isolation Forest score). LightGBM had the highest validation PR-AUC (0.9878);
HistGradientBoosting (0.9820) is non-inferior within the pre-registered 0.01 margin (95% paired-bootstrap upper
bound of the gap 0.0084) and is the simplest such model, so it was selected.

## Performance (test period, threshold 0.7397)

| Metric | Value | 95% bootstrap CI |
|---|---|---|
| PR-AUC | 0.970 (baseline 0.0039, lift 251×) | 0.963 - 0.975 |
| ROC-AUC | 0.9995 | |
| Recall | 0.962 | 0.953 - 0.969 |
| Precision | 0.724 | 0.708 - 0.739 |
| False-positive rate | 0.14% | 0.13% - 0.15% |
| Alerts per day | 14.7 | |
| Fraud dollars caught | 98.7% | |
| Net benefit (0.9 × caught − $15 × alerts) | $964,182 | $909k - $1,022k |

Monthly recall on the test period stays between 0.94 and 0.98, monthly precision between 0.67 and 0.76. Most missed fraud is low-value, where an alert
would cost more than it recovers.

## Explainability

Global and local SHAP explanations (`reports/figures/16-18`, `reports/explanations.json`). The strongest drivers:
amount, merchant category, the card's spend in the last 24 hours, night-time, and transaction velocity. TreeSHAP
values did not add up to the model output for this model's native categorical splits (max error 14 log-odds),
so SHAP's permutation explainer on the log-odds output is used (exactly additive).

## Fairness

`age` and `gender` are inputs. On the test period recall is high for every group but not identical: 0.947 for
women against 0.981 for men, with 15.7 against 12.5 false alarms per 10,000 transactions; across age bands recall
ranges from about 0.94 to 0.99 (notebook 06, section 6.7, and `reports/fairness.json`). Before any use on real data,
retrain without `gender` and compare, and keep the group metrics in monthly monitoring.

## Calibration

Scores rank well but are not calibrated probabilities (class weighting inflates them); the threshold is
expressed in score units and chosen on validation. Add isotonic calibration if probabilities are needed.

## Limitations and risks

* **Simulated data**: patterns are cleaner than real fraud; expect lower absolute performance on real data.
* **Concept drift**: fraud tactics change. Monthly monitoring flagged feature drift in Aug and Dec 2020
  (seasonal spending) without a performance drop; the policy recommends review/retraining in such months.
* **Cold start**: cards with no history get neutral history features; their scores rely on amount, time and
  category.
* **Feedback loop**: once deployed, only alerted transactions get analyst labels; keep sampling non-alerted
  transactions for labels (chargebacks) to avoid biasing retraining.

## Maintenance

Retrain when the monitoring policy fires (drift share > 30%, recall < 0.85 or precision < 0.40) and promote a
challenger only if it is non-inferior to the champion on the latest labelled period (see the monitoring runbook).
