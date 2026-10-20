# Problem statement

## Context

Card issuers authorise millions of transactions a day. A small fraction are fraudulent, and every one that
goes through costs the issuer (or the customer) money. Fraud teams cannot review every transaction, so they
need a system that **ranks transactions by fraud risk, raises a manageable number of alerts, and explains
each alert**, and that keeps working as behaviour changes.

## Objectives (from the project brief)

1. Build a machine-learning system that identifies fraudulent transactions.
2. Improve detection while **minimising false-positive alerts**.
3. Provide an **interactive dashboard** for transaction monitoring and reporting.
4. Show the full flow **from data collection to visualisation**.
5. Support **quicker decision-making** for financial institutions.

## Users

| User | Need |
|---|---|
| Fraud analyst | a ranked alert queue, the reasons for each alert, a way to record the verdict |
| Fraud-operations manager | alert volume, catch rate, cost and benefit; the effect of moving the threshold |
| Model risk / governance | evidence that the model is valid, fair, reproducible and monitored |
| Data platform team | a pipeline that runs on open-source infrastructure |

## Formal problem

For each transaction *x* at time *t*, using only information available at *t* (the transaction and the card's
earlier history), produce a score *s(x)* such that ranking by *s* puts fraud first, and an alert rule
*s(x) ≥ τ* where the threshold *τ* maximises

```
net benefit = 0.9 × (fraud $ in alerts) − 15 × (number of alerts)
```

on the validation period. Constraints: no information from the future (chronological split, prior-only
features), interpretable alerts, a daily alert volume a small team can clear.

## Success criteria (set before modelling, checked on the untouched test period)

| Criterion | Target | Achieved (test, Jun-Dec 2020) |
|---|---|---|
| Recall | ≥ 0.90 | **0.962** (95% CI 0.953-0.969) |
| Precision | ≥ 0.50 | **0.724** (95% CI 0.708-0.739) |
| Alerts per day | ≤ 20 | **14.7** |
| PR-AUC lift over random | ≥ 50× | **251×** (PR-AUC 0.970 vs baseline 0.0039) |

All four are checked automatically by `python -m fraudlens.verify` (stage *1-problem* of the verification report).

## Data

Sparkov simulated card transactions (Kaggle `kartik2112/fraud-detection`, CC0): 1,852,394 transactions by
~1,000 customers at ~800 merchants, 1 Jan 2019 to 31 Dec 2020, label `is_fraud` (0.52% overall).
Simulated data has no real personal information but cleaner patterns than production data; results are
therefore optimistic in absolute terms.

## Scope

In scope: batch and real-time scoring, explanation, dashboard, monitoring, the open-source data platform,
CI/CD and deployment. Out of scope: integration with a real authorisation system, case management, deep
learning models (not required at this stage by the supervisor).
