"""Colours and labels shared by every chart and by the dashboard (no plotting imports)."""
from __future__ import annotations

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
LEGIT, FRAUD = SERIES[0], SERIES[1]
STATUS = {"caught": "#0ca30c", "false_alert": "#fab219", "missed": "#d03b3b", "cleared": "#b9b8b1"}
OUTCOME_LABEL = {"caught": "Fraud caught", "false_alert": "False alert", "missed": "Fraud missed",
                 "cleared": "Cleared"}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
DIVERGING = "RdBu_r"

MODEL_ORDER = ["logistic", "hist_gb", "xgboost", "lightgbm", "isolation_forest", "hybrid_gb"]
MODEL_LABEL = {"logistic": "Logistic regression", "hist_gb": "Gradient boosting (sklearn)",
               "xgboost": "XGBoost", "lightgbm": "LightGBM", "isolation_forest": "Isolation Forest",
               "hybrid_gb": "Hybrid gradient boosting"}


FEATURE_LABEL = {
    "amount": "Amount ($)", "log_amount": "log(amount)", "hour": "Hour of day", "day_of_week": "Day of week",
    "is_weekend": "Weekend", "is_night": "Night-time (22:00-03:59)", "age": "Cardholder age",
    "log_city_pop": "log(city population)", "secs_since_last_txn": "Seconds since card's previous transaction",
    "amt_to_card_mean": "Amount / card's average", "amt_zscore_card": "Amount z-score for the card",
    "txn_count_1h": "Card transactions, past 1h", "txn_count_24h": "Card transactions, past 24h",
    "txn_count_7d": "Card transactions, past 7 days", "amt_sum_1h": "Card spend, past 1h ($)",
    "amt_sum_24h": "Card spend, past 24h ($)", "amt_sum_7d": "Card spend, past 7 days ($)",
    "category_share_card": "Share of card's history in this category", "category": "Merchant category",
    "gender": "Gender", "anomaly_score": "Isolation Forest anomaly score",
}


def feature_label(name: str) -> str:
    return FEATURE_LABEL.get(name, name.replace("_", " "))
