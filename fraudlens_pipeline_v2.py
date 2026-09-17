# ============================================================
# FraudLens — Pipeline for fraudTest.csv (Sparkov-style dataset)
# Covers workflow steps: 1 (gather data), 2 (store in DB), 6 (build/evaluate models)
# ============================================================

import pandas as pd
import numpy as np
import sqlite3

# ------------------------------------------------------------
# STEP 1: LOAD THE DATASET
# ------------------------------------------------------------
df = pd.read_csv("fraudTest.csv")
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

print("Raw shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Fraud rate: {:.4f}%".format(df["is_fraud"].mean() * 100))

# ------------------------------------------------------------
# STEP 2: FEATURE ENGINEERING
# ------------------------------------------------------------
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["dob"] = pd.to_datetime(df["dob"])

# Age at time of transaction
df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365

# Time-based features
df["trans_hour"] = df["trans_date_trans_time"].dt.hour
df["trans_day"] = df["trans_date_trans_time"].dt.day
df["trans_month"] = df["trans_date_trans_time"].dt.month
df["trans_dow"] = df["trans_date_trans_time"].dt.dayofweek

# Distance between cardholder location and merchant location (Haversine, km)
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))

df["distance_km"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])

# ------------------------------------------------------------
# STEP 3: DROP DIRECT IDENTIFIERS (privacy — these are simulated but still dropped)
# ------------------------------------------------------------
drop_cols = ["first", "last", "street", "cc_num", "trans_num", "dob",
             "trans_date_trans_time", "city", "job", "zip",
             "lat", "long", "merch_lat", "merch_long", "unix_time"]
df_clean = df.drop(columns=drop_cols)

print("\nCleaned shape:", df_clean.shape)
print("Columns:", df_clean.columns.tolist())

# ------------------------------------------------------------
# STEP 4: ENCODE CATEGORICAL FEATURES
# ------------------------------------------------------------
df_clean["gender"] = df_clean["gender"].map({"M": 0, "F": 1})

# One-hot encode category (14 categories) and state (low-mid cardinality)
df_encoded = pd.get_dummies(df_clean, columns=["category", "state"], drop_first=True)

# Merchant has high cardinality — frequency-encode instead of one-hot
merchant_freq = df_encoded["merchant"].value_counts(normalize=True)
df_encoded["merchant_freq"] = df_encoded["merchant"].map(merchant_freq)
df_encoded = df_encoded.drop(columns=["merchant"])

print("\nFinal encoded shape:", df_encoded.shape)

# ------------------------------------------------------------
# STEP 5: STORE IN A STRUCTURED DATABASE (SQLite)
# ------------------------------------------------------------
conn = sqlite3.connect("fraudlens.db")
df_encoded.to_sql("transactions", conn, if_exists="replace", index=False)

check = pd.read_sql("""
    SELECT COUNT(*) AS total_transactions,
           SUM(is_fraud) AS fraud_count,
           ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4) AS fraud_percentage
    FROM transactions
""", conn)
print("\n--- Database verification query ---")
print(check)

df_db = pd.read_sql("SELECT * FROM transactions", conn)
conn.close()

# ------------------------------------------------------------
# STEP 6: TRAIN/TEST SPLIT + SCALING
# ------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df_db.drop(columns=["is_fraud"])
y = df_db["is_fraud"]

num_cols = ["amt", "city_pop", "age", "distance_km", "merchant_freq"]
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("\nTrain size:", X_train.shape, " Test size:", X_test.shape)

# ------------------------------------------------------------
# STEP 7: BALANCE TRAINING DATA WITH SMOTE
# ------------------------------------------------------------
from imblearn.over_sampling import SMOTE

print("Before SMOTE:", y_train.value_counts().to_dict())
sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
print("After SMOTE: ", y_train_bal.value_counts().to_dict())

# ------------------------------------------------------------
# STEP 8: TRAIN & EVALUATE MULTIPLE MODELS
# ------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
}

results = []
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_bal, y_train_bal)
    trained_models[name] = model

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "Precision": round(precision_score(y_test, preds), 4),
        "Recall": round(recall_score(y_test, preds), 4),
        "F1": round(f1_score(y_test, preds), 4),
        "ROC-AUC": round(roc_auc_score(y_test, probs), 4),
    })

results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
print("\n=== Model Comparison ===")
print(results_df.to_string(index=False))
results_df.to_csv("model_comparison.csv", index=False)

# ------------------------------------------------------------
# STEP 9: CONFUSION MATRIX FOR BEST MODEL
# ------------------------------------------------------------
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
best_preds = best_model.predict(X_test)

print(f"\n=== Best model: {best_model_name} ===")
print(confusion_matrix(y_test, best_preds))
print(classification_report(y_test, best_preds, digits=4))

print("\nDone. Files created: fraudlens.db, model_comparison.csv")
