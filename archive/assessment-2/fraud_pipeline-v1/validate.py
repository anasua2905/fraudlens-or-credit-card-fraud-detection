"""Data contract for the raw Sparkov files.

Every check returns PASS, WARN or FAIL. Any FAIL stops the pipeline, so bad
data never reaches the model or the dashboard silently.
"""
from __future__ import annotations

import pandas as pd

EXPECTED_COLUMNS = [
    "trans_date_trans_time", "cc_num", "merchant", "category", "amt",
    "first", "last", "gender", "street", "city", "state", "zip",
    "lat", "long", "city_pop", "job", "dob", "trans_num", "unix_time",
    "merch_lat", "merch_long", "is_fraud",
]
# Kaggle's CSVs carry a leftover pandas index column.
TOLERATED_EXTRA_COLUMNS = {"Unnamed: 0"}

NOT_NULL = ["trans_date_trans_time", "cc_num", "merchant", "category", "amt",
            "lat", "long", "merch_lat", "merch_long", "dob", "trans_num", "is_fraud"]

KNOWN_CATEGORIES = {
    "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
    "health_fitness", "home", "kids_pets", "misc_net", "misc_pos",
    "personal_care", "shopping_net", "shopping_pos", "travel",
}


class DataValidationError(RuntimeError):
    pass


def _check(name: str, ok: bool, detail, severity: str = "FAIL") -> dict:
    return {"check": name, "status": "PASS" if ok else severity, "detail": detail}


def validate_raw(df: pd.DataFrame, name: str) -> dict:
    checks: list[dict] = []
    cols = set(df.columns)

    missing = [c for c in EXPECTED_COLUMNS if c not in cols]
    checks.append(_check("required_columns_present", not missing, {"missing": missing}))
    extra = sorted(cols - set(EXPECTED_COLUMNS) - TOLERATED_EXTRA_COLUMNS)
    checks.append(_check("no_unexpected_columns", not extra, {"extra": extra}, "WARN"))
    if missing:  # later checks depend on these columns
        return _summarise(name, df, checks)

    nulls = {c: int(n) for c, n in df[NOT_NULL].isna().sum().items() if n}
    checks.append(_check("critical_columns_not_null", not nulls, nulls))

    ts = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    checks.append(_check("timestamps_parseable", int(ts.isna().sum()) == 0,
                         {"unparseable": int(ts.isna().sum()),
                          "min": ts.min(), "max": ts.max()}))

    dob = pd.to_datetime(df["dob"], errors="coerce")
    checks.append(_check("dob_parseable_and_before_txn",
                         int(dob.isna().sum()) == 0 and bool((dob < ts).all()),
                         {"unparseable": int(dob.isna().sum())}))

    bad_amt = int((df["amt"] <= 0).sum())
    checks.append(_check("amount_positive", bad_amt == 0,
                         {"non_positive": bad_amt, "min": df["amt"].min(), "max": df["amt"].max()}))

    bad_geo = int((~df["lat"].between(-90, 90) | ~df["long"].between(-180, 180)
                   | ~df["merch_lat"].between(-90, 90) | ~df["merch_long"].between(-180, 180)).sum())
    checks.append(_check("coordinates_in_range", bad_geo == 0, {"out_of_range_rows": bad_geo}))

    labels = sorted(df["is_fraud"].dropna().unique().tolist())
    checks.append(_check("label_is_binary", set(labels) <= {0, 1}, {"values": labels}))

    genders = sorted(df["gender"].dropna().unique().tolist())
    checks.append(_check("gender_values_known", set(genders) <= {"M", "F"}, {"values": genders}, "WARN"))

    unknown_cat = sorted(set(df["category"].dropna().unique()) - KNOWN_CATEGORIES)
    checks.append(_check("categories_known", not unknown_cat, {"unknown": unknown_cat}, "WARN"))

    dups = int(df["trans_num"].duplicated().sum())
    checks.append(_check("transaction_id_unique", dups == 0, {"duplicates": dups}, "WARN"))

    # unix_time in public copies of Sparkov is offset from the readable timestamp,
    # so the pipeline derives its own epoch from trans_date_trans_time instead.
    ok_ts = ts.notna()
    offset = ((ts[ok_ts] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)) - df.loc[ok_ts, "unix_time"]
    checks.append(_check("unix_time_consistent_with_timestamp", offset.nunique() <= 1 and offset.iloc[0] == 0,
                         {"distinct_offsets_seconds": int(offset.nunique()),
                          "example_offset_days": round(float(offset.median()) / 86400, 2),
                          "action": "unix_time ignored; epoch derived from trans_date_trans_time"},
                         "WARN"))

    return _summarise(name, df, checks)


def _summarise(name: str, df: pd.DataFrame, checks: list[dict]) -> dict:
    status = "FAIL" if any(c["status"] == "FAIL" for c in checks) else (
        "WARN" if any(c["status"] == "WARN" for c in checks) else "PASS")
    profile = {"rows": len(df), "columns": len(df.columns)}
    if "is_fraud" in df:
        profile["fraud_count"] = int(df["is_fraud"].sum())
        profile["fraud_rate_pct"] = round(100 * float(df["is_fraud"].mean()), 4)
    return {"dataset": name, "status": status, "profile": profile, "checks": checks}


def assert_valid(report: dict) -> None:
    if report["status"] == "FAIL":
        failed = [c for c in report["checks"] if c["status"] == "FAIL"]
        raise DataValidationError(f"{report['dataset']} failed validation: {failed}")
