FILENAME = "03_data_preparation.ipynb"

CELLS = [
("md", r"""
# 3 · Data preparation

| | |
|---|---|
| Lifecycle stage | 3 of 8: *Data preparation* (clean, transform, validate) |
| Tools | pandas, Dask, PyArrow (Parquet) |
| Code | `validate.py` (data contract), `clean.py` (silver), `quality.py` (audit), `features.py`, `split.py`, `prepare.py` |
| Outputs | `data/silver/transactions.parquet`, `data/gold/features_{train,val,test}.parquet`, `reports/validation_report.json`, `reports/cleaning_audit.json` |

```
bronze CSVs ──► data contract ──► silver (typed, de-duplicated, PII removed, card id hashed)
            ──► features (only each card's PRIOR transactions) ──► chronological split ──► gold
```
"""),
("setup", r"""
from fraudlens import features as F
from fraudlens.clean import SILVER_COLUMNS, hash_card_numbers
from fraudlens.datasets import load_features, load_silver"""),
("md", r"""
## 3.1 Validate: the data contract

Before anything is transformed, each raw file is checked against a contract (`validate.py`): required
columns, types, value ranges, nulls, known categories, unique transaction ids. Failures stop the pipeline;
warnings are recorded and handled in cleaning.
"""),
("code", r"""
val = read_json(P["reports"] / "validation_report.json")
for rep in val:
    print(f"{rep['dataset']}: {rep['status']}  rows={rep['profile']['rows']:,}  fraud rate={rep['profile']['fraud_rate_pct']}%")
pd.DataFrame([{**c, "dataset": rep["dataset"]} for rep in val for c in rep["checks"]])[["dataset", "check", "status", "detail"]]
"""),
("md", r"""
## 3.2 Clean: the silver layer

* drop the leftover index column, parse timestamps and dates, cast the label to `int8`;
* remove exact duplicate transaction ids and impossible rows (none were found in this dataset);
* strip the generator's `fraud_` prefix from every merchant name (an artefact, not a label);
* **privacy:** drop `first`, `last`, `street`; replace the card number with a salted SHA-256 `card_id`.
"""),
("code", r"""
run = read_json(P["reports"] / "pipeline_run.json")
display(pd.DataFrame(run["cleaning"]))
demo = pd.Series([4263982640269299, 4263982640269299, 30074693890476])
print("hash_card_numbers ->", hash_card_numbers(demo, salt="demo-salt").tolist(), "(same card -> same id, not reversible)")
silver = load_silver(cfg)
print(f"silver: {len(silver):,} rows x {silver.shape[1]} columns, {silver.memory_usage(deep=True).sum() / 1e6:,.0f} MB in memory")
assert not {"first", "last", "street", "cc_num"} & set(silver.columns)
silver.dtypes.to_frame("dtype").T
"""),
("md", r"""
## 3.3 Data-quality audit and cleaning decisions

`quality.py` profiles the silver data and records every cleaning decision with its reason. The key one:
**amount outliers are kept**, because three quarters of all fraud sits in the top 5% of amounts; removing
"outliers" would remove the signal. Scale-sensitive models get train-fitted clipping instead.
"""),
("code", r"""
audit = read_json(P["reports"] / "cleaning_audit.json")
display(pd.DataFrame(audit["outliers_train"]).T)
display(pd.Series(audit["consistency"]).to_frame("value").T)
pd.DataFrame(audit["decisions"])
"""),
("md", r"""
## 3.4 Scaling out with Dask

The raw files fit in memory on a laptop, but the same preparation has to work when they do not. Dask
reads the CSVs in parallel 64 MB partitions with the pandas API. Here it recomputes monthly volume and
fraud counts straight from the raw files, and the result must match the pandas silver layer exactly
(no rows were removed). At production scale the same logic would run on a Dask cluster or PySpark.
"""),
("code", r"""
import time
import dask
import dask.dataframe as dd

raw_files = [str(P["raw"] / f) for f in cfg["dataset"]["files"].values()]
t0 = time.perf_counter()
ddf = dd.read_csv(raw_files, usecols=["trans_date_trans_time", "amt", "is_fraud"], blocksize="64MB")
ddf = ddf.assign(month=ddf["trans_date_trans_time"].str.slice(0, 7))
monthly_dask = ddf.groupby("month").agg({"is_fraud": ["count", "sum"], "amt": "sum"}).compute()
monthly_dask.columns = ["transactions", "frauds", "amount"]
monthly_dask = monthly_dask.sort_index()
print(f"Dask: {ddf.npartitions} partitions, {time.perf_counter() - t0:.1f}s, dask {dask.__version__}")

monthly_pd = (silver.assign(month=silver["ts"].dt.strftime("%Y-%m"), is_fraud=silver["is_fraud"].astype("int64")).groupby("month")
              .agg(transactions=("is_fraud", "size"), frauds=("is_fraud", "sum"), amount=("amount", "sum")))
same = (monthly_dask[["transactions", "frauds"]].astype(int).equals(monthly_pd[["transactions", "frauds"]].astype(int))
        and np.allclose(monthly_dask["amount"], monthly_pd["amount"]))
print("Dask (raw CSV) and pandas (silver) agree:", same)
monthly_pd.assign(fraud_rate_pct=lambda t: 100 * t["frauds"].astype(float) / t["transactions"]).head(24)
"""),
("md", r"""
## 3.5 Transform: leakage-safe features

Every behavioural feature is computed from the card's **previous** transactions only, exactly as a live
system would see them at authorisation time:

| Group | Features | Hypothesis (stage 1) |
|---|---|---|
| Transaction | `amount`, `log_amount`, `hour`, `day_of_week`, `is_weekend`, `is_night`, `category` | H1, H2, H4 |
| Customer | `age`, `gender`, `log_city_pop` | H5 |
| Card history | `amt_to_card_mean`, `amt_zscore_card`, `secs_since_last_txn`, `category_share_card` | H1, H4 |
| Velocity (1 h / 24 h / 7 d) | `txn_count_*`, `amt_sum_*` | H3 |

Running totals (`card_prior_txn_count`, ...) only ever grow, so later periods contain values never seen
in training; they were measured as drifting and excluded (`preprocess.exclude_features`).
"""),
("code", r"""
train = load_features(cfg, "train")
num = F.numeric_features(cfg["features"]["velocity_windows"])
print(f"gold features (train): {len(train):,} rows, {train.shape[1]} columns")
train[["transaction_id", "ts", "card_id", "amount", "category", "amt_to_card_mean", "secs_since_last_txn",
       "txn_count_1h", "amt_sum_24h", "txn_count_7d", "is_fraud"]].head()
"""),
("md", r"""
### Proof that the velocity features use only the past

Recompute `txn_count_24h` and `amt_sum_24h` for one card by brute force (a loop over the card's earlier
transactions) and compare with the vectorised pipeline values.
"""),
("code", r"""
card = train["card_id"].value_counts().index[0]
c = train[train["card_id"] == card].sort_values(["ts", "transaction_id"]).head(400).reset_index(drop=True)
t = ((c["ts"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy()
brute_n = [int(((t[:j] >= t[j] - 86400)).sum()) for j in range(len(c))]
brute_s = [float(c["amount"].to_numpy()[:j][t[:j] >= t[j] - 86400].sum()) for j in range(len(c))]
print("txn_count_24h matches:", np.array_equal(brute_n, c["txn_count_24h"].to_numpy()))
print("amt_sum_24h matches:  ", np.allclose(brute_s, c["amt_sum_24h"].to_numpy()))
"""),
("md", r"""
## 3.6 Split by time, never at random

Kaggle's `fraudTest.csv` continues `fraudTrain.csv` in time, so it becomes the **test** period. The last
six weeks of `fraudTrain.csv` become the **validation** period, used to pick the champion and its
threshold. A random split would let the model learn from the future.
"""),
("code", r"""
sp = pd.DataFrame(run["split"]["splits"]).T
display(sp)
fig, ax = plt.subplots(figsize=(10, 2.4))
for i, (name, r) in enumerate(sp.iterrows()):
    start, end = pd.Timestamp(r["start"]), pd.Timestamp(r["end"])
    ax.barh(0, (end - start).days, left=start, color=viz.SERIES[i], label=f"{name} ({int(r['rows']):,} rows)")
ax.set_yticks([]); ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.45)); ax.set_title("Chronological split")
plt.show()
"""),
("code", r"""
v = read_json(P["reports"] / "verification_report.json")
pd.DataFrame([c for c in v["checks"] if c["stage"] == "3-preparation"])[["check", "status", "detail"]]
"""),
("md", r"""
## 3.7 Summary

* The data contract passes (two warnings handled in cleaning); no duplicates or impossible rows.
* Silver is typed, privacy-reduced (no names, streets or card numbers) and stored as Parquet.
* Features are provably leakage-safe; the split is chronological; verification checks prove both.
* Dask reproduces the same aggregates from the raw files, so the preparation scales beyond memory.

**Next:** [04 · Exploratory data analysis](04_eda.ipynb)
"""),
]
