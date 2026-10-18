FILENAME = "05_modelling.ipynb"

CELLS = [
("md", r"""
# 5 · Modelling

| | |
|---|---|
| Lifecycle stage | 5 of 8: *Modelling* (feature engineering, algorithm comparison, tuning, cross-validation) |
| Tools | scikit-learn, XGBoost, LightGBM, CatBoost (comparison only), MLflow |
| Code | `preprocess.py`, `models.py`, `tuning.py`, `train.py`, `selection.py`, `tracking.py` |
| Outputs | `artifacts/model_*.joblib`, `artifacts/fraudlens_model.joblib` (champion), `reports/model_results.json`, MLflow runs and registered model `fraudlens-champion` |

**Protocol** (fixed before training): tune with *time-series* cross-validation inside the training period,
fit on train, choose the champion and its threshold on validation, and score the test period **once**.
"""),
("setup", r"""
from fraudlens import models as M
from fraudlens import tuning
from fraudlens.train import load_view
res = read_json(P["reports"] / "model_results.json")
print(res["protocol"])"""),
("md", r"""
## 5.1 Feature engineering for two model families

`preprocess.py` builds two views of the same gold features, both fitted on **train only**:

* **tree view** (gradient boosting): raw numeric features plus native categorical codes; no scaling needed.
* **linear view** (logistic regression, Isolation Forest): log transforms, train-fitted quantile clipping,
  standardisation, cyclical encoding of hour/day (sin/cos), one-hot categories, missing-value indicators.
"""),
("code", r"""
pre = read_json(P["reports"] / "preprocessing_report.json")
print("fit on:", pre["fit_on"], "| class counts (train):", pre["class_counts_train"],
      "| suggested scale_pos_weight:", pre["suggested_scale_pos_weight"])
print("excluded features:", pre["excluded_features"], "| added:", pre["added_features"])
for name, v in pre["views"].items():
    print(f"\n{name} view: {v['n_features']} features")
    print(", ".join(v["feature_names"]))
"""),
("md", r"""
## 5.2 Candidate algorithms and class imbalance

Imbalance is handled with **class weights** (no SMOTE or other resampling): weighting keeps the real
data distribution, so scores stay meaningful as fraud rates, and it is cheap on 1.2 M rows.

| Model | Why it is a candidate |
|---|---|
| Logistic regression | Transparent baseline; tells us how much non-linearity buys |
| HistGradientBoosting (scikit-learn) | Fast gradient boosting with native categorical support, no extra dependency |
| XGBoost, LightGBM | State-of-the-art boosted trees for tabular data |
| Isolation Forest | Unsupervised anomaly detector trained on legitimate transactions only; finds "unusual", not "fraud-like" |
| Hybrid | HistGradientBoosting with the Isolation Forest anomaly score as an extra feature |
"""),
("code", r"""
cands = pd.DataFrame({"library": {m: r["library"] for m, r in res["results"].items()},
                      "view": {m: r["view"] for m, r in res["results"].items()},
                      "imbalance handling": res["imbalance_handling"],
                      "train seconds": res["train_seconds"]})
cands
"""),
("md", r"""
## 5.3 Hyper-parameter tuning with time-series cross-validation

Random K-fold would train on the future. `tuning.time_folds` cuts the (time-ordered) training period into
blocks and validates each block on a model trained on everything **before** it (expanding window).
"""),
("code", r"""
folds = tuning.time_folds(1_185_094, 3)
fig, ax = plt.subplots(figsize=(9, 1.9))
for i, (tr, va) in enumerate(folds):
    ax.barh(i, len(tr), color=viz.SERIES[0]); ax.barh(i, len(va), left=len(tr), color=viz.SERIES[1])
ax.set_yticks(range(3), [f"fold {i + 1}" for i in range(3)]); ax.invert_yaxis()
ax.set_xlabel("training rows in time order"); ax.set_title("Expanding-window CV: blue = fit, orange = validate")
plt.show()

rows = []
for m, t in res["tuning"].items():
    for c in t["candidates"]:
        rows.append({"model": m, "candidate": c["candidate"], **c["params"],
                     "cv_pr_auc_mean": c.get("cv_pr_auc_mean"), "cv_pr_auc_std": c.get("cv_pr_auc_std")})
tun = pd.DataFrame(rows)
print({m: {"best": t["best_params"], "cv_pr_auc": round(t["best_cv_pr_auc_mean"], 4), "±": round(t["best_cv_pr_auc_std"], 4)}
       for m, t in res["tuning"].items()})
tun.sort_values(["model", "cv_pr_auc_mean"], ascending=[True, False]).groupby("model").head(4)
"""),
("md", r"""
## 5.4 Live algorithm comparison (time-series CV)

A compact re-run of the comparison so it can be reproduced in minutes: every 4th training transaction
(time order preserved), three expanding-window folds, default settings, and **CatBoost** added as an
extra contender. The full-size, tuned comparison from the pipeline follows in 5.5.
"""),
("code", r"""
import time
from sklearn.metrics import average_precision_score

tree, lin = load_view(cfg, "tree")["train"], load_view(cfg, "linear")["train"]
step = 4
Xt, yt = tree["X"].iloc[::step].reset_index(drop=True), tree["y"][::step]
Xl = lin["X"].iloc[::step].reset_index(drop=True)
cat_cols = [c for c in M.CATEGORICAL_IN_TREE_VIEW if c in Xt.columns]
print(f"{len(Xt):,} rows, {int(yt.sum()):,} frauds")

def make(name):
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=300, learning_rate=0.1, depth=6, auto_class_weights="Balanced",
                                  cat_features=cat_cols, verbose=0, random_seed=42, thread_count=4,
                                  allow_writing_files=False)
    m = M.build(name, 42, categorical=cat_cols, scale_pos_weight=pre["suggested_scale_pos_weight"])
    if name == "lightgbm":
        m.set_params(n_jobs=2)   # avoids OpenMP thread contention with XGBoost in the same process
    return m

import importlib.util
names = ["logistic", "hist_gb", "xgboost", "lightgbm"] + (["catboost"] if importlib.util.find_spec("catboost") else [])
rows = []
for name in names:
    X = Xl if M.VIEW_BY_MODEL.get(name) == "linear" else Xt
    for k, (tr, va) in enumerate(tuning.time_folds(len(X), 3)):
        t0 = time.perf_counter()
        model = make(name)
        M.fit(model, name, X.iloc[tr], yt[tr], categorical=cat_cols if name == "lightgbm" else None)
        ap = average_precision_score(yt[va], M.positive_scores(model, X.iloc[va]))
        rows.append({"model": name, "fold": k + 1, "pr_auc": ap, "seconds": time.perf_counter() - t0})
cv = pd.DataFrame(rows)
cv_summary = cv.groupby("model").agg(pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
                                     seconds=("seconds", "sum")).sort_values("pr_auc_mean", ascending=False)
cv_summary
"""),
("code", r"""
fig, ax = plt.subplots(figsize=(8, 3.4))
order = cv_summary.index.tolist()
for i, m in enumerate(order):
    v = cv.loc[cv["model"] == m, "pr_auc"]
    ax.scatter(v, [i] * len(v), color=viz.model_color(m) if m != "catboost" else viz.SERIES[6], s=40)
    ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color=viz.INK)
ax.set_yticks(range(len(order)), order); ax.invert_yaxis()
ax.set_xlabel("PR-AUC per fold (line = mean)"); ax.set_title("Time-series CV on a 1-in-4 sample of the training period")
plt.show()
"""),
("md", r"""
## 5.5 Full comparison from the pipeline (tuned, full training data)

Validation PR-AUC decides; test is shown for completeness but was not used for any choice.
"""),
("code", r"""
comp = pd.DataFrame(res["comparison"]).set_index("model")
display(comp)
fig, ax = plt.subplots(figsize=(8, 3.6))
c2 = comp.sort_values("val_pr_auc")
ax.barh(np.arange(len(c2)) + 0.2, c2["val_pr_auc"], height=0.4, color=viz.SERIES[0], label="validation")
ax.barh(np.arange(len(c2)) - 0.2, c2["test_pr_auc"], height=0.4, color=viz.SERIES[1], label="test")
ax.set_yticks(range(len(c2)), c2.index); ax.set_xlabel("PR-AUC"); ax.legend(loc="lower right")
ax.set_title("Supervised boosted trees dominate; the anomaly detector alone is far behind")
plt.show()
"""),
("md", r"""
## 5.6 Champion selection: the simplest non-inferior model

Rule (registered in advance in `selection.py`): rank models by complexity
(`logistic < hist_gb < lightgbm < xgboost < hybrid`) and choose the **simplest** one whose validation
PR-AUC is not worse than the best by more than **0.01**, judged by the upper bound of a 95% paired
bootstrap confidence interval of the gap. This avoids paying operational complexity (extra libraries,
a second model in the hybrid) for differences that are within noise.
"""),
("code", r"""
sel = res["selection"]
print(sel["reason"])
gaps = pd.DataFrame(sel["comparisons"]).T.loc[[m for m in sel["complexity_order"] if m in sel["comparisons"]]]
display(gaps)
fig, ax = plt.subplots(figsize=(7, 2.8))
for i, (m, r) in enumerate(gaps.iterrows()):
    ax.errorbar(r["gap_mean"], i, xerr=[[r["gap_mean"] - r["ci95_low"]], [r["ci95_high"] - r["gap_mean"]]],
                fmt="o", color=viz.SERIES[1] if m == res["champion"] else viz.MUTED, capsize=4)
ax.axvline(sel["margin"], ls="--", color=viz.INK_2); ax.annotate(" margin 0.01", (sel["margin"], len(gaps) - 0.5))
ax.set_yticks(range(len(gaps)), gaps.index); ax.invert_yaxis(); ax.set_xlim(-0.005, 0.04)
ax.set_xlabel("PR-AUC gap to the best model (95% bootstrap CI)"); ax.set_title(f"Champion: {res['champion']}")
plt.show()
"""),
("md", r"""
## 5.7 The deployable champion

The champion, its fitted preprocessor and the validation-chosen threshold are saved together as one
`FraudModel` object, so training and serving cannot drift apart. The pipeline also proves that the saved
object scores the stored training matrices **exactly** (train/serve parity test).
"""),
("code", r"""
from fraudlens.scoring import FraudModel
fm = FraudModel.load(P["artifacts"] / "fraudlens_model.joblib")
print(f"champion={fm.champion}  view={fm.view}  threshold={fm.threshold:.4f} ({fm.threshold_strategy})")
print(fm.model)
"""),
("md", r"""
## 5.8 Experiment tracking: MLflow

Every training run is logged to MLflow (parameters, nested tuning runs, metrics, figures, the model), and
the champion is registered as `fraudlens-champion` with the alias `champion`.
"""),
("code", r"""
import mlflow
from mlflow.tracking import MlflowClient
info = res.get("mlflow", {})
uri = f"sqlite:///{(ROOT / 'mlruns' / 'mlflow.db').as_posix()}"
if (ROOT / "mlruns" / "mlflow.db").exists():
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=[cfg["platform"]["mlflow_experiment"]], max_results=200)
    cols = [c for c in ["run_id", "tags.mlflow.runName", "status", "metrics.val_pr_auc", "metrics.test_pr_auc"] if c in runs]
    print(f"{len(runs)} runs in experiment '{cfg['platform']['mlflow_experiment']}'")
    display(runs[cols].head(12))
    client = MlflowClient()
    for v in client.search_model_versions(f"name='{cfg['platform']['registered_model']}'"):
        print(f"registered: {v.name} v{v.version} (run {v.run_id[:8]}) aliases={getattr(v, 'aliases', [])}")
else:
    print("No local MLflow store here; tracking info from the run:", info)
"""),
("md", r"""
## 5.9 Summary

* Boosted trees reach validation PR-AUC ≈ 0.98 (lift ≈ 164× over random); logistic regression ≈ 0.68 and
  the unsupervised Isolation Forest ≈ 0.36, so the patterns are strongly non-linear but well captured by trees.
* LightGBM, XGBoost, HistGradientBoosting and the hybrid are statistically within the 0.01 margin; the rule
  picks **HistGradientBoosting** (scikit-learn only, smallest artefact, fastest to serve).
* The anomaly score adds nothing measurable on top of the supervised model on this dataset; it remains
  valuable as a detector of *new* patterns and is monitored, but it is not in the deployed path.

**Next:** [06 · Evaluation & interpretation](06_evaluation_interpretation.ipynb)
"""),
]
