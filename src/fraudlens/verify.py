"""End-to-end verification: does every stage agree with every other?

Each check compares two independent sources that must match, for example the
row count in the acquisition manifest against the rows in the feature tables,
or the metrics stored in model_results.json against a fresh re-scoring of the
packaged model. A FAIL means the reported results do not describe the system
on disk.

    python -m fraudlens.verify [--no-checksums]     (exit code 1 if any check fails)
"""
from __future__ import annotations

import json
import sys

import joblib
import numpy as np

from . import evaluate as E
from . import features as F
from . import models as M
from .acquire import sha256sum
from .datasets import load_features, load_silver
from .scoring import FraudModel
from .utils import get_logger, load_config, read_table, table_exists, write_json

log = get_logger()
SPLITS = ["train", "val", "test"]


class Checker:
    def __init__(self):
        self.checks: list[dict] = []

        self.criteria: list[dict] = []

    def check(self, stage: str, name: str, ok: bool, detail=None):
        ok = bool(ok)
        self.checks.append({"stage": stage, "check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
        (log.info if ok else log.error)("%-4s | %-15s | %s", "PASS" if ok else "FAIL", stage, name)

    def criterion(self, name: str, value, target, ok: bool):
        """Business success criteria are reported, not treated as integrity failures."""
        self.criteria.append({"criterion": name, "value": value, "target": target, "met": bool(ok)})
        (log.info if ok else log.warning)("%-4s | %-15s | %s (%s vs target %s)", "MET" if ok else "MISS",
                                          "1-problem", name, value, target)

    @property
    def passed(self) -> bool:
        return all(c["status"] == "PASS" for c in self.checks)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _days(ts) -> float:
    return max((ts.max() - ts.min()).total_seconds() / 86400, 1)


PARITY_COLUMNS = ["amt_to_card_mean", "amt_zscore_card", "secs_since_last_txn", "txn_count_1h", "txn_count_24h",
                  "txn_count_7d", "amt_sum_24h", "amt_sum_7d", "hour", "is_night", "age"]


def online_batch_parity(cfg: dict, silver, test_feats, n: int = 2000) -> dict:
    """Replay the first `n` test transactions through the online store, as the API would, and
    compare each feature with the batch value computed by the pipeline."""
    from .online import CardStore
    start = test_feats["ts"].min()
    store = CardStore.from_history(silver[silver["ts"] < start], cfg["features"]["velocity_windows"],
                                   cfg["features"]["night_hours"])
    stream = silver[silver["ts"] >= start].head(n)
    batch = test_feats.set_index("transaction_id")
    total = matched = 0
    worst: dict[str, int] = {}
    for txn in stream.to_dict("records"):
        f = store.features(txn)
        store.update(txn)
        b = batch.loc[txn["transaction_id"]]
        for col in PARITY_COLUMNS:
            total += 1
            if np.isclose(float(f[col]), float(b[col]), rtol=1e-6, atol=1e-6, equal_nan=True):
                matched += 1
            else:
                worst[col] = worst.get(col, 0) + 1
    return {"transactions": len(stream), "comparisons": total, "match_rate": round(matched / max(total, 1), 5),
            "mismatches_by_feature": worst}


def run_verification(cfg: dict, verify_checksums: bool = True) -> dict:
    P = cfg["paths"]
    c = Checker()

    # ---------------------------------------------------------- 1. problem
    crit = cfg.get("success_criteria", {})
    c.check("1-problem", "success criteria defined in config", bool(crit), crit)

    # ---------------------------------------------------------- 2. acquisition
    manifest = _load(P["raw"] / "manifest.json")
    files = manifest["files"]
    entries = files.values() if isinstance(files, dict) else files
    if verify_checksums:
        for f in entries:
            c.check("2-acquisition", f"{f['file']} checksum matches manifest",
                    sha256sum(P["raw"] / f["file"]) == f["sha256"])
    raw_rows = sum(f["rows"] for f in entries)
    c.check("2-acquisition", "dataset licence and source recorded", bool(manifest.get("licence") and manifest.get("source_url")))

    # ---------------------------------------------------------- 3. preparation
    validation = _load(P["reports"] / "validation_report.json")
    c.check("3-preparation", "raw files passed the data contract", all(v["status"] != "FAIL" for v in validation),
            [v["status"] for v in validation])
    silver = load_silver(cfg)
    run = _load(P["reports"] / "pipeline_run.json")
    removed = sum(s.get("duplicates_removed", 0) + s.get("invalid_rows_removed", 0) for s in run["cleaning"])
    c.check("3-preparation", "silver rows = manifest rows - removed rows", len(silver) == raw_rows - removed,
            {"manifest": raw_rows, "removed": removed, "silver": len(silver)})
    c.check("3-preparation", "no direct identifiers in silver",
            not {"cc_num", "first", "last", "street"} & set(silver.columns))
    feats = {s: load_features(cfg, s) for s in SPLITS}
    split_rows = {s: len(df) for s, df in feats.items()}
    c.check("3-preparation", "split rows sum to silver rows", sum(split_rows.values()) == len(silver), split_rows)
    ids = [set(df["transaction_id"]) for df in feats.values()]
    c.check("3-preparation", "no transaction appears in two splits",
            not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]))
    c.check("3-preparation", "splits are chronological (train < val < test)",
            feats["train"]["ts"].max() <= feats["val"]["ts"].min()
            and feats["val"]["ts"].max() <= feats["test"]["ts"].min())
    c.check("3-preparation", "no feature is derived from the label",
            not any("fraud" in f for f in F.numeric_features(cfg["features"]["velocity_windows"])))
    store_cards = read_table(P["artifacts"] / "card_state_cards", columns=["card_id", "n"])
    c.check("3-preparation", "online store holds every card and every transaction",
            len(store_cards) == silver["card_id"].nunique() and int(store_cards["n"].sum()) == len(silver),
            {"cards": len(store_cards), "transactions": int(store_cards["n"].sum())})

    # ---------------------------------------------------------- 4. EDA
    for f in ("eda_summary.json", "cleaning_audit.json"):
        c.check("4-eda", f"{f} written", (P["reports"] / f).exists())
    c.check("4-eda", "EDA figures written", len(list(P["figures"].glob("0*.png"))) >= 9,
            len(list(P["figures"].glob("0*.png"))))
    c.check("4-eda", "interactive EDA explorer written", (P["interactive"] / "eda_explorer.html").exists())

    # ---------------------------------------------------------- 5. modelling
    exclude = set(cfg["preprocess"].get("exclude_features", []))
    view_cols = {}
    for view in ["linear", "tree"]:
        pre = joblib.load(P["artifacts"] / f"preprocessor_{view}.joblib")
        names = list(pre.get_feature_names_out())
        view_cols[view] = names
        X = read_table(P["processed"] / f"{view}_train", columns=["transaction_id"])
        c.check("5-modelling", f"{view}: train matrix rows = feature rows", len(X) == split_rows["train"])
        c.check("5-modelling", f"{view}: excluded features absent", not (exclude & set(names)),
                sorted(exclude & set(names)))
    results = _load(P["reports"] / "model_results.json")
    champion = results["champion"]
    for r in results["comparison"]:
        name = r["model"]
        path = P["artifacts"] / f"model_{name}.joblib"
        c.check("5-modelling", f"{name}: saved model exists", path.exists())
        if not path.exists() or name not in M.SUPERVISED:
            continue
        model = joblib.load(path)
        expected = view_cols[M.VIEW_BY_MODEL[name]] + (["anomaly_score"] if name == "hybrid_gb" else [])
        got = list(getattr(model, "feature_names_in_", None) if getattr(model, "feature_names_in_", None) is not None
                   else expected)
        c.check("5-modelling", f"{name}: input features = preprocessed columns", got == expected)
        method = results["imbalance_handling"].get(name, "")
        c.check("5-modelling", f"{name}: imbalance handled by weighting (no resampling)",
                "weight" in str(method).lower(), method)
    tuned = results.get("tuning", {})
    c.check("5-modelling", "hyper-parameters tuned with time-series CV", bool(tuned) or not cfg["model"]["tuning"]["enabled"],
            sorted(tuned))
    sel = results["selection"]
    c.check("5-modelling", "reported champion = champion chosen by the rule", sel["champion"] == champion)
    order = sel["complexity_order"]
    equivalent = [m for m, v in sel["comparisons"].items() if v["equivalent_to_best"]]
    simplest = sorted(equivalent, key=lambda m: order.index(m) if m in order else len(order))[0]
    c.check("5-modelling", "champion is the simplest model equivalent to the best", champion == simplest,
            {"equivalent": equivalent, "champion": champion})

    # ---------------------------------------------------------- 6. evaluation
    tcfg = cfg["model"]["threshold"]
    strategy = results["threshold_strategy"]
    reported = results["results"][champion]
    fm = FraudModel.load(P["artifacts"] / "fraudlens_model.joblib")
    c.check("6-evaluation", "packaged model = reported champion", fm.champion == champion)
    c.check("6-evaluation", "packaged threshold = threshold chosen on validation",
            abs(fm.threshold - reported["thresholds_from_validation"][strategy]) < 1e-5)
    sv = fm.predict(feats["val"])
    st = fm.predict(feats["test"])
    yv, yt = feats["val"]["is_fraud"].to_numpy(), feats["test"]["is_fraud"].to_numpy()
    thr = E.choose_threshold(yv, sv, feats["val"]["amount"].to_numpy(), _days(feats["val"]["ts"]), tcfg, strategy)
    c.check("6-evaluation", "threshold re-derived on validation = reported",
            abs(thr - reported["thresholds_from_validation"][strategy]) < 1e-5,
            {"recomputed": round(float(thr), 6), "reported": reported["thresholds_from_validation"][strategy]})
    m = E.evaluate(yt, st, feats["test"]["amount"].to_numpy(), fm.threshold, _days(feats["test"]["ts"]), tcfg)
    r = reported["test"]
    c.check("6-evaluation", "re-scored test PR-AUC = reported", abs(m["pr_auc"] - r["pr_auc"]) < 1e-4,
            {"recomputed": m["pr_auc"], "reported": r["pr_auc"]})
    c.check("6-evaluation", "re-scored test confusion matrix = reported",
            (m["true_positives"], m["false_positives"], m["false_negatives"])
            == (r["true_positives"], r["false_positives"], r["false_negatives"]),
            {"recomputed": [m["true_positives"], m["false_positives"], m["false_negatives"]],
             "reported": [r["true_positives"], r["false_positives"], r["false_negatives"]]})
    c.check("6-evaluation", "re-scored net benefit = reported", abs(m["net_benefit"] - r["net_benefit"]) < 0.01)
    ci = results.get("champion_test_ci95", {})
    c.check("6-evaluation", "test PR-AUC inside its bootstrap 95% CI",
            bool(ci) and ci["pr_auc"]["low"] <= r["pr_auc"] <= ci["pr_auc"]["high"], ci.get("pr_auc"))
    expl = P["reports"] / "explanations.json"
    c.check("6-evaluation", "SHAP explanations written", expl.exists())
    if expl.exists():
        e = _load(expl)
        c.check("6-evaluation", "every test alert has SHAP reasons", e["alerts_explained"] == r["alerts"],
                {"explained": e["alerts_explained"], "alerts": r["alerts"]})
    for k, value in (("min_test_recall", r["recall"]), ("min_test_precision", r["precision"])):
        if k in crit:
            c.criterion(k, value, crit[k], value >= crit[k])
    if "max_alerts_per_day" in crit:
        c.criterion("max_alerts_per_day", r["alerts_per_day"], crit["max_alerts_per_day"],
                    r["alerts_per_day"] <= crit["max_alerts_per_day"])
    if "min_pr_auc_lift" in crit:
        lift = round(r["pr_auc"] / max(float(yt.mean()), 1e-9), 1)
        c.criterion("min_pr_auc_lift", lift, crit["min_pr_auc_lift"], lift >= crit["min_pr_auc_lift"])

    # ---------------------------------------------------------- 7. deployment
    meta_path = P["reports"] / "dashboard_meta.json"
    c.check("7-deployment", "batch scores generated", meta_path.exists() and table_exists(P["gold"] / "scored_transactions"))
    if meta_path.exists():
        meta = _load(meta_path)
        scored = read_table(P["gold"] / "scored_transactions", columns=["transaction_id", "split", "alert", "score"])
        c.check("7-deployment", "scores use the selected champion and threshold",
                meta["champion"] == champion and abs(meta["threshold"] - fm.threshold) < 1e-9)
        c.check("7-deployment", "scored rows = validation + test rows",
                len(scored) == split_rows["val"] + split_rows["test"])
        test_alerts = int(scored.loc[scored["split"].astype(str) == "test", "alert"].sum())
        c.check("7-deployment", "scored test alerts = reported alerts", test_alerts == r["alerts"],
                {"scored": test_alerts, "reported": r["alerts"]})
        # Online/batch parity: the feature store must reproduce the batch features
        parity = online_batch_parity(cfg, silver, feats["test"])
        c.check("7-deployment", "online feature store reproduces batch features (streamed replay)",
                parity["match_rate"] >= 0.995, parity)

    # ---------------------------------------------------------- 8. monitoring
    mon = P["monitoring"] / "summary.json"
    c.check("8-monitoring", "monitoring summary written", mon.exists())
    if mon.exists():
        s = _load(mon)
        expected = int(np.unique(np.concatenate([feats[k]["ts"].dt.strftime("%Y-%m").unique()
                                                 for k in ("val", "test")])).size)
        c.check("8-monitoring", "every scored month monitored", s["months"] == expected,
                {"monitored": s["months"], "scored_months": expected})
        c.check("8-monitoring", "retraining decision recorded", "retrain_recommended" in s)

    report = {
        "status": "PASS" if c.passed else "FAIL",
        "passed": sum(ch["status"] == "PASS" for ch in c.checks),
        "failed": sum(ch["status"] == "FAIL" for ch in c.checks),
        "champion": champion,
        "success_criteria_met": all(x["met"] for x in c.criteria),
        "success_criteria": c.criteria,
        "checks": c.checks,
    }
    write_json(report, P["reports"] / "verification_report.json")
    log.info("Verification %s: %s passed, %s failed", report["status"], report["passed"], report["failed"])
    return report


if __name__ == "__main__":
    rep = run_verification(load_config(), verify_checksums="--no-checksums" not in sys.argv)
    sys.exit(0 if rep["status"] == "PASS" else 1)
