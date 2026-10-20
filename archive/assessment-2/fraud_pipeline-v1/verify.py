"""End-to-end verification: does every stage agree with every other?

Each check compares two independent sources that must match, for example the
row count in the manifest against the rows in the processed matrices, or the
metrics stored in model_results.json against a fresh re-scoring of the saved
model. A FAIL means the reported results do not describe the system on disk.

Run:  python -m fraud_pipeline.verify      (exit code 1 if any check fails)
"""
from __future__ import annotations

import json
import sys

import joblib

from . import evaluate as E
from . import features as F
from . import models as M
from .datasets import load_features, load_silver
from .ingest import sha256sum
from .train import score_split
from .utils import get_logger, load_config, read_table, write_json

log = get_logger()
SPLITS = ["train", "val", "test"]


class Checker:
    def __init__(self):
        self.checks: list[dict] = []

    def check(self, stage: str, name: str, ok: bool, detail=None):
        self.checks.append({"stage": stage, "check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
        (log.info if ok else log.error)("%-4s | %-14s | %s", "PASS" if ok else "FAIL", stage, name)

    @property
    def passed(self) -> bool:
        return all(c["status"] == "PASS" for c in self.checks)


def _load(path):
    return json.loads(path.read_text())


def run_verification(cfg: dict, verify_checksums: bool = True) -> dict:
    P = cfg["paths"]
    c = Checker()

    # ---------------------------------------------------------- 1. ingestion
    manifest = _load(P["raw"] / "manifest.json")
    if verify_checksums:
        for split, f in manifest["files"].items():
            c.check("ingestion", f"{f['file']} checksum matches manifest",
                    sha256sum(P["raw"] / f["file"]) == f["sha256"])
    raw_rows = sum(f["rows"] for f in manifest["files"].values())

    # ------------------------------------------------------- 2. silver/gold
    silver = load_silver(cfg)
    c.check("cleaning", "silver rows = manifest rows (nothing lost)", len(silver) == raw_rows,
            {"manifest": raw_rows, "silver": len(silver)})
    c.check("cleaning", "no identifying columns in silver",
            not {"cc_num", "first", "last", "street"} & set(silver.columns))

    feats = {s: load_features(cfg, s) for s in SPLITS}
    split_rows = {s: len(df) for s, df in feats.items()}
    c.check("splitting", "split rows sum to silver rows", sum(split_rows.values()) == len(silver), split_rows)
    ids = [set(df["transaction_id"]) for df in feats.values()]
    c.check("splitting", "no transaction appears in two splits",
            not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]))
    c.check("splitting", "splits are chronological (train < val < test)",
            feats["train"]["ts"].max() <= feats["val"]["ts"].min()
            and feats["val"]["ts"].max() <= feats["test"]["ts"].min())
    c.check("features", "no feature is derived from the label",
            not any("fraud" in f for f in F.numeric_features(cfg["features"]["velocity_windows"])))

    # ------------------------------------------------------ 3. preprocessing
    exclude = set(cfg["preprocess"].get("exclude_features", []))
    view_cols = {}
    for view in ["linear", "tree"]:
        pre = joblib.load(P["artifacts"] / f"preprocessor_{view}.joblib")
        names = list(pre.get_feature_names_out())
        view_cols[view] = names
        for s in SPLITS:
            X = read_table(P["processed"] / f"{view}_{s}")
            c.check("preprocessing", f"{view}/{s}: matrix rows = feature-table rows", len(X) == split_rows[s],
                    {"matrix": len(X), "features": split_rows[s]})
            c.check("preprocessing", f"{view}/{s}: columns = fitted preprocessor output",
                    [col for col in X.columns if col not in ("transaction_id", "is_fraud")] == names)
            if s != "train":
                continue
            c.check("preprocessing", f"{view}: same row order as feature table",
                    bool((X["transaction_id"].to_numpy() == feats[s]["transaction_id"].to_numpy()).all()))
        c.check("preprocessing", f"{view}: excluded features absent", not (exclude & set(names)),
                sorted(exclude & set(names)))
    lin_train = read_table(P["processed"] / "linear_train")
    c.check("preprocessing", "linear view has no missing values",
            int(lin_train.drop(columns=["transaction_id"]).isna().sum().sum()) == 0)

    # ------------------------------------------------------------ 4. models
    results = _load(P["reports"] / "model_results.json")
    champion = results["champion"]
    trained = [r["model"] for r in results["comparison"]]
    for name in trained:
        path = P["artifacts"] / f"model_{name}.joblib"
        c.check("models", f"{name}: saved model exists", path.exists())
        if not path.exists():
            continue
        model = joblib.load(path)
        expected = view_cols[M.VIEW_BY_MODEL[name]] + (["anomaly_score"] if name == "hybrid_gb" else [])
        got = list(getattr(model, "feature_names_in_", expected))
        c.check("models", f"{name}: input features = preprocessed columns", got == expected)
        if name in {"logistic", "hist_gb", "hybrid_gb"}:
            c.check("models", f"{name}: imbalance handled by class weights (no resampling)",
                    getattr(model, "class_weight", None) == "balanced",
                    {"class_weight": getattr(model, "class_weight", None)})

    # ----------------------------------------------------------- 5. selection
    sel = results.get("selection")
    c.check("selection", "selection rule recorded in results", sel is not None)
    if sel:
        c.check("selection", "reported champion = champion chosen by the rule", sel["champion"] == champion,
                {"reported": champion, "rule": sel["champion"]})
        order = sel["complexity_order"]
        equivalent = [m for m, v in sel["comparisons"].items() if v["equivalent_to_best"]]
        simplest = sorted(equivalent, key=lambda m: order.index(m) if m in order else len(order))[0]
        c.check("selection", "champion is the simplest model equivalent to the best", champion == simplest,
                {"equivalent": equivalent, "champion": champion})

    # ------------------------------------------ 6. results reproduce from disk
    tcfg = cfg["model"]["threshold"]
    strategy = results["threshold_strategy"]
    reported = results["results"][champion]
    val_scored = score_split(cfg, champion, "val").merge(
        feats["val"][["transaction_id", "is_fraud", "amount", "ts"]], on="transaction_id")
    test_scored = score_split(cfg, champion, "test").merge(
        feats["test"][["transaction_id", "is_fraud", "amount", "ts"]], on="transaction_id")

    def days(df):
        return max((df["ts"].max() - df["ts"].min()).total_seconds() / 86400, 1)

    thr = E.choose_threshold(val_scored["is_fraud"].to_numpy(), val_scored["score"].to_numpy(),
                             val_scored["amount"].to_numpy(), days(val_scored), tcfg, strategy)
    c.check("reproduction", "threshold re-derived on validation = reported threshold",
            abs(thr - reported["thresholds_from_validation"][strategy]) < 1e-5,
            {"recomputed": round(thr, 6), "reported": reported["thresholds_from_validation"][strategy]})
    m = E.evaluate(test_scored["is_fraud"].to_numpy(), test_scored["score"].to_numpy(),
                   test_scored["amount"].to_numpy(), reported["thresholds_from_validation"][strategy],
                   days(test_scored), tcfg)
    r = reported["test"]
    c.check("reproduction", "re-scored test PR-AUC = reported", abs(m["pr_auc"] - r["pr_auc"]) < 1e-4,
            {"recomputed": m["pr_auc"], "reported": r["pr_auc"]})
    c.check("reproduction", "re-scored test confusion matrix = reported",
            (m["true_positives"], m["false_positives"], m["false_negatives"])
            == (r["true_positives"], r["false_positives"], r["false_negatives"]),
            {"recomputed": [m["true_positives"], m["false_positives"], m["false_negatives"]],
             "reported": [r["true_positives"], r["false_positives"], r["false_negatives"]]})
    c.check("reproduction", "re-scored test net benefit = reported", abs(m["net_benefit"] - r["net_benefit"]) < 0.01,
            {"recomputed": m["net_benefit"], "reported": r["net_benefit"]})

    # ------------------------------------------------------------ 7. dashboard
    meta_path = P["reports"] / "dashboard_meta.json"
    c.check("dashboard", "dashboard data has been generated", meta_path.exists())
    if meta_path.exists():
        meta = _load(meta_path)
        scored = read_table(P["gold"] / "scored_transactions")
        c.check("dashboard", "dashboard uses the selected champion", meta["champion"] == champion,
                {"dashboard": meta["champion"], "champion": champion})
        c.check("dashboard", "dashboard threshold = reported threshold",
                abs(meta["threshold"] - reported["thresholds_from_validation"][strategy]) < 1e-9)
        c.check("dashboard", "dashboard rows = validation + test rows",
                len(scored) == split_rows["val"] + split_rows["test"])
        test_alerts = int(scored.loc[scored["split"].astype(str) == "test", "alert"].sum())
        c.check("dashboard", "dashboard test alerts = reported alerts", test_alerts == r["alerts"],
                {"dashboard": test_alerts, "reported": r["alerts"]})

    report = {
        "status": "PASS" if c.passed else "FAIL",
        "passed": sum(ch["status"] == "PASS" for ch in c.checks),
        "failed": sum(ch["status"] == "FAIL" for ch in c.checks),
        "champion": champion,
        "checks": c.checks,
    }
    write_json(report, P["reports"] / "verification_report.json")
    log.info("Verification %s: %s passed, %s failed", report["status"], report["passed"], report["failed"])
    return report


if __name__ == "__main__":
    rep = run_verification(load_config(), verify_checksums="--no-checksums" not in sys.argv)
    sys.exit(0 if rep["status"] == "PASS" else 1)
