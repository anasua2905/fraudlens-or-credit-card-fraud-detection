"""Build the compact deploy bundle for the Streamlit dashboard (Streamlit Community Cloud).

The full pipeline output is several GB; the hosted dashboard needs only:

    dashboard/bundle/
        model.joblib                  champion FraudModel (preprocessor + model + threshold)
        card_state_*.parquet          online feature-store snapshot (live scoring)
        scored.parquet                validation + test transactions with scores and reasons
        reports/*.json, monitoring    model results, explanations, monitoring, verification
        figures/*.png                 static figures for the report tab
        manifest.json                 file list, sizes, checksums, library versions

and writes dashboard/requirements.txt pinned to the library versions that
trained the model (a pickled scikit-learn model must be loaded by the same
version). The bundle is small enough (< 50 MB) to commit to GitHub, which is
where Streamlit Community Cloud reads it from.

    python -m fraudlens.bundle
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata as md
import shutil
from pathlib import Path

import pandas as pd

from .utils import get_logger, load_config, read_json, read_table, rel, write_json

log = get_logger()

SCORED_COLUMNS = ["transaction_id", "ts", "card_id", "merchant", "category", "amount", "gender", "age", "city",
                  "state", "lat", "lon", "hour", "day_of_week", "is_fraud", "split", "score", "alert", "outcome",
                  "amt_to_card_mean", "amt_zscore_card", "txn_count_1h", "txn_count_24h", "amt_sum_24h",
                  "secs_since_last_txn", "is_night", "category_share_card", "reason_1", "reason_2", "reason_3"]
REPORTS = ["model_results.json", "dashboard_meta.json", "explanations.json", "verification_report.json",
           "pipeline_run.json", "validation_report.json", "cleaning_audit.json", "preprocessing_report.json",
           "acquisition_manifest.json", "acquisition_metadata.json", "eda_summary.json", "data_catalog.json",
           "lake_publication.json", "explanation_quality.json", "stability.json", "fairness.json"]
MONITORING = ["summary.json", "monthly.csv", "feature_drift.csv"]
FIGURE_PREFIXES = ("0", "1")
MAX_BUNDLE_MB = 95


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def requirements(champion_library: str) -> list[str]:
    pins = {"streamlit": None, "pandas": None, "numpy": None, "scikit-learn": None, "joblib": None,
            "pyarrow": None, "plotly": None, "pyyaml": None, "shap": None}
    if champion_library in ("xgboost", "lightgbm"):
        pins[champion_library] = None
    lines = []
    for pkg in pins:
        v = _version(pkg) or (_version("xgboost-cpu") if pkg == "xgboost" else None)
        # exact pins for what unpickles the model; compatible-release pins for the UI stack
        exact = pkg in ("scikit-learn", "numpy", "joblib", "xgboost", "lightgbm")
        if v is None:
            lines.append(pkg)
        elif exact:
            lines.append(f"{pkg}=={v}")
        else:
            major_minor = ".".join(v.split(".")[:2])
            lines.append(f"{pkg}>={major_minor}")
    return lines


def _compact(df: pd.DataFrame) -> pd.DataFrame:
    df = df[[c for c in SCORED_COLUMNS if c in df.columns]].copy()
    for c in ("merchant", "category", "gender", "city", "state", "split", "outcome", "card_id",
              "reason_1", "reason_2", "reason_3"):
        if c in df:
            df[c] = df[c].astype("category")
    for c in df.select_dtypes("float64").columns:
        if c not in ("lat", "lon"):
            df[c] = df[c].astype("float32")
    return df


def run_bundle(cfg: dict) -> dict:
    P = cfg["paths"]
    out: Path = P["bundle"]
    if out.exists():
        shutil.rmtree(out)
    (out / "reports" / "monitoring").mkdir(parents=True)
    (out / "figures").mkdir()

    shutil.copy2(P["artifacts"] / "fraudlens_model.joblib", out / "model.joblib")
    if (P["artifacts"] / "shap_background.parquet").exists():
        shutil.copy2(P["artifacts"] / "shap_background.parquet", out / "shap_background.parquet")
    for name in ("cards", "categories", "window"):
        read_table(P["artifacts"] / f"card_state_{name}").to_parquet(out / f"card_state_{name}.parquet",
                                                                    index=False, compression="zstd")
    scored = _compact(read_table(P["gold"] / "scored_transactions"))
    scored.to_parquet(out / "scored.parquet", index=False, compression="zstd")
    for f in REPORTS:
        if (P["reports"] / f).exists():
            shutil.copy2(P["reports"] / f, out / "reports" / f)
    for f in MONITORING:
        if (P["monitoring"] / f).exists():
            shutil.copy2(P["monitoring"] / f, out / "reports" / "monitoring" / f)
    for f in sorted(P["figures"].glob("*.png")):
        if f.name.startswith(FIGURE_PREFIXES):
            shutil.copy2(f, out / "figures" / f.name)

    results = read_json(P["reports"] / "model_results.json")
    lib = results["results"][results["champion"]]["library"]
    reqs = requirements(lib)
    (cfg["root"] / "dashboard" / "requirements.txt").write_text(
        "# Generated by `python -m fraudlens.bundle`: pinned to the versions that trained the model.\n"
        + "\n".join(reqs) + "\n", encoding="utf-8")

    files = [{"file": p.relative_to(out).as_posix(), "bytes": p.stat().st_size, "sha256": _sha(p)}
             for p in sorted(out.rglob("*")) if p.is_file()]
    total_mb = sum(f["bytes"] for f in files) / 1e6
    manifest = {"built_at": dt.datetime.now().isoformat(timespec="seconds"), "champion": results["champion"],
                "rows_scored": int(len(scored)), "total_mb": round(total_mb, 1), "requirements": reqs,
                "files": files}
    write_json(manifest, out / "manifest.json")
    if total_mb > MAX_BUNDLE_MB:
        log.warning("Bundle is %.0f MB; GitHub rejects files over 100 MB", total_mb)
    log.info("Deploy bundle: %s files, %.1f MB -> %s", len(files), total_mb, rel(out, cfg))
    return manifest


if __name__ == "__main__":
    run_bundle(load_config())
