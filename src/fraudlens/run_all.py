"""Run the whole FraudLens lifecycle, or any slice of it.

    python -m fraudlens.run_all                         # every stage
    python -m fraudlens.run_all --local-dir D:/data     # acquire by copying local CSVs
    python -m fraudlens.run_all --start train           # resume from a stage
    python -m fraudlens.run_all --only monitor bundle   # just these stages
    python -m fraudlens.run_all --quick                 # 200k training rows, no tuning (smoke test)

Stages (lifecycle step in brackets):
    acquire [2] -> prepare [3] -> quality [3] -> eda [4] -> preprocess [5] -> train [5]
    -> explain [6] -> score [7] -> monitor [8] -> verify -> bundle [7]
    -> publish [platform] -> crawl [platform] -> olap [platform]

The platform stages (publish to MinIO, register in the Hive Metastore, load
ClickHouse and benchmark it against Trino) skip themselves when the Docker
platform is not running.
"""
from __future__ import annotations

import argparse
import time

from .utils import get_logger, load_config, write_json

log = get_logger()

STAGES = ["acquire", "prepare", "quality", "eda", "preprocess", "train", "explain", "score", "monitor",
          "verify", "bundle", "publish", "crawl", "olap"]


def _stage(name: str, cfg: dict, opts: dict):
    if name == "acquire":
        from .acquire import acquire
        return acquire(cfg, opts.get("source", "auto"), opts.get("local_dir"))
    if name == "prepare":
        from .prepare import run_prepare
        return run_prepare(cfg)
    if name == "quality":
        from .quality import run_quality_audit
        return run_quality_audit(cfg)
    if name == "eda":
        import matplotlib
        matplotlib.use("Agg")
        from .eda import run_eda
        return run_eda(cfg)
    if name == "preprocess":
        from .preprocess import run_preprocessing
        return run_preprocessing(cfg)
    if name == "train":
        from .train import run_training
        return run_training(cfg)
    if name == "explain":
        from .explain import run_explain
        return run_explain(cfg)
    if name == "score":
        from .batch_score import run_scoring
        return run_scoring(cfg)
    if name == "monitor":
        from .monitor import run_monitoring
        return run_monitoring(cfg)
    if name == "verify":
        from .verify import run_verification
        rep = run_verification(cfg, verify_checksums=opts.get("checksums", True))
        if rep["status"] != "PASS" and opts.get("strict", True):
            raise SystemExit(f"Verification failed: {rep['failed']} check(s); see reports/verification_report.json")
        return rep
    if name == "bundle":
        from .bundle import run_bundle
        return run_bundle(cfg)
    if name == "publish":
        from .publish import run_publish
        return run_publish(cfg)
    if name == "crawl":
        from .crawler import run_crawler
        return run_crawler(cfg)
    if name == "olap":
        from .olap import run_olap
        return run_olap(cfg)
    raise ValueError(name)


def run_all(cfg: dict | None = None, start: str | None = None, stop: str | None = None,
            only: list[str] | None = None, skip: list[str] | None = None, **opts) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    cfg = cfg or load_config()
    stages = only or STAGES[STAGES.index(start) if start else 0: STAGES.index(stop) + 1 if stop else None]
    stages = [s for s in stages if s not in (skip or [])]
    timings = {}
    for s in stages:
        t0 = time.perf_counter()
        log.info("=" * 18 + f" stage: {s} " + "=" * 18)
        _stage(s, cfg, opts)
        timings[s] = round(time.perf_counter() - t0, 1)
    write_json({"stages": stages, "seconds": timings, "total_seconds": round(sum(timings.values()), 1)},
               cfg["paths"]["reports"] / "run_all.json")
    log.info("Done: %s", ", ".join(f"{k} {v:.0f}s" for k, v in timings.items()))
    return timings


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the FraudLens lifecycle")
    ap.add_argument("--start", choices=STAGES)
    ap.add_argument("--stop", choices=STAGES)
    ap.add_argument("--only", nargs="+", choices=STAGES)
    ap.add_argument("--skip", nargs="+", choices=STAGES, default=[])
    ap.add_argument("--source", choices=["auto", "kaggle", "local"], default="auto")
    ap.add_argument("--ingest", choices=["direct", "nifi"], help="how raw files reach the lake (default: config)")
    ap.add_argument("--local-dir")
    ap.add_argument("--config")
    ap.add_argument("--quick", action="store_true", help="200k training rows, no tuning")
    ap.add_argument("--no-checksums", action="store_true")
    ap.add_argument("--no-strict", action="store_true", help="do not stop when verification fails")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.ingest:
        cfg["platform"]["ingest"] = a.ingest
    if a.quick:
        cfg["model"]["max_train_rows"] = 200_000
        cfg["model"]["tuning"]["enabled"] = False
        cfg["model"]["selection_bootstrap"] = 100
        cfg["evaluation"]["bootstrap"] = 50
    run_all(cfg, a.start, a.stop, a.only, a.skip, source=a.source, local_dir=a.local_dir,
            checksums=not a.no_checksums, strict=not a.no_strict)


if __name__ == "__main__":
    main()
