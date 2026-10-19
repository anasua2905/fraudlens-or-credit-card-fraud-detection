"""Run the complete FraudLens pipeline, from raw files to verified dashboard data.

    python -m fraud_pipeline.run_all                       # everything (raw files must be in data/raw or on Kaggle)
    python -m fraud_pipeline.run_all --local-dir PATH      # copy raw CSVs from PATH first
    python -m fraud_pipeline.run_all --from train          # resume from a later stage

Stages: ingest -> quality -> eda -> preprocess -> train -> score -> verify
"""
from __future__ import annotations

import argparse
import sys
import time

import matplotlib.pyplot as plt

from .eda import run_eda
from .pipeline import run as run_pipeline
from .preprocess import run_preprocessing
from .quality import run_quality_audit
from .score import run_scoring
from .train import run_training
from .utils import get_logger, load_config, write_json
from .verify import run_verification

log = get_logger()
STAGES = ["ingest", "quality", "eda", "preprocess", "train", "score", "verify"]


def run_all(cfg: dict, start: str = "ingest", source: str = "kaggle", local_dir: str | None = None,
            checksums: bool = True) -> dict:
    plt.switch_backend("Agg")
    timings: dict = {}
    steps = {
        "ingest": lambda: run_pipeline(cfg, source=source, local_dir=local_dir),
        "quality": lambda: run_quality_audit(cfg),
        "eda": lambda: run_eda(cfg),
        "preprocess": lambda: run_preprocessing(cfg),
        "train": lambda: run_training(cfg),
        "score": lambda: run_scoring(cfg),
        "verify": lambda: run_verification(cfg, verify_checksums=checksums),
    }
    outputs = {}
    for stage in STAGES[STAGES.index(start):]:
        log.info("==== %s ====", stage.upper())
        t0 = time.perf_counter()
        outputs[stage] = steps[stage]()
        timings[stage] = round(time.perf_counter() - t0, 1)
    summary = {
        "stages_run": list(timings),
        "seconds": timings,
        "total_seconds": round(sum(timings.values()), 1),
        "champion": outputs.get("train", {}).get("champion"),
        "verification": outputs.get("verify", {}).get("status"),
    }
    write_json(summary, cfg["paths"]["reports"] / "run_all_summary.json")
    log.info("All stages finished in %.1fs | verification: %s", summary["total_seconds"], summary["verification"])
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full FraudLens pipeline")
    ap.add_argument("--from", dest="start", choices=STAGES, default="ingest")
    ap.add_argument("--local-dir", default=None, help="folder containing fraudTrain.csv and fraudTest.csv")
    ap.add_argument("--no-checksums", action="store_true", help="skip re-hashing the raw files")
    a = ap.parse_args()
    summary = run_all(load_config(), start=a.start, source="local" if a.local_dir else "kaggle",
                      local_dir=a.local_dir, checksums=not a.no_checksums)
    sys.exit(0 if summary["verification"] in ("PASS", None) else 1)


if __name__ == "__main__":
    main()
