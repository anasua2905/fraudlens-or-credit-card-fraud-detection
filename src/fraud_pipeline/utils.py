"""Shared helpers: config loading, logging, and table IO."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_logger(name: str = "fraud_pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def load_config(path: str | Path | None = None, root: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config and resolve every entry in `paths` to an absolute Path."""
    root = Path(root) if root else PROJECT_ROOT
    path = Path(path) if path else root / "config" / "pipeline.yaml"
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["paths"] = {k: (root / v).resolve() for k, v in cfg["paths"].items()}
    for p in cfg["paths"].values():
        p.mkdir(parents=True, exist_ok=True)
    return cfg


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def write_table(df: pd.DataFrame, path_without_suffix: Path, fmt: str = "parquet") -> Path:
    """Write a table as Parquet (preferred) or CSV fallback. Returns the written path."""
    if fmt == "parquet" and _parquet_available():
        out = path_without_suffix.with_suffix(".parquet")
        df.to_parquet(out, index=False)
    else:
        if fmt == "parquet":
            get_logger().warning("pyarrow not installed; writing CSV instead of Parquet")
        out = path_without_suffix.with_suffix(".csv")
        df.to_csv(out, index=False)
    return out


def read_table(path_without_suffix: Path) -> pd.DataFrame:
    pq, csv = path_without_suffix.with_suffix(".parquet"), path_without_suffix.with_suffix(".csv")
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv, parse_dates=["ts"])
    raise FileNotFoundError(f"No table found at {pq} or {csv}")


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
