"""Shared helpers: configuration, logging, table and JSON IO."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PATH_DEFAULTS = {
    "raw": "data/raw",
    "silver": "data/silver",
    "gold": "data/gold",
    "processed": "data/processed",
    "artifacts": "artifacts",
    "reports": "reports",
    "figures": "reports/figures",
    "interactive": "reports/interactive",
    "monitoring": "reports/monitoring",
    "bundle": "dashboard/bundle",
}


def get_logger(name: str = "fraudlens") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def _load_platform_env(root: Path) -> None:
    """Pick up the host-side settings (FRAUDLENS_*, MLFLOW_*) from platform/.env when it exists.

    Variables already set in the environment win, so CI and shells can override them.
    """
    env = root / "platform" / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = (x.strip() for x in line.split("=", 1))
            if k.startswith(("FRAUDLENS_", "MLFLOW_")):
                os.environ.setdefault(k, v)


def load_config(path: str | Path | None = None, root: str | Path | None = None) -> dict[str, Any]:
    """Load config/pipeline.yaml and resolve every entry in `paths` to an absolute Path.

    `root` is the project root (defaults to this repository). Platform settings can be
    overridden with environment variables, so secrets never live in the YAML file.
    """
    root = Path(root).resolve() if root else PROJECT_ROOT
    path = Path(path) if path else root / "config" / "pipeline.yaml"
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    paths = {**PATH_DEFAULTS, **cfg.get("paths", {})}
    cfg["root"] = root
    cfg["paths"] = {k: (root / v).resolve() for k, v in paths.items()}
    for p in cfg["paths"].values():
        p.mkdir(parents=True, exist_ok=True)
    plat = cfg.setdefault("platform", {})
    _load_platform_env(root)
    env_map = {
        "s3_endpoint": "FRAUDLENS_S3_ENDPOINT",
        "s3_access_key": "FRAUDLENS_S3_ACCESS_KEY",
        "s3_secret_key": "FRAUDLENS_S3_SECRET_KEY",
        "trino_host": "FRAUDLENS_TRINO_HOST",
        "trino_port": "FRAUDLENS_TRINO_PORT",
        "trino_user": "FRAUDLENS_TRINO_USER",
        "mlflow_tracking_uri": "MLFLOW_TRACKING_URI",
        "ingest": "FRAUDLENS_INGEST",
        "clickhouse_url": "FRAUDLENS_CLICKHOUSE_URL",
        "clickhouse_user": "FRAUDLENS_CLICKHOUSE_USER",
        "clickhouse_password": "FRAUDLENS_CLICKHOUSE_PASSWORD",
    }
    for key, env in env_map.items():
        if os.environ.get(env):
            plat[key] = os.environ[env]
    return cfg


def rel(path: Path | str, cfg: dict) -> str:
    """Project-relative POSIX path for reports (absolute paths leak the local machine)."""
    try:
        return Path(path).resolve().relative_to(cfg["root"]).as_posix()
    except ValueError:
        return Path(path).as_posix()


def parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def write_table(df: pd.DataFrame, path_without_suffix: Path, fmt: str = "parquet") -> Path:
    """Write Parquet (preferred) or CSV if pyarrow is missing. Returns the written path."""
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet" and parquet_available():
        out = path_without_suffix.with_suffix(".parquet")
        df.to_parquet(out, index=False)
    else:
        if fmt == "parquet":
            get_logger().warning("pyarrow not installed; writing CSV instead of Parquet")
        out = path_without_suffix.with_suffix(".csv")
        df.to_csv(out, index=False)
    return out


def read_table(path_without_suffix: Path, columns: list[str] | None = None) -> pd.DataFrame:
    pq, csv = path_without_suffix.with_suffix(".parquet"), path_without_suffix.with_suffix(".csv")
    if pq.exists():
        return pd.read_parquet(pq, columns=columns)
    if csv.exists():
        df = pd.read_csv(csv, usecols=columns)
        for c in ("ts", "dob"):
            if c in df:
                df[c] = pd.to_datetime(df[c])
        return df
    raise FileNotFoundError(f"No table found at {pq} or {csv}")


def table_exists(path_without_suffix: Path) -> bool:
    return path_without_suffix.with_suffix(".parquet").exists() or path_without_suffix.with_suffix(".csv").exists()


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, Path):
        return o.as_posix()
    return str(o)


def _clean_floats(o: Any) -> Any:
    """NaN/inf are not valid JSON; replace them with null."""
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, dict):
        return {str(k): _clean_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_floats(v) for v in o]
    return o


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_clean_floats(obj), fh, indent=2, default=_json_default)


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
