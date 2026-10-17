"""Bronze layer: acquire the raw Sparkov CSVs and record a manifest.

The manifest (checksum, size, row count, timestamp) makes every downstream
result traceable to the exact source files it was built from.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import shutil
import subprocess
from pathlib import Path

from .utils import get_logger, write_json

log = get_logger()


def sha256sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def count_rows(path: Path) -> int:
    """Data rows (excluding header). Sparkov has no embedded newlines in fields."""
    with open(path, "rb") as fh:
        return sum(1 for _ in fh) - 1


def download_from_kaggle(slug: str, dest: Path) -> None:
    """Download and unzip a Kaggle dataset.

    Requires the `kaggle` package and credentials in ~/.kaggle/kaggle.json
    (or KAGGLE_USERNAME / KAGGLE_KEY environment variables).
    """
    log.info("Downloading %s from Kaggle into %s", slug, dest)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"],
        check=True,
    )


def copy_local(src_dir: Path, dest: Path, filenames: list[str]) -> None:
    for name in filenames:
        src = Path(src_dir) / name
        if not src.exists():
            raise FileNotFoundError(f"Expected {src}")
        if src.resolve() != (dest / name).resolve():
            shutil.copy2(src, dest / name)
            log.info("Copied %s -> %s", src, dest / name)


def build_manifest(raw_dir: Path, files: dict[str, str], source: str) -> dict:
    entries = {}
    for split, name in files.items():
        path = raw_dir / name
        entries[split] = {
            "file": name,
            "bytes": path.stat().st_size,
            "rows": count_rows(path),
            "sha256": sha256sum(path),
        }
        log.info("Manifest %-5s %-15s rows=%s", split, name, f"{entries[split]['rows']:,}")
    return {
        "source": source,
        "ingested_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "files": entries,
    }


def ingest(cfg: dict, source: str = "kaggle", local_dir: str | Path | None = None) -> dict:
    """Populate the bronze layer and return the manifest.

    source = "kaggle"  -> download via Kaggle API (skipped if files already present)
    source = "local"   -> copy CSVs from `local_dir`
    """
    raw_dir: Path = cfg["paths"]["raw"]
    files: dict[str, str] = cfg["dataset"]["files"]
    present = all((raw_dir / f).exists() for f in files.values())

    if source == "kaggle":
        if present:
            log.info("Raw files already present in %s; skipping download", raw_dir)
        else:
            download_from_kaggle(cfg["dataset"]["kaggle_slug"], raw_dir)
    elif source == "local":
        if local_dir is None:
            raise ValueError("local_dir is required when source='local'")
        copy_local(Path(local_dir), raw_dir, list(files.values()))
    else:
        raise ValueError(f"Unknown source {source!r}")

    manifest = build_manifest(raw_dir, files, source)
    write_json(manifest, raw_dir / "manifest.json")
    return manifest
