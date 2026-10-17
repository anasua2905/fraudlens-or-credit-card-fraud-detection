"""Lifecycle stage 2 - Data acquisition.

Gets the Sparkov files from the Kaggle API (open dataset) or a local folder,
lands them untouched in the bronze layer, and collects metadata: checksums,
sizes, row counts, a column profile and the dataset's provenance and licence.
If the data platform is running, the raw files are also landed in the
MinIO data lake (bucket `fraudlens-raw`), either directly (boto3) or through
the Apache NiFi ingestion flow (`platform.ingest: nifi`): the files are dropped
in the NiFi landing folder and NiFi writes them to the lake together with an
ingest record per file; this stage waits until every file has arrived intact.

    python -m fraudlens.acquire                       # Kaggle API (needs ~/.kaggle/kaggle.json)
    python -m fraudlens.acquire --local-dir D:/data    # copy files you already downloaded
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

import pandas as pd

from .utils import get_logger, load_config, rel, write_json

log = get_logger()


def sha256sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def count_rows(path: Path) -> int:
    """Data rows (excluding the header). Sparkov has no embedded newlines in fields."""
    with open(path, "rb") as fh:
        return sum(1 for _ in fh) - 1


def download_from_kaggle(slug: str, dest: Path) -> None:
    """Download and unzip a Kaggle dataset with the official `kaggle` API client.

    Credentials: ~/.kaggle/kaggle.json, or KAGGLE_USERNAME and KAGGLE_KEY.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("Install the Kaggle client (pip install kaggle) or use --local-dir") from exc
    api = KaggleApi()
    api.authenticate()
    log.info("Downloading %s from the Kaggle API into %s", slug, dest)
    api.dataset_download_files(slug, path=str(dest), unzip=False, quiet=False)
    for z in dest.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        z.unlink()


def copy_local(src_dir: Path, dest: Path, filenames: list[str]) -> None:
    for name in filenames:
        src = Path(src_dir) / name
        if not src.exists():
            raise FileNotFoundError(f"Expected {src}")
        if src.resolve() != (dest / name).resolve():
            shutil.copy2(src, dest / name)
            log.info("Copied %s -> %s", src, dest / name)


def profile_columns(path: Path, sample_rows: int = 100_000) -> list[dict]:
    """Column-level metadata from a sample: inferred type, nulls, distinct values, example."""
    df = pd.read_csv(path, nrows=sample_rows)
    out = []
    for c in df.columns:
        s = df[c]
        out.append({
            "column": c,
            "inferred_dtype": str(s.dtype),
            "null_pct_in_sample": round(100 * float(s.isna().mean()), 3),
            "distinct_in_sample": int(s.nunique()),
            "example": None if s.dropna().empty else str(s.dropna().iloc[0])[:40],
        })
    return out


def land_via_nifi(cfg: dict, raw_dir: Path, names: dict[str, str]) -> dict:
    """Drop files in NiFi's landing folder and wait until NiFi has written them to the raw layer.

    names maps lake key -> local file name. Returns the NiFi ingest records per file.
    """
    from . import lake
    plat = cfg["platform"]
    landing = Path(cfg["root"]) / plat.get("nifi_landing", "data/landing")
    landing.mkdir(parents=True, exist_ok=True)
    bucket = plat["buckets"]["raw"]
    started = time.time()
    for key, name in names.items():
        tmp = landing / f".{name}.part"           # NiFi ignores dot-files, so it never sees half a copy
        shutil.copyfile(raw_dir / name, tmp)       # copyfile: a fresh mtime makes ListFile pick it up again
        tmp.replace(landing / name)
    log.info("Dropped %s file(s) in the NiFi landing folder %s", len(names), landing)
    s3 = lake.client(cfg)
    timeout = float(plat.get("nifi_timeout_s", 900))
    pending = dict(names)
    while pending:
        for key, name in list(pending.items()):
            try:
                head = s3.head_object(Bucket=bucket, Key=key)
            except Exception:
                continue
            fresh = head["LastModified"].timestamp() >= started - 5
            if fresh and head["ContentLength"] == (raw_dir / name).stat().st_size:
                log.info("NiFi landed s3://%s/%s (%s bytes)", bucket, key, f"{head['ContentLength']:,}")
                pending.pop(key)
        if pending and time.time() - started > timeout:
            raise TimeoutError(f"NiFi did not land {sorted(pending)} within {timeout:.0f}s; "
                               "check the flow at https://localhost:8443/nifi")
        if pending:
            time.sleep(5)
    records = {}
    for key, name in names.items():                # the ingest record is written right after the file
        for _ in range(24):
            try:
                records[name] = json.loads(lake.read_bytes(cfg, bucket, f"sparkov/_ingest_log/{name}.json"))
                break
            except Exception:
                time.sleep(5)
    for name in names.values():
        (landing / name).unlink(missing_ok=True)
    return records


def build_manifest(cfg: dict, source: str) -> dict:
    raw_dir: Path = cfg["paths"]["raw"]
    entries = {}
    for split, name in cfg["dataset"]["files"].items():
        path = raw_dir / name
        entries[split] = {
            "file": name,
            "bytes": path.stat().st_size,
            "rows": count_rows(path),
            "sha256": sha256sum(path),
        }
        log.info("Manifest %-5s %-15s rows=%s", split, name, f"{entries[split]['rows']:,}")
    return {
        "dataset": cfg["dataset"]["kaggle_slug"],
        "source_url": cfg["dataset"]["source_url"],
        "licence": cfg["dataset"]["licence"],
        "acquired_via": source,
        "acquired_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "files": entries,
    }


def acquire(cfg: dict, source: str = "auto", local_dir: str | Path | None = None,
            upload: bool | None = None) -> dict:
    """Populate the bronze layer; return the manifest.

    source: "kaggle" (API), "local" (copy from local_dir) or "auto"
            (files already present > local_dir if given > Kaggle API).
    upload: True/False to force; None uploads only if the MinIO lake is reachable.
    """
    raw_dir: Path = cfg["paths"]["raw"]
    files: list[str] = list(cfg["dataset"]["files"].values())
    present = all((raw_dir / f).exists() for f in files)

    if source == "auto":
        source = "existing" if present else ("local" if local_dir else "kaggle")
    if source == "kaggle":
        download_from_kaggle(cfg["dataset"]["kaggle_slug"], raw_dir)
    elif source == "local":
        if local_dir is None:
            raise ValueError("local_dir is required when source='local'")
        copy_local(Path(local_dir), raw_dir, files)
    elif source != "existing":
        raise ValueError(f"Unknown source {source!r}")

    manifest = build_manifest(cfg, source)
    write_json(manifest, raw_dir / "manifest.json")
    metadata = {
        "dataset": manifest["dataset"],
        "description": "Simulated credit card transactions (Sparkov generator), 1 Jan 2019 - 31 Dec 2020, "
                       "1,000 customers and 800 merchants, labelled is_fraud.",
        "licence": manifest["licence"],
        "source_url": manifest["source_url"],
        "columns": profile_columns(raw_dir / files[0]),
    }
    write_json(metadata, cfg["paths"]["reports"] / "acquisition_metadata.json")

    from . import lake
    if upload is None:
        upload = lake.available(cfg)
    if upload:
        bucket = cfg["platform"]["buckets"]["raw"]
        lake.ensure_buckets(cfg)
        # one folder per file, so the crawler can register each as an external CSV table
        keys = {f"sparkov/{split_name}/{f}": f for split_name, f in cfg["dataset"]["files"].items()}
        keys["sparkov/_manifest/manifest.json"] = "manifest.json"
        route = cfg["platform"].get("ingest", "direct")
        manifest["lake"] = {"bucket": bucket, "prefix": "sparkov/", "ingested_by": route}
        if route == "nifi":
            manifest["lake"]["nifi_ingest_records"] = land_via_nifi(cfg, raw_dir, keys)
        else:
            for key, f in keys.items():
                lake.upload_file(cfg, raw_dir / f, bucket, key)
        log.info("Raw files are in s3://%s/sparkov/ (via %s)", bucket, route)
    write_json({**manifest, "raw_dir": rel(raw_dir, cfg)}, cfg["paths"]["reports"] / "acquisition_manifest.json")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2: acquire the Sparkov dataset")
    ap.add_argument("--source", choices=["auto", "kaggle", "local"], default="auto")
    ap.add_argument("--local-dir", default=None)
    ap.add_argument("--upload", choices=["auto", "yes", "no"], default="auto")
    a = ap.parse_args()
    up = {"auto": None, "yes": True, "no": False}[a.upload]
    acquire(load_config(), a.source, a.local_dir, up)


if __name__ == "__main__":
    main()
