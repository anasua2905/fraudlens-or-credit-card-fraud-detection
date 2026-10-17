"""MinIO data lake helpers (S3-compatible object storage, the open-source stand-in for Amazon S3).

The pipeline computes locally and publishes every layer to the lake so that
Trino, Superset and anyone else on the platform read one governed copy:

    s3://fraudlens-raw/sparkov/...           bronze: the CSVs exactly as acquired
    s3://fraudlens-silver/transactions/...   silver: cleaned, typed, PII removed
    s3://fraudlens-gold/<table>/...          gold: features, scores, monitoring

All functions are no-ops that return False/[] when the lake is not reachable,
so the pipeline still runs on a laptop without Docker.
"""
from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .utils import get_logger

log = get_logger()


def _settings(cfg: dict) -> tuple[str, str, str]:
    p = cfg.get("platform", {})
    return p.get("s3_endpoint"), p.get("s3_access_key"), p.get("s3_secret_key")


@lru_cache(maxsize=4)
def _client_cached(endpoint: str, key: str, secret: str):
    import boto3
    from botocore.config import Config
    return boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=key, aws_secret_access_key=secret,
                        region_name="us-east-1",
                        config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                                      connect_timeout=3, read_timeout=60, retries={"max_attempts": 2}))


def client(cfg: dict):
    endpoint, key, secret = _settings(cfg)
    return _client_cached(endpoint, key, secret)


def available(cfg: dict) -> bool:
    """True when boto3 is installed and the MinIO endpoint answers with valid credentials."""
    endpoint, key, secret = _settings(cfg)
    if not endpoint or not key:
        return False
    try:
        client(cfg).list_buckets()
        return True
    except ImportError:
        log.info("boto3 not installed; skipping the data lake")
    except Exception as exc:  # connection refused, bad credentials, ...
        log.info("Data lake not reachable at %s (%s); continuing locally", endpoint, type(exc).__name__)
    return False


def bucket(cfg: dict, layer: str) -> str:
    return cfg["platform"]["buckets"][layer]


def ensure_buckets(cfg: dict) -> None:
    s3 = client(cfg)
    existing = {b["Name"] for b in s3.list_buckets().get("Buckets", [])}
    for name in cfg["platform"]["buckets"].values():
        if name not in existing:
            s3.create_bucket(Bucket=name)
            log.info("Created bucket %s", name)


def upload_file(cfg: dict, path: Path, bucket_name: str, key: str) -> str:
    client(cfg).upload_file(str(path), bucket_name, key)
    uri = f"s3://{bucket_name}/{key}"
    log.info("Uploaded %s -> %s", Path(path).name, uri)
    return uri


def delete_prefix(cfg: dict, bucket_name: str, prefix: str) -> int:
    """Remove every object under a prefix (used before re-publishing a table)."""
    s3 = client(cfg)
    keys = list_keys(cfg, bucket_name, prefix)
    for i in range(0, len(keys), 1000):
        s3.delete_objects(Bucket=bucket_name, Delete={"Objects": [{"Key": k} for k in keys[i:i + 1000]]})
    return len(keys)


def upload_dataframe(cfg: dict, df: pd.DataFrame, bucket_name: str, key: str) -> str:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    client(cfg).upload_fileobj(buf, bucket_name, key)
    return f"s3://{bucket_name}/{key}"


def list_keys(cfg: dict, bucket_name: str, prefix: str = "") -> list[str]:
    s3 = client(cfg)
    keys, token = [], None
    while True:
        kw = {"Bucket": bucket_name, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        keys += [o["Key"] for o in resp.get("Contents", [])]
        if not resp.get("IsTruncated"):
            return keys
        token = resp["NextContinuationToken"]


def read_bytes(cfg: dict, bucket_name: str, key: str, nbytes: int | None = None) -> bytes:
    kw = {"Bucket": bucket_name, "Key": key}
    if nbytes:
        kw["Range"] = f"bytes=0-{nbytes - 1}"
    return client(cfg).get_object(**kw)["Body"].read()


def read_parquet(cfg: dict, bucket_name: str, key: str) -> pd.DataFrame:
    obj = client(cfg).get_object(Bucket=bucket_name, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))
