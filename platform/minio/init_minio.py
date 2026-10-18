"""Create the FraudLens buckets, policies and service accounts in MinIO (idempotent).

Governance in the lake (the Lake Formation role):
  * three buckets, one per layer: fraudlens-raw (bronze), fraudlens-silver, fraudlens-gold
  * versioning on the raw bucket, so acquired source files can never be silently replaced
  * least-privilege service accounts:
        fraudlens-pipeline  read/write all three buckets      (runs the pipeline)
        fraudlens-trino     read-only all three buckets        (query engine)
        fraudlens-analyst   read-only gold, explicitly denied raw and silver
                            (also used by ClickHouse, which only serves gold analytics)
        fraudlens-ingest    write-only into s3://fraudlens-raw/sparkov/   (Apache NiFi)

Runs inside docker compose (service minio-init) or from the host:
    python platform/minio/init_minio.py          (reads platform/.env)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUCKETS = ["fraudlens-raw", "fraudlens-silver", "fraudlens-gold"]
ACCOUNTS = {  # policy name -> (access key env, secret key env)
    "fraudlens-pipeline": ("PIPELINE_ACCESS_KEY", "PIPELINE_SECRET_KEY", "pipeline.json"),
    "fraudlens-trino": ("TRINO_S3_ACCESS_KEY", "TRINO_S3_SECRET_KEY", "trino.json"),
    "fraudlens-analyst": ("ANALYST_ACCESS_KEY", "ANALYST_SECRET_KEY", "analyst.json"),
    "fraudlens-ingest": ("INGEST_ACCESS_KEY", "INGEST_SECRET_KEY", "ingest.json"),
}


def load_env_file() -> None:
    env = HERE.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main() -> int:
    load_env_file()
    from minio import Minio, MinioAdmin
    from minio.commonconfig import ENABLED
    from minio.credentials import StaticProvider
    from minio.versioningconfig import VersioningConfig

    endpoint = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
    root_user, root_pw = os.environ["MINIO_ROOT_USER"], os.environ["MINIO_ROOT_PASSWORD"]
    s3 = Minio(endpoint, access_key=root_user, secret_key=root_pw, secure=False)

    for attempt in range(90):                      # wait for the server (up to 3 minutes)
        try:
            s3.list_buckets()
            break
        except Exception as exc:
            print(f"waiting for MinIO at {endpoint} ({type(exc).__name__}: {str(exc)[:200]})", flush=True)
            time.sleep(2)
    else:
        print("MinIO did not become ready", file=sys.stderr)
        return 1

    for b in BUCKETS:
        if not s3.bucket_exists(b):
            s3.make_bucket(b)
            print(f"created bucket {b}")
    s3.set_bucket_versioning("fraudlens-raw", VersioningConfig(ENABLED))
    print("versioning enabled on fraudlens-raw")

    admin = MinioAdmin(endpoint, credentials=StaticProvider(root_user, root_pw), secure=False)
    for policy, (key_env, secret_env, policy_file) in ACCOUNTS.items():
        admin.policy_add(policy, str(HERE / "policies" / policy_file))
        access, secret = os.environ.get(key_env), os.environ.get(secret_env)
        if not access or not secret:
            print(f"skipping {policy}: {key_env}/{secret_env} not set")
            continue
        admin.user_add(access, secret)
        if hasattr(admin, "attach_policy"):
            try:
                admin.attach_policy([policy], user=access)
            except Exception as exc:          # already attached
                if "already" not in str(exc).lower():
                    raise
        else:
            admin.policy_set(policy, user=access)
        print(f"user {access} -> policy {policy}")
    print(json.dumps({"buckets": BUCKETS, "users": [os.environ[a] for a, _, _ in ACCOUNTS.values() if os.environ.get(a)]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
