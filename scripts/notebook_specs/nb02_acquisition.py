FILENAME = "02_data_acquisition.ipynb"

CELLS = [
("md", r"""
# 2 · Data acquisition

| | |
|---|---|
| Lifecycle stage | 2 of 8: *Data acquisition* (collect data from APIs and the web, collect metadata) |
| Tools | Kaggle API client, `requests`, BeautifulSoup, Apache NiFi, MinIO (S3) |
| Code | `src/fraudlens/acquire.py`, `platform/nifi/build_flow.py`, `src/fraudlens/lake.py` |
| Outputs | `data/raw/*.csv` (bronze), `reports/acquisition_manifest.json`, `reports/acquisition_metadata.json`, raw layer in MinIO |

Acquisition has three jobs:

1. **Get the data** reproducibly: the official Kaggle API (or a local copy of the same files).
2. **Collect metadata**: provenance, licence, checksums, sizes, row counts and a column profile, so any
   later stage can prove it is using exactly these files.
3. **Land it in the lake** untouched (bronze / raw layer): directly with boto3, or through the
   **Apache NiFi** ingestion flow, which also writes an ingest record per file.
"""),
("setup", r"""
from fraudlens import acquire"""),
("md", r"""
## 2.1 Source: Kaggle API

`acquire.download_from_kaggle` uses the official `kaggle` client (credentials in `~/.kaggle/kaggle.json`
or `KAGGLE_USERNAME`/`KAGGLE_KEY`). The run recorded below copied an already-downloaded local copy
(`--source local`), which produces identical files; the checksums prove it.
"""),
("code", r"""
import inspect
print(inspect.getsource(acquire.download_from_kaggle))
man = read_json(P["reports"] / "acquisition_manifest.json")
{k: man[k] for k in ("dataset", "source_url", "licence", "acquired_via", "acquired_at_utc")}
"""),
("md", r"""
## 2.2 Metadata from the web: `requests` + BeautifulSoup

The dataset was produced with the open-source **Sparkov** generator. We collect its provenance (title,
description, licence) from the generator's public repository page, the way a data engineer documents
an external source. If the machine is offline the cell says so instead of failing.
"""),
("code", r"""
import requests
from bs4 import BeautifulSoup

def page_metadata(url: str) -> dict:
    r = requests.get(url, timeout=15, headers={"User-Agent": "FraudLens-metadata-collector/1.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    meta = lambda name: (soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name}) or {}).get("content")
    return {"url": url, "status": r.status_code, "title": soup.title.get_text(strip=True) if soup.title else None,
            "description": meta("og:description") or meta("description"),
            "licence_mentioned": next((w for w in ("CC0", "MIT", "Apache", "GPL") if w in r.text), None)}

sources = ["https://github.com/namebrandon/Sparkov_Data_Generation", man["source_url"]]
rows = []
for url in sources:
    try:
        rows.append(page_metadata(url))
    except Exception as exc:
        rows.append({"url": url, "status": f"unreachable ({type(exc).__name__})"})
pd.DataFrame(rows)
"""),
("md", r"""
## 2.3 Integrity: checksums, sizes and row counts

The manifest stores a SHA-256 checksum per file. Recomputing it here proves the bronze files have not
changed since acquisition (the verification stage does the same on every run).
"""),
("code", r"""
rows = []
for split, f in man["files"].items():
    path = P["raw"] / f["file"]
    if path.exists():
        sha = acquire.sha256sum(path)
        rows.append({"split": split, "file": f["file"], "rows": f["rows"], "MB": round(f["bytes"] / 1e6, 1),
                     "sha256 (first 16)": sha[:16], "matches manifest": sha == f["sha256"]})
    else:
        rows.append({"split": split, "file": f["file"], "rows": f["rows"], "matches manifest": "file not present"})
pd.DataFrame(rows)
"""),
("md", r"""
## 2.4 Column profile (metadata collection)

A profile of the raw columns from a 100,000-row sample: inferred type, missing values, distinct values.
It already tells us what preparation must deal with: a leftover index column (`Unnamed: 0`), timestamps
stored as text, card numbers stored as integers, and direct identifiers (`first`, `last`, `street`) that
must not leave the raw layer.
"""),
("code", r"""
meta = read_json(P["reports"] / "acquisition_metadata.json")
print(meta["description"])
prof = pd.DataFrame(meta["columns"])
PII = {"cc_num", "first", "last", "street", "dob", "zip", "lat", "long", "job"}
prof["governance"] = prof["column"].map(lambda c: "PII" if c in PII else "")
prof.loc[prof["governance"] == "PII", "example"] = "(hidden)"
prof
"""),
("code", r"""
raw_path = P["raw"] / man["files"]["train"]["file"]
safe_cols = ["trans_date_trans_time", "merchant", "category", "amt", "gender", "city", "state", "city_pop",
             "unix_time", "merch_lat", "merch_long", "is_fraud"]
if raw_path.exists():
    display(pd.read_csv(raw_path, nrows=5, usecols=safe_cols))
"""),
("md", r"""
## 2.5 Landing in the lake: MinIO and Apache NiFi

On the Docker platform the raw files land in the **raw (bronze) bucket** of MinIO, one folder per file so
the crawler can register each as an external table:

```
s3://fraudlens-raw/sparkov/train/fraudTrain.csv
s3://fraudlens-raw/sparkov/test/fraudTest.csv
s3://fraudlens-raw/sparkov/_manifest/manifest.json
s3://fraudlens-raw/sparkov/_ingest_log/<file>.json      <- NiFi ingest records
```

With `platform.ingest: nifi` (or `--ingest nifi`), the acquisition stage drops the files into
`data/landing/`, and the NiFi flow built by `platform/nifi/build_flow.py` does the rest:

```
ListFile(/landing) -> FetchFile -> UpdateAttribute(lake.key, ingested.at)
    -> PutS3Object(s3://fraudlens-raw/<lake.key>)
    -> AttributesToJSON -> PutS3Object(s3://fraudlens-raw/sparkov/_ingest_log/<file>.json)
```

NiFi writes with a **write-only service account** (`fraudlens-ingest`, see `platform/minio/policies/ingest.json`)
that can put objects under `sparkov/` and nothing else; the raw bucket is **versioned**, so a source file can
never be silently replaced. The acquisition stage waits until every file has arrived with the right size and
copies NiFi's ingest records into the manifest.
"""),
("code", r"""
from fraudlens import lake
if lake.available(cfg):
    keys = lake.list_keys(cfg, cfg["platform"]["buckets"]["raw"], "sparkov/")
    display(pd.DataFrame({"object": keys}))
    if man.get("lake", {}).get("nifi_ingest_records"):
        display(pd.DataFrame(man["lake"]["nifi_ingest_records"]).T)
else:
    print("The Docker platform is not running here, so the lake is not reachable; the pipeline ran in local mode.")
    pub = P["reports"] / "lake_publication.json"
    print("Lake publication status:", read_json(pub).get("status") if pub.exists() else "not run")
    print("The platform workflow (deploy/github/workflows/platform.yml) runs this stage through NiFi and MinIO.")
"""),
("md", r"""
## 2.6 Summary

* Two CSV files, 1,852,394 transactions, 501 MB, licence CC0, checksums recorded and re-verified.
* Provenance and a column profile are stored as JSON next to the data (`reports/`).
* The raw layer is immutable (versioned bucket, write-only ingest account), and NiFi records an ingest
  record per file for lineage.

**Next:** [03 · Data preparation](03_data_preparation.ipynb)
"""),
]
