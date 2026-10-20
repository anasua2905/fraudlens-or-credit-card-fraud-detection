"""Build (or check) the FraudLens ingestion flow in Apache NiFi through its REST API.

Lifecycle stage 2 (data acquisition) on the platform: NiFi is the crawler/ingest
tool. Files dropped into the landing folder are picked up, given a lake key,
written to the raw layer in MinIO, and an ingest record (file size, ETag,
version, timestamp: the metadata collection step) is written next to them.

    ListFile(/landing) -> FetchFile -> UpdateAttribute(lake.key, ingested.at)
        -> PutS3Object(s3://fraudlens-raw/<lake.key>)
        -> AttributesToJSON -> PutS3Object(s3://fraudlens-raw/sparkov/_ingest_log/<file>.json)

    fraudTrain.csv  ->  s3://fraudlens-raw/sparkov/train/fraudTrain.csv
    fraudTest.csv   ->  s3://fraudlens-raw/sparkov/test/fraudTest.csv
    manifest.json   ->  s3://fraudlens-raw/sparkov/_manifest/manifest.json

Properties are set by their display names (as shown in the NiFi UI); the script
looks up each processor's property descriptors, so it does not depend on
internal property names that change between NiFi releases.

Runs in docker compose (service nifi-init) or from the host:
    python platform/nifi/build_flow.py                 # build if missing, then start
    python platform/nifi/build_flow.py --rebuild       # delete and rebuild the flow
    python platform/nifi/build_flow.py --status        # print queue/processor status
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)   # NiFi's self-signed certificate

HERE = Path(__file__).resolve().parent
GROUP_NAME = "FraudLens ingest"
RAW_KEY = ("${filename:endsWith('.csv'):ifElse("
           "${filename:replace('fraud', ''):replace('.csv', ''):toLower():prepend('sparkov/'):append('/')"
           ":append(${filename})}, "
           "${filename:prepend('sparkov/_manifest/')})}")


def load_env_file() -> None:
    env = HERE.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


class NiFi:
    def __init__(self, url: str, user: str, password: str):
        self.url = url.rstrip("/") + "/nifi-api"
        self.s = requests.Session()
        self.s.verify = False
        self.user, self.password = user, password

    # -------------------------------------------------------------- plumbing
    def login(self, timeout: int = 420) -> None:
        t0 = time.time()
        while True:
            try:
                r = self.s.post(f"{self.url}/access/token", data={"username": self.user, "password": self.password},
                                timeout=10)
                if r.status_code in (200, 201):
                    self.s.headers["Authorization"] = f"Bearer {r.text.strip()}"
                    print("logged in to NiFi")
                    return
                msg = f"HTTP {r.status_code}: {r.text[:200]}"
            except requests.RequestException as exc:
                msg = type(exc).__name__
            if time.time() - t0 > timeout:
                raise SystemExit(f"NiFi did not become ready at {self.url} ({msg})")
            print(f"waiting for NiFi ({msg})")
            time.sleep(10)

    def req(self, method: str, path: str, **kw):
        r = self.s.request(method, f"{self.url}{path}", timeout=60, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"{method} {path} -> HTTP {r.status_code}: {r.text[:500]}")
        return r.json() if r.text and r.headers.get("content-type", "").startswith("application/json") else r.text

    def get(self, path, **kw):
        return self.req("GET", path, **kw)

    def post(self, path, body):
        return self.req("POST", path, json=body)

    def put(self, path, body):
        return self.req("PUT", path, json=body)

    # -------------------------------------------------------------- types
    def _type(self, kind: str, simple_name: str) -> dict:
        key = "processorTypes" if kind == "processor" else "controllerServiceTypes"
        path = "/flow/processor-types" if kind == "processor" else "/flow/controller-service-types"
        types = self.get(path)[key]
        hits = [t for t in types if t["type"].split(".")[-1] == simple_name]
        if not hits:
            raise SystemExit(f"NiFi has no {kind} type {simple_name}; is the bundle (NAR) installed?")
        t = sorted(hits, key=lambda t: t["bundle"]["version"])[-1]
        return {"type": t["type"], "bundle": t["bundle"]}

    @staticmethod
    def _map_properties(component: dict, wanted: dict[str, str]) -> dict[str, str]:
        """Translate display names (or names) to property names; dynamic properties pass through."""
        desc = component.get("descriptors", {})
        by_display = {d["displayName"].lower(): d for d in desc.values()}
        out = {}
        for k, v in wanted.items():
            d = desc.get(k) or by_display.get(k.lower())
            if d is None:
                out[k] = v                                # dynamic property (e.g. UpdateAttribute)
                continue
            allowed = d.get("allowableValues") or []
            if allowed and v is not None:
                vals = {a["allowableValue"]["value"].lower(): a["allowableValue"]["value"] for a in allowed}
                vals.update({a["allowableValue"]["displayName"].lower(): a["allowableValue"]["value"]
                             for a in allowed})
                if v.lower() not in vals and not d.get("identifiesControllerService"):
                    raise SystemExit(f"{k}={v!r} not in allowed values {sorted(vals)}")
                v = vals.get(v.lower(), v)
            out[d["name"]] = v
        return out

    # -------------------------------------------------------------- components
    def root_id(self) -> str:
        return self.get("/flow/process-groups/root")["processGroupFlow"]["id"]

    def find_group(self, parent: str, name: str) -> dict | None:
        for g in self.get(f"/process-groups/{parent}/process-groups")["processGroups"]:
            if g["component"]["name"] == name:
                return g
        return None

    def create_group(self, parent: str, name: str) -> str:
        g = self.post(f"/process-groups/{parent}/process-groups",
                      {"revision": {"version": 0}, "component": {"name": name, "position": {"x": 0, "y": 0}}})
        return g["id"]

    def create_service(self, pg: str, simple_type: str, name: str, props: dict) -> str:
        t = self._type("service", simple_type)
        cs = self.post(f"/process-groups/{pg}/controller-services",
                       {"revision": {"version": 0}, "component": {**t, "name": name}})
        mapped = self._map_properties(cs["component"], props)
        cs = self.put(f"/controller-services/{cs['id']}",
                      {"revision": cs["revision"], "component": {"id": cs["id"], "properties": mapped}})
        return cs["id"]

    def enable_service(self, sid: str) -> None:
        cs = self.get(f"/controller-services/{sid}")
        if cs["component"]["state"] != "ENABLED":
            if cs["component"].get("validationErrors"):
                raise SystemExit(f"controller service invalid: {cs['component']['validationErrors']}")
            self.put(f"/controller-services/{sid}/run-status", {"revision": cs["revision"], "state": "ENABLED"})
        for _ in range(30):
            if self.get(f"/controller-services/{sid}")["component"]["state"] == "ENABLED":
                return
            time.sleep(1)
        raise SystemExit("controller service did not enable")

    def create_processor(self, pg: str, simple_type: str, name: str, x: int, y: int, props: dict,
                         auto_terminate: list[str] = (), schedule: str | None = None) -> str:
        t = self._type("processor", simple_type)
        p = self.post(f"/process-groups/{pg}/processors",
                      {"revision": {"version": 0},
                       "component": {**t, "name": name, "position": {"x": x, "y": y}}})
        config = {"properties": self._map_properties(p["component"]["config"], props),
                  "autoTerminatedRelationships": list(auto_terminate)}
        if schedule:
            config["schedulingPeriod"] = schedule
        p = self.put(f"/processors/{p['id']}",
                     {"revision": p["revision"], "component": {"id": p["id"], "config": config}})
        return p["id"]

    def connect(self, pg: str, src: str, dst: str, rels: list[str]) -> None:
        self.post(f"/process-groups/{pg}/connections",
                  {"revision": {"version": 0},
                   "component": {"source": {"id": src, "groupId": pg, "type": "PROCESSOR"},
                                 "destination": {"id": dst, "groupId": pg, "type": "PROCESSOR"},
                                 "selectedRelationships": rels}})

    def processors(self, pg: str) -> list[dict]:
        return self.get(f"/process-groups/{pg}/processors")["processors"]

    def set_group_state(self, pg: str, state: str) -> None:
        self.put(f"/flow/process-groups/{pg}", {"id": pg, "state": state})

    def delete_group(self, pg: str) -> None:
        self.set_group_state(pg, "STOPPED")
        time.sleep(2)
        for c in self.get(f"/process-groups/{pg}/connections")["connections"]:
            self.post(f"/flowfile-queues/{c['id']}/drop-requests", {})
        time.sleep(2)
        self.put(f"/flow/process-groups/{pg}/controller-services", {"id": pg, "state": "DISABLED"})
        time.sleep(3)
        g = self.get(f"/process-groups/{pg}")
        self.req("DELETE", f"/process-groups/{pg}", params={"version": g["revision"]["version"]})


def build(nifi: NiFi, landing: str, s3_endpoint: str, bucket: str, key: str, secret: str) -> str:
    root = nifi.root_id()
    pg = nifi.create_group(root, GROUP_NAME)
    creds = nifi.create_service(pg, "AWSCredentialsProviderControllerService", "MinIO ingest credentials",
                                {"Access Key ID": key, "Secret Access Key": secret})
    nifi.enable_service(creds)
    s3 = {"Bucket": bucket, "Region": "us-east-1", "AWS Credentials Provider Service": creds,
          "Endpoint Override URL": s3_endpoint, "Use Path Style Access": "true"}

    lst = nifi.create_processor(pg, "ListFile", "List landing folder", 0, 0,
                                {"Input Directory": landing, "File Filter": r"[^\.].*\.(csv|json)",
                                 "Recurse Subdirectories": "false"}, schedule="10 sec")
    fetch = nifi.create_processor(pg, "FetchFile", "Fetch file", 0, 200, {},
                                  auto_terminate=["not.found", "permission.denied", "failure"])
    tag = nifi.create_processor(pg, "UpdateAttribute", "Assign lake key", 0, 400,
                                {"lake.key": RAW_KEY,
                                 "ingested.at": "${now():format(\"yyyy-MM-dd'T'HH:mm:ss.SSSXXX\", 'UTC')}"})
    put_raw = nifi.create_processor(pg, "PutS3Object", "Write to raw layer (MinIO)", 0, 600,
                                    {**s3, "Object Key": "${lake.key}"})
    to_json = nifi.create_processor(pg, "AttributesToJSON", "Build ingest record", 0, 800,
                                    {"Attributes List": "filename,file.size,file.lastModifiedTime,lake.key,"
                                                        "ingested.at,s3.bucket,s3.etag,s3.version,uuid",
                                     "Destination": "flowfile-content", "Include Core Attributes": "true"},
                                    auto_terminate=["failure"])
    put_log = nifi.create_processor(pg, "PutS3Object", "Write ingest record", 0, 1000,
                                    {**s3, "Object Key": "sparkov/_ingest_log/${filename}.json",
                                     "Content Type": "application/json"},
                                    auto_terminate=["success"])
    nifi.connect(pg, lst, fetch, ["success"])
    nifi.connect(pg, fetch, tag, ["success"])
    nifi.connect(pg, tag, put_raw, ["success"])
    nifi.connect(pg, put_raw, to_json, ["success"])
    nifi.connect(pg, put_raw, put_raw, ["failure"])          # retry (failed FlowFiles are penalised)
    nifi.connect(pg, to_json, put_log, ["success"])
    nifi.connect(pg, put_log, put_log, ["failure"])
    return pg


def check_valid(nifi: NiFi, pg: str) -> None:
    bad = []
    for _ in range(15):
        bad = [(p["component"]["name"], p["component"].get("validationErrors"))
               for p in nifi.processors(pg) if p["component"].get("validationStatus") == "INVALID"]
        if not bad:
            return
        time.sleep(2)
    raise SystemExit("invalid processors: " + json.dumps(bad, indent=1))


def status(nifi: NiFi, pg: str) -> dict:
    st = nifi.get(f"/flow/process-groups/{pg}/status")["processGroupStatus"]["aggregateSnapshot"]
    out = {"flowFilesIn": st.get("flowFilesIn"), "bytesIn": st.get("bytesIn"),
           "flowFilesQueued": st.get("flowFilesQueued"),
           "processors": {p["component"]["name"]: p["component"]["state"] for p in nifi.processors(pg)}}
    return out


def main() -> int:
    load_env_file()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=os.environ.get("NIFI_URL", "https://localhost:8443"))
    ap.add_argument("--landing", default=os.environ.get("NIFI_LANDING_DIR", "/landing"))
    ap.add_argument("--s3-endpoint", default=os.environ.get("NIFI_S3_ENDPOINT", "http://minio:9000"))
    ap.add_argument("--bucket", default="fraudlens-raw")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    nifi = NiFi(a.url, os.environ["NIFI_USERNAME"], os.environ["NIFI_PASSWORD"])
    nifi.login()
    root = nifi.root_id()
    existing = nifi.find_group(root, GROUP_NAME)
    if a.status:
        print(json.dumps(status(nifi, existing["id"]) if existing else {"flow": "missing"}, indent=1))
        return 0
    if existing and a.rebuild:
        nifi.delete_group(existing["id"])
        print("deleted the existing flow")
        existing = None
    if existing:
        pg = existing["id"]
        print(f"flow '{GROUP_NAME}' already exists ({pg})")
    else:
        pg = build(nifi, a.landing, a.s3_endpoint, a.bucket,
                   os.environ["INGEST_ACCESS_KEY"], os.environ["INGEST_SECRET_KEY"])
        print(f"built flow '{GROUP_NAME}' ({pg})")
    check_valid(nifi, pg)
    nifi.set_group_state(pg, "RUNNING")
    time.sleep(3)
    print(json.dumps(status(nifi, pg), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
