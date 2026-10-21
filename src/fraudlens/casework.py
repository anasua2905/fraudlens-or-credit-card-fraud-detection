"""The analyst workflow: every scored transaction becomes a case; alerts wait for a decision.

This closes the loop the scoring path opens:

    new transaction -> online features -> model score -> decision
        approve (score < threshold)  -> case closed as "approved"
        alert   (score >= threshold) -> case "open" in the review queue
                                        -> analyst records "fraud" / "legitimate" (+ note)
                                        -> the decision is a label

Decisions inform monitoring in three ways (`feedback_summary`):

  * live precision: the share of reviewed alerts confirmed as fraud, with a Wilson
    95% interval, compared with the policy floor (config `monitoring.precision_floor`);
    a sustained drop means the threshold or the model needs review;
  * workload: the open backlog, the age of the oldest open alert and the time to decision;
  * new training data: `labelled_examples()` returns the model inputs of every decided
    case with its label, ready for the next retraining.

The store is a single SQLite file (one writer at a time, safe for the API and the dashboard
on one host). Every change is written to an audit table with who made it and when.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sqlite3
import threading
import uuid
from pathlib import Path

import pandas as pd

from .utils import PROJECT_ROOT

DEFAULT_DB = PROJECT_ROOT / "data" / "cases" / "cases.db"
LABELS = ("fraud", "legitimate")

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    transaction_id TEXT,
    received_at TEXT NOT NULL,
    ts TEXT,
    card_id TEXT,
    amount REAL,
    category TEXT,
    merchant TEXT,
    score REAL,
    threshold REAL,
    model TEXT,
    alert INTEGER,
    status TEXT NOT NULL,              -- open | approved | closed
    reasons TEXT,                      -- JSON list
    contributions TEXT,                -- JSON list of {feature, value, shap}
    features TEXT,                     -- JSON object: the model inputs used for this score
    clear_below_amount REAL,           -- counterfactual: an amount at which it would have been approved
    source TEXT,                       -- api | dashboard | replay
    decision TEXT,                     -- fraud | legitimate
    analyst TEXT,
    note TEXT,
    decided_at TEXT
);
CREATE INDEX IF NOT EXISTS cases_status ON cases(status);
CREATE INDEX IF NOT EXISTS cases_txn ON cases(transaction_id);
CREATE TABLE IF NOT EXISTS audit (
    at TEXT NOT NULL,
    actor TEXT,
    role TEXT,
    action TEXT NOT NULL,
    case_id TEXT,
    detail TEXT
);
"""


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """Wilson score interval for a proportion k/n (well behaved for small n and rates near 0 or 1)."""
    if n <= 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(max(0.0, mid - half), 4), round(min(1.0, mid + half), 4)


class CaseStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.environ.get("FRAUDLENS_CASE_DB") or DEFAULT_DB)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._con = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._con.row_factory = sqlite3.Row
        with self._lock:
            self._con.execute("PRAGMA journal_mode=WAL")
            self._con.executescript(SCHEMA)
            self._con.commit()

    # ------------------------------------------------------------ writing
    def _audit(self, action: str, case_id: str | None, actor: str | None, role: str | None, detail: dict | None):
        self._con.execute("INSERT INTO audit VALUES (?, ?, ?, ?, ?, ?)",
                          (_now(), actor, role, action, case_id, json.dumps(detail or {}, default=str)))

    def record_scored(self, result: dict, txn: dict | None = None, source: str = "api",
                      actor: str | None = None, role: str | None = None) -> str:
        """Store a scored transaction. Alerts open a case for review; the rest are closed as approved."""
        txn = txn or {}
        case_id = "C-" + uuid.uuid4().hex[:10]
        alert = bool(result.get("alert"))
        row = (case_id, result.get("transaction_id") or txn.get("transaction_id"), _now(), result.get("ts"),
               result.get("card_id"), result.get("amount"), result.get("category"), txn.get("merchant"),
               result.get("score"), result.get("threshold"), result.get("model"), int(alert),
               "open" if alert else "approved", json.dumps(result.get("reasons") or []),
               json.dumps(result.get("contributions") or [], default=str),
               json.dumps(result.get("features") or {}, default=str), result.get("clear_below_amount"),
               source, None, None, None, None)
        with self._lock:
            self._con.execute(f"INSERT INTO cases VALUES ({', '.join('?' * len(row))})", row)
            self._audit("scored", case_id, actor, role, {"score": result.get("score"), "alert": alert,
                                                         "source": source})
            self._con.commit()
        return case_id

    def decide(self, case_id: str, label: str, analyst: str, note: str | None = None,
               role: str | None = "analyst") -> dict:
        """Record an analyst's decision on a case (an alert, or an approved transaction later disputed)."""
        if label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}")
        if not analyst:
            raise ValueError("an analyst name is required")
        with self._lock:
            cur = self._con.execute("SELECT status FROM cases WHERE case_id = ?", (case_id,)).fetchone()
            if cur is None:
                raise KeyError(case_id)
            self._con.execute("UPDATE cases SET status = 'closed', decision = ?, analyst = ?, note = ?, "
                              "decided_at = ? WHERE case_id = ?", (label, analyst, note, _now(), case_id))
            self._audit("decided", case_id, analyst, role, {"label": label, "previous_status": cur["status"],
                                                            "note": note})
            self._con.commit()
        return self.get(case_id)

    def decide_transaction(self, transaction_id: str, label: str, analyst: str, note: str | None = None,
                           extra: dict | None = None, role: str | None = "analyst") -> dict:
        """Decision by transaction id; a historical (replayed) transaction gets a case created for it."""
        with self._lock:
            hit = self._con.execute("SELECT case_id FROM cases WHERE transaction_id = ? "
                                    "ORDER BY received_at DESC LIMIT 1", (transaction_id,)).fetchone()
        if hit is None:
            extra = dict(extra or {})
            source = extra.pop("source", "replay")
            res = {"transaction_id": transaction_id, "alert": True, **extra}
            case_id = self.record_scored(res, extra, source=source, actor=analyst, role=role)
        else:
            case_id = hit["case_id"]
        return self.decide(case_id, label, analyst, note, role)

    # ------------------------------------------------------------ reading
    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        for k in ("reasons", "contributions", "features"):
            d[k] = json.loads(d[k]) if d.get(k) else ([] if k != "features" else {})
        d["alert"] = bool(d["alert"])
        return d

    def get(self, case_id: str) -> dict:
        with self._lock:
            r = self._con.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        if r is None:
            raise KeyError(case_id)
        return self._row(r)

    def cases(self, status: str | None = None, limit: int = 200) -> list[dict]:
        sql, args = "SELECT * FROM cases", []
        if status:
            sql, args = sql + " WHERE status = ?", [status]
        sql += " ORDER BY CASE status WHEN 'open' THEN 0 ELSE 1 END, score DESC, received_at DESC LIMIT ?"
        with self._lock:
            rows = self._con.execute(sql, [*args, int(limit)]).fetchall()
        return [self._row(r) for r in rows]

    def frame(self) -> pd.DataFrame:
        with self._lock:
            return pd.read_sql_query("SELECT case_id, transaction_id, received_at, ts, card_id, amount, category, "
                                     "score, threshold, alert, status, source, decision, analyst, note, decided_at "
                                     "FROM cases", self._con)

    def audit_log(self, limit: int = 200) -> pd.DataFrame:
        with self._lock:
            return pd.read_sql_query("SELECT * FROM audit ORDER BY at DESC LIMIT ?", self._con, params=(limit,))

    def labelled_examples(self) -> pd.DataFrame:
        """Model inputs + analyst label for every decided case (input to the next retraining)."""
        with self._lock:
            rows = self._con.execute("SELECT case_id, transaction_id, ts, card_id, score, decision, features "
                                     "FROM cases WHERE decision IS NOT NULL").fetchall()
        out = []
        for r in rows:
            feats = json.loads(r["features"]) if r["features"] else {}
            out.append({"case_id": r["case_id"], "transaction_id": r["transaction_id"], "ts": r["ts"],
                        "card_id": r["card_id"], "score": r["score"], **feats,
                        "is_fraud": int(r["decision"] == "fraud")})
        return pd.DataFrame(out)

    def feedback_summary(self, precision_floor: float = 0.40, min_decisions: int = 20,
                         window_days: int | None = None) -> dict:
        """How analyst decisions inform monitoring (see the module docstring)."""
        f = self.frame()
        now = pd.Timestamp.now()
        if window_days is not None and len(f):
            f = f[pd.to_datetime(f["received_at"]) >= now - pd.Timedelta(days=window_days)]
        scored, alerts = len(f), int(f["alert"].sum()) if len(f) else 0
        reviewed = f[(f["alert"] == 1) & f["decision"].notna()] if len(f) else f
        confirmed = int((reviewed["decision"] == "fraud").sum()) if len(reviewed) else 0
        n = len(reviewed)
        low, high = wilson(confirmed, n)
        open_ = f[f["status"] == "open"] if len(f) else f
        age = (now - pd.to_datetime(open_["received_at"])).dt.total_seconds() / 3600 if len(open_) else pd.Series(dtype=float)
        decided = f[f["decision"].notna()] if len(f) else f
        ttd = ((pd.to_datetime(decided["decided_at"]) - pd.to_datetime(decided["received_at"])).dt.total_seconds() / 3600
               if len(decided) else pd.Series(dtype=float))
        missed = int(((f["alert"] == 0) & (f["decision"] == "fraud")).sum()) if len(f) else 0
        precision = round(confirmed / n, 4) if n else None
        if n < min_decisions:
            verdict, reason = "insufficient", f"{n} reviewed alerts; at least {min_decisions} are needed"
        elif high is not None and high < precision_floor:
            verdict, reason = "review_threshold", (f"live precision {precision:.1%} (95% CI {low:.1%}-{high:.1%}) is "
                                                   f"below the {precision_floor:.0%} floor")
        else:
            verdict, reason = "ok", f"live precision {precision:.1%} is consistent with the {precision_floor:.0%} floor"
        by_day = []
        if len(decided):
            d = decided.assign(day=pd.to_datetime(decided["decided_at"]).dt.date.astype(str))
            by_day = (d.groupby("day").agg(decisions=("case_id", "size"),
                                           confirmed_fraud=("decision", lambda s: int((s == "fraud").sum())))
                      .reset_index().to_dict("records"))
        return {
            "generated_at": _now(), "store": str(self.path), "window_days": window_days,
            "scored": scored, "alerts": alerts, "approved": scored - alerts,
            "reviewed_alerts": n, "confirmed_fraud": confirmed, "confirmed_legitimate": n - confirmed,
            "live_precision": precision, "live_precision_ci95": [low, high],
            "fraud_reported_on_approved": missed,
            "open_alerts": int(len(open_)), "oldest_open_hours": round(float(age.max()), 2) if len(age) else None,
            "median_hours_to_decision": round(float(ttd.median()), 2) if len(ttd) else None,
            "analysts": sorted(decided["analyst"].dropna().unique().tolist()) if len(decided) else [],
            "labelled_examples": int(len(decided)),
            "policy": {"precision_floor": precision_floor, "min_decisions": min_decisions},
            "verdict": verdict, "reason": reason, "by_day": by_day,
        }


_STORES: dict[str, CaseStore] = {}


def get_store(path: str | Path | None = None) -> CaseStore:
    """One CaseStore per file per process."""
    key = str(Path(path or os.environ.get("FRAUDLENS_CASE_DB") or DEFAULT_DB).resolve())
    if key not in _STORES:
        _STORES[key] = CaseStore(key)
    return _STORES[key]
