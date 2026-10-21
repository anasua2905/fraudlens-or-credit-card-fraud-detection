"""The analyst workflow store: cases, decisions, audit trail and what the decisions tell monitoring."""
from __future__ import annotations

import pytest

from fraudlens import security
from fraudlens.casework import CaseStore, wilson


def _result(i: int, alert: bool, score: float) -> dict:
    return {"transaction_id": f"t{i}", "card_id": f"c{i % 3}", "ts": "2020-12-01T10:00:00", "amount": 100.0 + i,
            "category": "shopping_net", "score": score, "threshold": 0.74, "model": "hist_gb", "alert": alert,
            "reasons": ["Amount is 9x this card's usual spend"], "features": {"amount": 100.0 + i, "hour": 10}}


def test_cases_decisions_and_feedback_summary(tmp_path):
    s = CaseStore(tmp_path / "cases.db")
    ids = [s.record_scored(_result(i, i % 2 == 0, 0.9 if i % 2 == 0 else 0.1), source="test") for i in range(50)]
    assert len(s.cases("open", 500)) == 25 and len(s.cases("approved", 500)) == 25
    for n, cid in enumerate(ids[::2]):                      # review every alert: 60% confirmed fraud
        s.decide(cid, "fraud" if n % 5 < 3 else "legitimate", "alice")
    s.decide(ids[1], "fraud", "bob", "customer reported it")  # an approved transaction later disputed
    fb = s.feedback_summary(precision_floor=0.4, min_decisions=20)
    assert fb["reviewed_alerts"] == 25 and fb["confirmed_fraud"] == 15 and fb["live_precision"] == 0.6
    assert fb["open_alerts"] == 0 and fb["fraud_reported_on_approved"] == 1
    lo, hi = fb["live_precision_ci95"]
    assert lo < 0.6 < hi and fb["verdict"] == "ok"
    ex = s.labelled_examples()
    assert len(ex) == 26 and set(ex["is_fraud"]) == {0, 1} and "amount" in ex
    assert set(s.audit_log(500)["action"]) == {"scored", "decided"}
    with pytest.raises(ValueError):
        s.decide(ids[0], "maybe", "alice")
    with pytest.raises(KeyError):
        s.decide("C-none", "fraud", "alice")


def test_low_live_precision_triggers_a_threshold_review(tmp_path):
    s = CaseStore(tmp_path / "cases.db")
    for i in range(40):
        s.decide(s.record_scored(_result(i, True, 0.8)), "fraud" if i < 4 else "legitimate", "alice")
    fb = s.feedback_summary(precision_floor=0.4)
    assert fb["live_precision"] == 0.1 and fb["verdict"] == "review_threshold"


def test_replayed_transaction_decision_creates_a_case(tmp_path):
    s = CaseStore(tmp_path / "cases.db")
    case = s.decide_transaction("hist-1", "legitimate", "alice", extra={"score": 0.8, "source": "replay"})
    assert case["status"] == "closed" and case["source"] == "replay" and case["decision"] == "legitimate"


def test_wilson_interval():
    assert wilson(0, 0) == (None, None)
    lo, hi = wilson(15, 25)
    assert 0.39 < lo < 0.42 and 0.76 < hi < 0.79


def test_api_keys_and_dashboard_users():
    keys = security.parse_keys("scorer:k1,analyst:k2")
    assert security.role_for_key(keys, "k2") == "analyst" and security.role_for_key(keys, "bad") is None
    with pytest.raises(ValueError):
        security.parse_keys("root:k")
    h = security.hash_password("s3cret-pass")
    users = security.parse_users(f"alice:analyst:{h},bob:viewer:plain-pass")
    assert security.authenticate(users, "alice", "s3cret-pass") == "analyst"
    assert security.authenticate(users, "alice", "wrong") is None
    assert security.authenticate(users, "bob", "plain-pass") == "viewer"
