import json
from pathlib import Path

import pandas as pd

from app.ingestion import load_tickets
from app.anomaly import detect_anomalies, summary

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "support_tickets.csv"


def _df():
    return load_tickets(CSV_PATH)


# --- ground-truth counts, hand-verified against the real CSV (see docs/TRD.md §9) ---

def test_overdue_unresolved_matches_verified_count():
    hits = [h for h in detect_anomalies(_df()) if h["rule"] == "overdue_unresolved"]
    assert len(hits) == 80


def test_resolution_outlier_matches_verified_count():
    hits = [h for h in detect_anomalies(_df()) if h["rule"] == "resolution_outlier"]
    assert len(hits) == 21


def test_overdue_unresolved_explains_the_deviation():
    # auditability: a flagged ticket must say *why* — the threshold and how
    # far over it, not just the raw age.
    hits = [h for h in detect_anomalies(_df()) if h["rule"] == "overdue_unresolved"]
    for h in hits:
        assert h["threshold_hours"] == 24
        assert h["over_by_hours"] == round(h["age_hours"] - 24, 1)
        assert h["over_by_hours"] > 0


def test_resolution_outlier_explains_the_deviation():
    hits = [h for h in detect_anomalies(_df()) if h["rule"] == "resolution_outlier"]
    for h in hits:
        assert h["deviation_hrs"] == round(h["resolution_time_hrs"] - h["threshold_hrs"], 1)
        assert h["deviation_hrs"] > 0


def test_data_quality_clean_on_real_data():
    # the provided CSV has no missing/invalid values outside the expected
    # null pattern, so this rule should find nothing here (it's still
    # exercised on synthetic bad rows below).
    hits = [h for h in detect_anomalies(_df()) if h["rule"] == "data_quality"]
    assert hits == []


def test_summary_matches_total_and_is_json_serializable():
    result = summary(_df())
    assert result["data"]["total"] == 80 + 21
    assert result["rows_used"] == 80 + 21
    json.dumps(result)


# --- edge cases, synthetic data ---

def _minimal_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def test_rule_a_boundary_is_strictly_greater_than_24h():
    # reference "now" is max(created_at) in the data, so an anchor row at
    # `now` itself sets that reference without being flagged (Resolved).
    now = pd.Timestamp("2024-01-02 00:00")
    df = _minimal_df([
        {"ticket_id": "TKT-ANCHOR", "created_at": now, "status": "Resolved",
         "priority": "Low", "category": "General", "resolution_time_hrs": 1.0, "customer_rating": 5},
        # exactly 24h old: must NOT be flagged ("older than 24 hours")
        {"ticket_id": "TKT-A", "created_at": now - pd.Timedelta(hours=24), "status": "Open",
         "priority": "Critical", "category": "Technical", "resolution_time_hrs": None, "customer_rating": None},
        # 24h01m old: must be flagged
        {"ticket_id": "TKT-B", "created_at": now - pd.Timedelta(hours=24, minutes=1), "status": "Open",
         "priority": "Critical", "category": "Technical", "resolution_time_hrs": None, "customer_rating": None},
    ])
    hits = {h["ticket_id"] for h in detect_anomalies(df) if h["rule"] == "overdue_unresolved"}
    assert hits == {"TKT-B"}


def test_rule_b_skips_when_too_few_resolved_tickets():
    df = _minimal_df([
        {"ticket_id": "TKT-A", "created_at": "2024-01-01", "status": "Resolved", "priority": "Low",
         "category": "General", "resolution_time_hrs": 500.0, "customer_rating": 5},
    ])
    hits = [h for h in detect_anomalies(df) if h["rule"] == "resolution_outlier"]
    assert hits == []  # one data point can't support an IQR — must not flag everything


def test_rule_c_flags_missing_fields_on_resolved_ticket():
    df = _minimal_df([
        {"ticket_id": "TKT-A", "created_at": "2024-01-01", "status": "Resolved", "priority": "Low",
         "category": "General", "resolution_time_hrs": None, "customer_rating": 5},
    ])
    hits = [h for h in detect_anomalies(df) if h["rule"] == "data_quality"]
    assert len(hits) == 1
    assert hits[0]["ticket_id"] == "TKT-A"


def test_rule_c_flags_invalid_enum_value():
    df = _minimal_df([
        {"ticket_id": "TKT-A", "created_at": "2024-01-01", "status": "Resolved", "priority": "Urgent",
         "category": "General", "resolution_time_hrs": 2.0, "customer_rating": 5},
    ])
    hits = [h for h in detect_anomalies(df) if h["rule"] == "data_quality"]
    assert any("priority" in h["issue"] for h in hits)
