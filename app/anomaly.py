"""Deterministic anomaly detection over the tickets DataFrame.

Three rules (docs/TRD.md §9): overdue high-priority unresolved tickets,
statistical resolution-time outliers (IQR), and data-quality violations.
Pure pandas, no ML/training — explainable and reproducible.
"""
import pandas as pd

ENUM_VALUES = {
    "category": {"Billing", "Technical", "General"},
    "priority": {"Low", "Medium", "High", "Critical"},
    "status": {"Open", "Resolved", "Escalated"},
}


def _rule_overdue_unresolved(df: pd.DataFrame) -> list[dict]:
    """Rule A: unresolved High/Critical tickets older than 24h.

    "Now" = latest created_at in the dataset, not the wall clock — the data
    is historical (Jan-Mar 2024), so a live clock would flag everything.
    """
    reference_now = df["created_at"].max()
    age_hours = (reference_now - df["created_at"]).dt.total_seconds() / 3600
    overdue = age_hours > 24
    high_priority_unresolved = df["status"].isin(["Open", "Escalated"]) & df["priority"].isin(["High", "Critical"])
    hits = df[high_priority_unresolved & overdue]
    return [
        {
            "rule": "overdue_unresolved",
            "ticket_id": row.ticket_id,
            "priority": row.priority,
            "status": row.status,
            "age_hours": round(age_hours[idx], 1),
            "threshold_hours": 24,
            "over_by_hours": round(age_hours[idx] - 24, 1),
        }
        for idx, row in hits.iterrows()
    ]


def _rule_resolution_outlier(df: pd.DataFrame) -> list[dict]:
    """Rule B: resolution_time_hrs beyond Q3 + 1.5*IQR (resolved tickets only)."""
    resolved = df[df["status"] == "Resolved"]
    resolution = resolved["resolution_time_hrs"].dropna()

    # ponytail: fewer than 4 points makes quartiles meaningless (could flag
    # everything or nothing); skip rather than emit a bogus threshold.
    if len(resolution) < 4:
        return []

    q1, q3 = resolution.quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    hits = resolved[resolved["resolution_time_hrs"] > upper]
    return [
        {
            "rule": "resolution_outlier",
            "ticket_id": row.ticket_id,
            "resolution_time_hrs": row.resolution_time_hrs,
            "threshold_hrs": round(upper, 1),
            "deviation_hrs": round(row.resolution_time_hrs - upper, 1),
        }
        for _, row in hits.iterrows()
    ]


def _rule_data_quality(df: pd.DataFrame) -> list[dict]:
    """Rule C: resolved tickets missing required fields, or invalid enum values."""
    hits = []

    incomplete = df[
        (df["status"] == "Resolved")
        & (df["resolution_time_hrs"].isna() | df["customer_rating"].isna())
    ]
    for _, row in incomplete.iterrows():
        hits.append({
            "rule": "data_quality",
            "ticket_id": row.ticket_id,
            "issue": "resolved but missing resolution_time_hrs or customer_rating",
        })

    for col, allowed in ENUM_VALUES.items():
        for _, row in df[~df[col].isin(allowed)].iterrows():
            hits.append({
                "rule": "data_quality",
                "ticket_id": row.ticket_id,
                "issue": f"invalid {col} value: {row[col]!r}",
            })

    return hits


def detect_anomalies(df: pd.DataFrame) -> list[dict]:
    """Run all three rules, return every flagged ticket tagged by rule."""
    return _rule_overdue_unresolved(df) + _rule_resolution_outlier(df) + _rule_data_quality(df)


def summary(df: pd.DataFrame, **_) -> dict:
    """query_engine-shaped response for the 'anomaly_summary' NL operation."""
    hits = detect_anomalies(df)
    by_rule: dict[str, int] = {}
    for hit in hits:
        by_rule[hit["rule"]] = by_rule.get(hit["rule"], 0) + 1

    answer = (
        f"{len(hits)} anomalies found: " + ", ".join(f"{count} {rule}" for rule, count in by_rule.items())
        if hits else "No anomalies found."
    )
    return {"answer": answer, "data": {"total": len(hits), "by_rule": by_rule, "anomalies": hits}, "rows_used": len(hits)}
