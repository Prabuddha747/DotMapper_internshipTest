from pathlib import Path

import pandas as pd
import pytest

from app.ingestion import load_tickets

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "support_tickets.csv"


def test_loads_all_rows():
    df = load_tickets(CSV_PATH)
    assert len(df) == 500


def test_ticket_id_unique():
    df = load_tickets(CSV_PATH)
    assert df["ticket_id"].is_unique


def test_created_at_is_datetime():
    df = load_tickets(CSV_PATH)
    assert pd.api.types.is_datetime64_any_dtype(df["created_at"])


def test_null_counts_match_verified_dataset():
    df = load_tickets(CSV_PATH)
    assert df["resolution_time_hrs"].isna().sum() == 173
    assert df["customer_rating"].isna().sum() == 173
    assert df["response_time_hrs"].isna().sum() == 0


def test_missing_column_raises(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("ticket_id,created_at,category\nTKT-001,2024-01-01 00:00,Billing\n")

    with pytest.raises(ValueError, match="missing required columns"):
        load_tickets(bad_csv)


def test_duplicate_ticket_id_raises(tmp_path):
    header = ",".join(
        [
            "ticket_id", "created_at", "category", "priority", "status",
            "response_time_hrs", "resolution_time_hrs", "agent_id",
            "customer_rating", "issue_summary",
        ]
    )
    row = "TKT-001,2024-01-01 00:00,Billing,Low,Resolved,1.0,2.0,AGT-01,5,dup test"
    dup_csv = tmp_path / "dup.csv"
    dup_csv.write_text(f"{header}\n{row}\n{row}\n")

    with pytest.raises(ValueError, match="duplicate ticket_id"):
        load_tickets(dup_csv)
