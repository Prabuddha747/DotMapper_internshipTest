"""Load support_tickets.csv into a validated in-memory DataFrame.

No database layer: 500 rows, one reader, re-ingested on every boot — a
pandas DataFrame held in app.state IS the store (see docs/TRD.md §6).
"""
import pandas as pd

REQUIRED_COLUMNS = [
    "ticket_id", "created_at", "category", "priority", "status",
    "response_time_hrs", "resolution_time_hrs", "agent_id",
    "customer_rating", "issue_summary",
]
NUMERIC_COLUMNS = ["response_time_hrs", "resolution_time_hrs", "customer_rating"]


def load_tickets(csv_path: str) -> pd.DataFrame:
    """Read, validate, and type-cast the tickets CSV. Raises on any bad data."""
    df = pd.read_csv(csv_path)

    # Fail loud at startup, not deep inside a query, if the schema drifted.
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"support_tickets.csv missing required columns: {missing}")

    # ticket_id is the primary key everywhere downstream; must be unique.
    if df["ticket_id"].duplicated().any():
        dupes = df.loc[df["ticket_id"].duplicated(), "ticket_id"].tolist()
        raise ValueError(f"duplicate ticket_id values: {dupes}")

    # Parse types explicitly; errors="raise" surfaces malformed rows instead
    # of silently coercing them to NaN and corrupting later averages/IQR.
    df["created_at"] = pd.to_datetime(df["created_at"], errors="raise")
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df
