"""Deterministic analytics over the tickets DataFrame.

The LLM (app/llm.py) only picks an operation + filters; every number in a
response is computed here, in plain pandas, so answers can never be
hallucinated (see docs/TRD.md §5).
"""
import pandas as pd

NUMERIC_FIELDS = ["response_time_hrs", "resolution_time_hrs", "customer_rating"]


_COMPARISON_OPS = {
    "gt": lambda series, v: series > v,
    "gte": lambda series, v: series >= v,
    "lt": lambda series, v: series < v,
    "lte": lambda series, v: series <= v,
}


def _apply_filters(df: pd.DataFrame, filters: dict | None) -> tuple[pd.DataFrame, list[str]]:
    """Apply equality/isin filters, or a {"gt"/"gte"/"lt"/"lte": number}
    comparison for numeric columns (e.g. "resolution time over 12 hours").
    Unknown columns/ops are reported back instead of silently ignored."""
    unmatched = []
    for col, value in (filters or {}).items():
        if col not in df.columns:
            unmatched.append(col)
            continue
        if isinstance(value, dict):
            op = next(iter(value), None)
            if op not in _COMPARISON_OPS:
                unmatched.append(f"{col} (unsupported comparison '{op}')")
                continue
            df = df[_COMPARISON_OPS[op](df[col], value[op])]
            continue
        values = value if isinstance(value, list) else [value]
        df = df[df[col].isin(values)]
    return df, unmatched


def _validate_numeric_field(field: str | None) -> None:
    """Reject a non-numeric field before pandas throws a less readable error."""
    if field not in NUMERIC_FIELDS:
        raise ValueError(f"'{field}' is not numeric; expected one of {NUMERIC_FIELDS}")


def _records(df: pd.DataFrame, limit: int) -> list[dict]:
    """Row subset as JSON-safe dicts: datetimes to strings, NaN to None
    (bare NaN is not valid JSON and breaks browser JSON.parse)."""
    subset = df.head(limit).copy()
    subset["created_at"] = subset["created_at"].astype(str)
    subset = subset.astype(object).where(subset.notna(), None)
    return subset.to_dict(orient="records")


def _pack(answer: str, data: dict, unmatched: list[str], rows_used: int) -> dict:
    """Common response shape. rows_used is the evidence trail: how many real
    DataFrame rows this answer was computed from (docs/TRD.md §13)."""
    result = {"answer": answer, "data": data, "rows_used": rows_used}
    if unmatched:
        result["unmatched_filters"] = unmatched
    return result


def count(df: pd.DataFrame, filters: dict | None = None, **_) -> dict:
    filtered, unmatched = _apply_filters(df, filters)
    n = len(filtered)
    return _pack(f"{n} ticket(s) match those filters.", {"count": n}, unmatched, rows_used=n)


def filter_rows(df: pd.DataFrame, filters: dict | None = None, limit: int = 50, **_) -> dict:
    filtered, unmatched = _apply_filters(df, filters)
    rows = _records(filtered, limit)
    answer = f"{len(filtered)} ticket(s) match, showing {len(rows)}."
    return _pack(answer, {"count": len(filtered), "tickets": rows}, unmatched, rows_used=len(filtered))


def group_count(df: pd.DataFrame, group_by: str | None = None, filters: dict | None = None, **_) -> dict:
    if group_by not in df.columns:
        raise ValueError(f"unknown group_by column: {group_by}")
    filtered, unmatched = _apply_filters(df, filters)
    # int(v): value_counts() yields numpy.int64, which stdlib json can't serialize.
    counts = {k: int(v) for k, v in filtered[group_by].value_counts().items()}
    top = max(counts, key=counts.get) if counts else None
    answer = (
        f"Top {group_by}: {top} ({counts[top]} tickets)."
        if top else f"No tickets match those filters to group by {group_by}."
    )
    return _pack(answer, {"counts": counts, "top": top}, unmatched, rows_used=len(filtered))


def average(df: pd.DataFrame, field: str | None = None, filters: dict | None = None, **_) -> dict:
    _validate_numeric_field(field)
    filtered, unmatched = _apply_filters(df, filters)
    values = filtered[field].dropna()  # exclude nulls, don't treat as 0
    if values.empty:
        return _pack(f"No tickets with a value for {field} match those filters.", {"average": None, "matched": 0}, unmatched, rows_used=0)
    avg = round(values.mean(), 2)
    return _pack(f"The average {field} is {avg} (over {len(values)} tickets).", {"average": avg, "matched": len(values)}, unmatched, rows_used=len(values))


def total(df: pd.DataFrame, field: str | None = None, filters: dict | None = None, **_) -> dict:
    _validate_numeric_field(field)
    filtered, unmatched = _apply_filters(df, filters)
    values = filtered[field].dropna()
    result = round(values.sum(), 2) if not values.empty else 0
    return _pack(f"Total {field}: {result} (over {len(values)} tickets).", {"sum": result, "matched": len(values)}, unmatched, rows_used=len(values))


def minimum(df: pd.DataFrame, field: str | None = None, filters: dict | None = None, **_) -> dict:
    _validate_numeric_field(field)
    filtered, unmatched = _apply_filters(df, filters)
    values = filtered[field].dropna()
    result = values.min() if not values.empty else None
    return _pack(f"Minimum {field}: {result}.", {"min": result, "matched": len(values)}, unmatched, rows_used=len(values))


def maximum(df: pd.DataFrame, field: str | None = None, filters: dict | None = None, **_) -> dict:
    _validate_numeric_field(field)
    filtered, unmatched = _apply_filters(df, filters)
    values = filtered[field].dropna()
    result = values.max() if not values.empty else None
    return _pack(f"Maximum {field}: {result}.", {"max": result, "matched": len(values)}, unmatched, rows_used=len(values))


def ratio(df: pd.DataFrame, field: str | None = None, field_b: str | None = None,
          agg: str = "sum", filters: dict | None = None, **_) -> dict:
    """Sum or average of two numeric fields, and their ratio. Pure arithmetic
    on real columns — same safety class as sum()/average(), not a derived
    metric that needs its own hallucination review."""
    _validate_numeric_field(field)
    _validate_numeric_field(field_b)
    agg = agg if agg in ("sum", "average") else "sum"
    filtered, unmatched = _apply_filters(df, filters)
    a, b = filtered[field].dropna(), filtered[field_b].dropna()
    value_a = round((a.sum() if agg == "sum" else a.mean()), 2) if not a.empty else 0
    value_b = round((b.sum() if agg == "sum" else b.mean()), 2) if not b.empty else 0

    data = {"agg": agg, "field": field, "value_a": value_a, "field_b": field_b, "value_b": value_b}
    if value_b == 0:
        answer = f"{agg} of {field} is {value_a}; {agg} of {field_b} is 0, so the ratio is undefined."
        data["ratio"] = None
    else:
        data["ratio"] = round(value_a / value_b, 4)
        answer = f"{agg} of {field}: {value_a}; {agg} of {field_b}: {value_b}; ratio {field}/{field_b} = {data['ratio']}."
    return _pack(answer, data, unmatched, rows_used=len(filtered))


def equalize(df: pd.DataFrame, group_by: str | None = None, filters: dict | None = None, **_) -> dict:
    """Minimum tickets that must move between group_by's categories so each
    holds an equal share of the real current data. Pure arithmetic on real
    counts (target = total/#categories) — no hypothetical rows are simulated,
    which is why this is safe to answer unlike an arbitrary what-if (docs/TRD.md §15)."""
    if group_by not in df.columns:
        raise ValueError(f"unknown group_by column: {group_by}")
    filtered, unmatched = _apply_filters(df, filters)
    counts = filtered[group_by].value_counts()
    n, k = len(filtered), len(counts)
    if k == 0:
        return _pack(f"No tickets to equalize by {group_by}.", {"moves": {}, "total_moved": 0}, unmatched, rows_used=0)

    target = n / k
    moves = {}
    for category, count in counts.items():
        diff = round(count - target, 2)
        moves[category] = {
            "current": int(count),
            "must_give_away": max(diff, 0),
            "must_receive": max(-diff, 0),
            "pct_of_category_to_move": round(diff / count * 100, 1) if diff > 0 else 0.0,
        }
    total_moved = round(sum(m["must_give_away"] for m in moves.values()), 2)

    answer = (
        f"To make {group_by} categories equal ({round(target, 1)} each of {n}), "
        f"{total_moved} ticket(s) must move: " +
        ", ".join(
            f"{cat} gives away {m['must_give_away']} ({m['pct_of_category_to_move']}%)"
            if m["must_give_away"] else f"{cat} receives {m['must_receive']}"
            for cat, m in moves.items()
        )
    )
    return _pack(answer, {"target_per_category": round(target, 2), "total_moved": total_moved, "moves": moves}, unmatched, rows_used=n)


# operation name -> handler; extended with "anomaly_summary" in app/anomaly.py's wiring (Part 3)
OPERATIONS = {
    "count": count,
    "filter": filter_rows,
    "group_count": group_count,
    "average": average,
    "sum": total,
    "min": minimum,
    "max": maximum,
    "ratio": ratio,
    "equalize": equalize,
}


def run(df: pd.DataFrame, operation: str, **kwargs) -> dict:
    """Dispatch a validated intent dict to its handler."""
    if operation not in OPERATIONS:
        raise ValueError(f"unknown operation: {operation}")
    return OPERATIONS[operation](df, **kwargs)
