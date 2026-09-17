import json
from pathlib import Path

import pytest

from app.ingestion import load_tickets
from app import query_engine as qe

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "support_tickets.csv"


@pytest.fixture(scope="module")
def df():
    return load_tickets(CSV_PATH)


# --- ground-truth numbers, verified against the real CSV directly ---

def test_count_critical_unresolved(df):
    result = qe.count(df, filters={"priority": "Critical", "status": ["Open", "Escalated"]})
    assert result["data"]["count"] == 31


def test_average_technical_rating(df):
    result = qe.average(df, field="customer_rating", filters={"category": "Technical"})
    assert result["data"]["average"] == 3.74
    assert result["data"]["matched"] == 104


def test_group_count_top_agent_resolved(df):
    result = qe.group_count(df, group_by="agent_id", filters={"status": "Resolved"})
    assert result["data"]["top"] == "AGT-09"


# --- edge cases ---

def test_filter_zero_matches_returns_zero_not_error(df):
    result = qe.count(df, filters={"priority": "Nonexistent"})
    assert result["data"]["count"] == 0


def test_average_on_empty_filtered_set_returns_none_not_nan(df):
    result = qe.average(df, field="customer_rating", filters={"priority": "Nonexistent"})
    assert result["data"]["average"] is None
    assert result["data"]["matched"] == 0


def test_average_excludes_nulls_not_zero(df):
    # every Open/Escalated ticket has a null customer_rating; only Resolved
    # tickets should count toward the average.
    result = qe.average(df, field="customer_rating", filters={"status": "Resolved"})
    assert result["data"]["matched"] == 327


def test_unknown_filter_column_is_reported_not_silently_dropped(df):
    result = qe.count(df, filters={"not_a_column": "x"})
    assert result["data"]["count"] == 500  # unmatched filter is ignored for matching...
    assert result["unmatched_filters"] == ["not_a_column"]  # ...but surfaced in the response


def test_unknown_operation_raises(df):
    with pytest.raises(ValueError, match="unknown operation"):
        qe.run(df, "delete_everything")


def test_non_numeric_field_rejected(df):
    with pytest.raises(ValueError, match="not numeric"):
        qe.average(df, field="agent_id")


def test_group_count_unknown_column_raises(df):
    with pytest.raises(ValueError, match="unknown group_by column"):
        qe.group_count(df, group_by="not_a_column")


def test_group_count_empty_group_has_no_top(df):
    result = qe.group_count(df, group_by="agent_id", filters={"priority": "Nonexistent"})
    assert result["data"]["top"] is None
    assert result["data"]["counts"] == {}


def test_filter_rows_are_json_serializable(df):
    # response includes real rows (with NaNs and a Timestamp) headed to an
    # HTTP JSON response — must round-trip through json.dumps cleanly.
    result = qe.filter_rows(df, filters={"status": "Open"}, limit=5)
    json.dumps(result)  # raises if anything (NaN, Timestamp, numpy scalar) leaks through
    assert result["data"]["tickets"][0]["customer_rating"] is None


def test_group_count_values_are_json_serializable(df):
    result = qe.group_count(df, group_by="agent_id")
    json.dumps(result)


def test_comparison_filter_gt(df):
    result = qe.count(df, filters={"resolution_time_hrs": {"gt": 12}})
    expected = (df["resolution_time_hrs"] > 12).sum()
    assert result["data"]["count"] == expected
    assert expected > 0  # sanity: the comparison actually filters something


def test_comparison_filter_excludes_unresolved_nulls(df):
    # resolution_time_hrs is null for unresolved tickets; a > comparison on a
    # nullable column must not match NaN as "greater than" anything.
    result = qe.count(df, filters={"resolution_time_hrs": {"gt": -1}})
    assert result["data"]["count"] == df["resolution_time_hrs"].notna().sum()


def test_comparison_filter_unsupported_op_is_reported(df):
    result = qe.count(df, filters={"resolution_time_hrs": {"unknown_op": 5}})
    assert result["unmatched_filters"] == ["resolution_time_hrs (unsupported comparison 'unknown_op')"]


def test_sum(df):
    result = qe.total(df, field="response_time_hrs")
    assert result["data"]["sum"] == 1310.3
    assert result["data"]["matched"] == 500


def test_min(df):
    result = qe.minimum(df, field="resolution_time_hrs")
    assert result["data"]["min"] == 1.2


def test_max(df):
    result = qe.maximum(df, field="resolution_time_hrs")
    assert result["data"]["max"] == 119.7


def test_rows_used_present_on_every_operation(df):
    # rows_used is the evidence trail: how many real DataFrame rows backed
    # the answer. Every operation must report it, not just the ones with an
    # obvious "count".
    assert qe.count(df, filters={"status": "Open"})["rows_used"] == 111
    assert qe.filter_rows(df, filters={"status": "Open"}, limit=5)["rows_used"] == 111
    assert qe.group_count(df, group_by="agent_id")["rows_used"] == 500
    assert qe.average(df, field="customer_rating")["rows_used"] == 327  # nulls excluded
    assert qe.total(df, field="response_time_hrs")["rows_used"] == 500
    assert qe.minimum(df, field="resolution_time_hrs")["rows_used"] == 327
    assert qe.maximum(df, field="resolution_time_hrs")["rows_used"] == 327


def test_rows_used_zero_on_empty_average(df):
    result = qe.average(df, field="customer_rating", filters={"priority": "Nonexistent"})
    assert result["rows_used"] == 0


def test_sum_min_max_on_empty_filtered_set_dont_crash(df):
    empty_filters = {"priority": "Nonexistent"}
    assert qe.total(df, field="response_time_hrs", filters=empty_filters)["data"]["sum"] == 0
    assert qe.minimum(df, field="response_time_hrs", filters=empty_filters)["data"]["min"] is None
    assert qe.maximum(df, field="response_time_hrs", filters=empty_filters)["data"]["max"] is None  # numpy.int64 counts must be converted to native int


# --- equalize: "minimum tickets to move to balance a column's categories" (docs/TRD.md §15/§16) ---

def test_equalize_status_ground_truth(df):
    # real counts: Resolved 327, Open 111, Escalated 62 of 500; target 166.67 each
    result = qe.equalize(df, group_by="status")
    assert result["data"]["target_per_category"] == 166.67
    assert result["data"]["total_moved"] == 160.33
    assert result["data"]["moves"]["Resolved"]["must_give_away"] == 160.33
    assert result["data"]["moves"]["Open"]["must_receive"] == 55.67
    assert result["data"]["moves"]["Escalated"]["must_receive"] == 104.67


def test_equalize_only_surplus_categories_give_away(df):
    result = qe.equalize(df, group_by="status")
    assert result["data"]["moves"]["Open"]["must_give_away"] == 0
    assert result["data"]["moves"]["Escalated"]["must_give_away"] == 0


def test_equalize_pct_of_category_matches_manual_math(df):
    # 160.33 / 327 current Resolved tickets = 49.0%
    result = qe.equalize(df, group_by="status")
    assert result["data"]["moves"]["Resolved"]["pct_of_category_to_move"] == 49.0


def test_equalize_swap_does_not_change_total_moved(df):
    # swapping which label owns which count is symmetric for equalization —
    # the minimum-transfer total is the same regardless of which two
    # categories were swapped beforehand (docs/TRD.md §16).
    swapped = df.copy()
    swapped["status"] = swapped["status"].replace({"Open": "Resolved", "Resolved": "Open"})
    result = qe.equalize(swapped, group_by="status")
    assert result["data"]["total_moved"] == 160.33


def test_equalize_respects_filters(df):
    result = qe.equalize(df, group_by="priority", filters={"status": "Resolved"})
    assert result["rows_used"] == 327


def test_equalize_other_column_priority(df):
    result = qe.equalize(df, group_by="priority")
    assert result["data"]["target_per_category"] == 125.0
    assert result["data"]["total_moved"] == 70.0


def test_equalize_unknown_column_raises(df):
    with pytest.raises(ValueError, match="unknown group_by column"):
        qe.equalize(df, group_by="not_a_column")


def test_equalize_empty_group_reports_zero_not_error(df):
    result = qe.equalize(df, group_by="status", filters={"priority": "Nonexistent"})
    assert result["data"]["total_moved"] == 0
    assert result["data"]["moves"] == {}


def test_equalize_rows_used_is_filtered_total(df):
    assert qe.equalize(df, group_by="status")["rows_used"] == 500


def test_equalize_values_are_json_serializable(df):
    result = qe.equalize(df, group_by="status")
    json.dumps(result)  # numpy int64/float64 from value_counts() must not leak through


def test_run_dispatches_to_equalize(df):
    result = qe.run(df, "equalize", group_by="status")
    assert result["data"]["total_moved"] == 160.33
