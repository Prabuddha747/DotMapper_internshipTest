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


def test_group_count_reports_percentage_share(df):
    # real counts: Resolved 327, Open 111, Escalated 62 of 500
    result = qe.group_count(df, group_by="status")
    assert result["data"]["percentages"] == {"Resolved": 65.4, "Open": 22.2, "Escalated": 12.4}


def test_group_count_ranks_by_field_average(df):
    result = qe.group_count(df, group_by="category", field="resolution_time_hrs", agg="average")
    assert result["data"]["top"] == "Technical"
    assert result["data"]["stats"]["Technical"] == 20.59


def test_group_count_field_ranking_respects_top_n(df):
    result = qe.group_count(df, group_by="category", field="resolution_time_hrs", top_n=1)
    assert len(result["data"]["stats"]) == 1


def test_group_count_field_ranking_ascending_order(df):
    # order="asc" must flip which category is "top" vs. desc
    desc = qe.group_count(df, group_by="category", field="resolution_time_hrs", order="desc")
    asc = qe.group_count(df, group_by="category", field="resolution_time_hrs", order="asc")
    assert desc["data"]["top"] != asc["data"]["top"]
    assert asc["data"]["top"] == "Billing"


def test_group_count_count_based_ascending_order(df):
    result = qe.group_count(df, group_by="status", order="asc")
    assert result["data"]["top"] == "Escalated"  # smallest count (62) of the three


def test_group_count_count_based_respects_top_n(df):
    result = qe.group_count(df, group_by="status", top_n=1)
    assert list(result["data"]["counts"].keys()) == ["Resolved"]


def test_group_count_field_ranking_invalid_field_raises(df):
    with pytest.raises(ValueError, match="not numeric"):
        qe.group_count(df, group_by="category", field="agent_id")


def test_group_count_field_ranking_values_are_json_serializable(df):
    result = qe.group_count(df, group_by="category", field="resolution_time_hrs")
    json.dumps(result)


# --- search: substring match on issue_summary ---

def test_search_finds_matching_tickets(df):
    result = qe.search(df, keyword="refund")
    assert result["data"]["count"] == 20
    assert all("refund" in t["issue_summary"].lower() for t in result["data"]["tickets"])


def test_search_is_case_insensitive(df):
    result = qe.search(df, keyword="REFUND")
    assert result["data"]["count"] == 20


def test_search_no_matches_returns_zero_not_error(df):
    result = qe.search(df, keyword="xyznonexistentkeyword")
    assert result["data"]["count"] == 0
    assert result["data"]["tickets"] == []


def test_search_empty_keyword_raises(df):
    with pytest.raises(ValueError, match="keyword"):
        qe.search(df, keyword="")


def test_search_respects_filters(df):
    result = qe.search(df, keyword="refund", filters={"status": "Resolved"})
    assert all(t["status"] == "Resolved" for t in result["data"]["tickets"])


def test_search_respects_limit(df):
    result = qe.search(df, keyword="refund", limit=2)
    assert len(result["data"]["tickets"]) == 2
    assert result["data"]["count"] == 20  # count is the real total match, limit only caps rows returned


def test_run_dispatches_to_search(df):
    result = qe.run(df, "search", keyword="refund")
    assert result["data"]["count"] == 20


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


def test_max_identifies_which_ticket(df):
    # a bare number is not a useful answer to "which ticket has the highest
    # resolution time" — the response must name the actual ticket.
    result = qe.maximum(df, field="resolution_time_hrs")
    assert result["data"]["ticket_id"] == "TKT-108"
    assert "TKT-108" in result["answer"]


def test_min_identifies_which_ticket(df):
    result = qe.minimum(df, field="resolution_time_hrs")
    assert result["data"]["ticket_id"] is not None
    assert result["data"]["ticket_id"] in result["answer"]


def test_min_max_ticket_id_none_on_empty_set(df):
    empty_filters = {"priority": "Nonexistent"}
    assert qe.minimum(df, field="resolution_time_hrs", filters=empty_filters)["data"]["ticket_id"] is None
    assert qe.maximum(df, field="resolution_time_hrs", filters=empty_filters)["data"]["ticket_id"] is None


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


# --- ratio: sum/average of two numeric fields plus their ratio ---

def test_ratio_sum_ground_truth(df):
    result = qe.ratio(df, field="response_time_hrs", field_b="resolution_time_hrs", agg="sum")
    assert result["data"]["value_a"] == 1310.3
    assert result["data"]["value_b"] == 6264.8
    assert result["data"]["ratio"] == 0.2092


def test_ratio_average(df):
    result = qe.ratio(df, field="response_time_hrs", field_b="resolution_time_hrs", agg="average")
    assert result["data"]["value_a"] == round(1310.3 / 500, 2)
    assert result["data"]["value_b"] == round(6264.8 / 327, 2)


def test_ratio_defaults_to_sum_for_unknown_agg(df):
    result = qe.ratio(df, field="response_time_hrs", field_b="resolution_time_hrs", agg="bogus")
    assert result["data"]["agg"] == "sum"
    assert result["data"]["value_a"] == 1310.3


def test_ratio_respects_filters(df):
    result = qe.ratio(df, field="response_time_hrs", field_b="resolution_time_hrs", filters={"status": "Resolved"})
    assert result["rows_used"] == 327


def test_ratio_non_numeric_field_rejected(df):
    with pytest.raises(ValueError, match="not numeric"):
        qe.ratio(df, field="agent_id", field_b="response_time_hrs")


def test_ratio_zero_denominator_is_undefined_not_a_crash(df):
    result = qe.ratio(df, field="response_time_hrs", field_b="customer_rating", filters={"priority": "Nonexistent"})
    assert result["data"]["ratio"] is None
    assert "undefined" in result["answer"]


def test_ratio_values_are_json_serializable(df):
    result = qe.ratio(df, field="response_time_hrs", field_b="resolution_time_hrs")
    json.dumps(result)


def test_run_dispatches_to_ratio(df):
    result = qe.run(df, "ratio", field="response_time_hrs", field_b="resolution_time_hrs")
    assert result["data"]["ratio"] == 0.2092
