from unittest.mock import MagicMock, patch

import httpx
import pytest
from groq import BadRequestError, RateLimitError

from app import llm


def _mock_groq_returning(content: str):
    """Patch app.llm.Groq so extract_intent() sees `content` as the model's raw reply."""
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    client = MagicMock()
    client.chat.completions.create.return_value = response
    return patch("app.llm.Groq", return_value=client)


@pytest.fixture(autouse=True)
def fake_api_key(monkeypatch, request):
    """Every mocked test gets a fake key; live-marked tests keep the real one from .env."""
    if "live" in request.keywords:
        return
    monkeypatch.setattr(llm.config, "GROQ_API_KEY", "test-key")


# --- happy path + validation/normalization ---

def test_valid_intent_passes_through():
    with _mock_groq_returning('{"operation": "count", "filters": {"priority": "Critical"}}'):
        intent = llm.extract_intent("How many critical tickets?")
    assert intent == {"operation": "count", "filters": {"priority": "Critical"}}


def test_case_insensitive_enum_value_is_normalized():
    # live-observed: the model returns lowercase values even when told exact casing
    with _mock_groq_returning('{"operation": "count", "filters": {"priority": "critical", "status": "open"}}'):
        intent = llm.extract_intent("How many critical open tickets?")
    assert intent["filters"] == {"priority": "Critical", "status": "Open"}


def test_valid_group_count_intent_passes_through():
    with _mock_groq_returning('{"operation": "group_count", "group_by": "agent_id", "filters": {}}'):
        intent = llm.extract_intent("Which agent resolved the most tickets?")
    assert intent == {"operation": "group_count", "filters": {}, "group_by": "agent_id", "order": "desc"}


def test_group_count_field_ranking_intent_passes_through():
    with _mock_groq_returning(
        '{"operation": "group_count", "group_by": "category", "field": "resolution_time_hrs", "agg": "average", "order": "desc", "top_n": 5}'
    ):
        intent = llm.extract_intent("5 categories with the highest average resolution time?")
    assert intent["field"] == "resolution_time_hrs"
    assert intent["agg"] == "average"
    assert intent["top_n"] == 5


def test_group_count_invalid_field_becomes_error():
    with _mock_groq_returning('{"operation": "group_count", "group_by": "category", "field": "agent_id"}'):
        intent = llm.extract_intent("rank categories by agent_id?")
    assert intent["operation"] == "error"


def test_valid_search_intent_passes_through():
    with _mock_groq_returning('{"operation": "search", "keyword": "login failure"}'):
        intent = llm.extract_intent("Find tickets mentioning login failure")
    assert intent == {"operation": "search", "filters": {}, "keyword": "login failure", "limit": 50}


def test_search_missing_keyword_becomes_error():
    with _mock_groq_returning('{"operation": "search"}'):
        intent = llm.extract_intent("search for something")
    assert intent["operation"] == "error"


def test_valid_average_intent_passes_through():
    with _mock_groq_returning('{"operation": "average", "field": "customer_rating", "filters": {}}'):
        intent = llm.extract_intent("Average rating?")
    assert intent == {"operation": "average", "filters": {}, "field": "customer_rating"}


def test_valid_filter_intent_gets_default_limit():
    with _mock_groq_returning('{"operation": "filter", "filters": {}}'):
        intent = llm.extract_intent("Show me tickets")
    assert intent["limit"] == 50


def test_non_enum_column_filter_passes_through_unchanged():
    # agent_id has no fixed enum set — any value is passed through as-is.
    with _mock_groq_returning('{"operation": "count", "filters": {"agent_id": "AGT-04"}}'):
        intent = llm.extract_intent("How many tickets for AGT-04?")
    assert intent["filters"] == {"agent_id": "AGT-04"}


def test_valid_equalize_intent_passes_through():
    with _mock_groq_returning('{"operation": "equalize", "group_by": "status"}'):
        intent = llm.extract_intent("How many tickets must move to balance each status?")
    assert intent == {"operation": "equalize", "filters": {}, "group_by": "status"}


def test_valid_ratio_intent_passes_through():
    with _mock_groq_returning(
        '{"operation": "ratio", "field": "response_time_hrs", "field_b": "resolution_time_hrs", "agg": "sum"}'
    ):
        intent = llm.extract_intent("Total response and resolution time, and their ratio?")
    assert intent == {
        "operation": "ratio", "filters": {},
        "field": "response_time_hrs", "field_b": "resolution_time_hrs", "agg": "sum",
    }


def test_ratio_invalid_field_becomes_error():
    with _mock_groq_returning('{"operation": "ratio", "field": "response_time_hrs", "field_b": "agent_id"}'):
        intent = llm.extract_intent("Ratio of response time to agent id?")
    assert intent["operation"] == "error"


def test_equalize_invalid_group_by_becomes_error():
    with _mock_groq_returning('{"operation": "equalize", "group_by": "issue_summary"}'):
        intent = llm.extract_intent("Balance issue_summary?")
    assert intent["operation"] == "error"


def test_anomaly_summary_passthrough():
    with _mock_groq_returning('{"operation": "anomaly_summary"}'):
        intent = llm.extract_intent("Any anomalies?")
    assert intent == {"operation": "anomaly_summary"}


# --- bad LLM output: dropped/noted, not crashed ---

def test_unknown_enum_value_is_dropped_and_noted():
    with _mock_groq_returning('{"operation": "count", "filters": {"priority": "Urgent"}}'):
        intent = llm.extract_intent("...")
    assert intent["filters"] == {}
    assert "priority" in intent["dropped_filters"][0]


def test_unknown_filter_column_is_dropped_and_noted():
    with _mock_groq_returning('{"operation": "count", "filters": {"made_up_column": "x"}}'):
        intent = llm.extract_intent("...")
    assert intent["filters"] == {}
    assert "made_up_column" in intent["dropped_filters"][0]


def test_unknown_operation_becomes_error():
    with _mock_groq_returning('{"operation": "delete_everything"}'):
        intent = llm.extract_intent("...")
    assert intent["operation"] == "error"


def test_invalid_group_by_becomes_error():
    with _mock_groq_returning('{"operation": "group_count", "group_by": "issue_summary"}'):
        intent = llm.extract_intent("...")
    assert intent["operation"] == "error"


def test_invalid_numeric_field_becomes_error():
    with _mock_groq_returning('{"operation": "average", "field": "agent_id"}'):
        intent = llm.extract_intent("...")
    assert intent["operation"] == "error"


def test_model_flagged_error_keeps_its_reason():
    # The model itself recognizing an out-of-scope question ("who are you?")
    # and saying so should reach the user, not get overwritten.
    with _mock_groq_returning('{"operation": "error", "reason": "This system only answers ticket questions."}'):
        intent = llm.extract_intent("who are you?")
    assert intent == {"operation": "error", "reason": "This system only answers ticket questions."}


def test_model_flagged_error_with_missing_reason_gets_a_fallback():
    with _mock_groq_returning('{"operation": "error"}'):
        intent = llm.extract_intent("who are you?")
    assert intent["operation"] == "error"
    assert intent["reason"]  # non-empty, not None/missing


def test_malformed_json_becomes_error_not_exception():
    with _mock_groq_returning("this is not json"):
        intent = llm.extract_intent("...")
    assert intent["operation"] == "error"


def test_json_array_instead_of_object_becomes_error():
    with _mock_groq_returning("[1, 2, 3]"):
        intent = llm.extract_intent("...")
    assert intent["operation"] == "error"


# --- infra failures degrade gracefully, never raise ---

def test_rate_limit_error_gets_a_distinct_retry_message():
    # Free-tier Groq limits are real — a 429 should tell the user to retry,
    # not dump a raw exception like other failures do.
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com/x"))
    rate_limit_error = RateLimitError("rate limited", response=response, body=None)
    with patch("app.llm.Groq", side_effect=rate_limit_error):
        intent = llm.extract_intent("...")
    assert intent["operation"] == "error"
    assert "rate limit" in intent["reason"].lower()


def test_bad_request_error_gets_a_clean_message_not_a_raw_dump():
    # Reproduces the real bug: asking something off-topic ("who are you?")
    # made the model answer in prose, which Groq's JSON-mode validator
    # rejects with a 400 — the raw Groq error dict was leaking to the user.
    response = httpx.Response(400, request=httpx.Request("POST", "https://api.groq.com/x"))
    bad_request_error = BadRequestError("json_validate_failed", response=response, body=None)
    with patch("app.llm.Groq", side_effect=bad_request_error):
        intent = llm.extract_intent("who are you?")
    assert intent["operation"] == "error"
    assert "json_validate_failed" not in intent["reason"]


def test_bad_request_error_message_does_not_claim_off_topic():
    # Live-found bug: this exception also fires for on-topic but unsupported
    # compound questions ("sum of X and Y, and their ratio") — the message
    # must not tell the user their question isn't about the dataset when it
    # might well be.
    response = httpx.Response(400, request=httpx.Request("POST", "https://api.groq.com/x"))
    bad_request_error = BadRequestError("json_validate_failed", response=response, body=None)
    with patch("app.llm.Groq", side_effect=bad_request_error):
        intent = llm.extract_intent("sum of response_time_hrs and resolution_time_hrs, and their ratio")
    assert intent["operation"] == "error"
    assert "only answers questions about" not in intent["reason"]
    assert "not about the ticket dataset" not in intent["reason"]


def test_groq_exception_becomes_error_not_crash():
    with patch("app.llm.Groq", side_effect=RuntimeError("network down")):
        intent = llm.extract_intent("...")
    assert intent["operation"] == "error"
    assert "network down" in intent["reason"]


def test_missing_api_key_short_circuits_without_calling_groq(monkeypatch):
    monkeypatch.setattr(llm.config, "GROQ_API_KEY", "")
    with patch("app.llm.Groq") as mock_groq_class:
        intent = llm.extract_intent("...")
    mock_groq_class.assert_not_called()
    assert intent["operation"] == "error"


# --- live smoke test against the real Groq API, skipped by default (pytest -m live) ---

@pytest.mark.live
def test_live_groq_extracts_valid_structured_intent():
    intent = llm.extract_intent("How many critical tickets are unresolved?")
    assert intent["operation"] == "count"
    assert intent["filters"].get("priority") == "Critical"


@pytest.mark.live
def test_live_off_topic_question_degrades_gracefully():
    # Reproduces the exact user-reported bug: "who are you?" used to raise a
    # raw Groq 400 (json_validate_failed) instead of a clean error.
    intent = llm.extract_intent("who are you?")
    assert intent["operation"] == "error"
    assert "json_validate_failed" not in intent["reason"]
    assert "Error code" not in intent["reason"]


@pytest.mark.live
def test_live_hypothetical_scenario_question_returns_error_not_a_wrong_answer():
    # Reproduces the exact user-reported bug: a "what if tickets moved between
    # statuses" question used to get silently mapped to group_count and
    # answered with the real (irrelevant) current distribution instead of
    # being recognized as unanswerable by these operations.
    question = (
        "what if the escalated tickets are resolved and resolved ones are "
        "returned to open and open are put on escalated, what would the new "
        "percentage distribution be"
    )
    intent = llm.extract_intent(question)
    assert intent["operation"] == "error"


# "Known-answer" questions the QA plan names explicitly — real Groq call,
# asserting the model picks the right operation/filters for each.

@pytest.mark.live
def test_live_swap_and_equalize_question_uses_equalize_not_error():
    # Reproduces the exact user-asked question: a swap-framed "hypothetical"
    # that's actually a well-defined equalize target — must route to the
    # real "equalize" operation, not get rejected as an unsimulable what-if.
    question = (
        "if the data between resolved and open tickets are swapped, what is "
        "the minimum percentage of exchange from each ticket status (open, "
        "escalated, resolved) that needs to be transferred to get an equal "
        "ticket percentage in each?"
    )
    intent = llm.extract_intent(question)
    assert intent["operation"] == "equalize"
    assert intent["group_by"] == "status"


@pytest.mark.live
def test_live_sum_and_ratio_question_uses_ratio_not_error():
    # Reproduces the exact user-asked question: a compound "sum of X and Y,
    # and their ratio" is on-topic, deterministic arithmetic on two real
    # columns — must route to "ratio", not the generic can't-process error.
    question = "what is the total sum of response time hrs and resolution time hrs and also give there ratio"
    intent = llm.extract_intent(question)
    assert intent["operation"] == "ratio"
    assert {intent["field"], intent["field_b"]} == {"response_time_hrs", "resolution_time_hrs"}


@pytest.mark.live
def test_live_group_count_question_picks_agent_id():
    intent = llm.extract_intent("Which agent resolved the most tickets?")
    assert intent["operation"] == "group_count"
    assert intent["group_by"] == "agent_id"


@pytest.mark.live
def test_live_average_rating_question_picks_correct_field_and_filter():
    intent = llm.extract_intent("What is the average rating for Technical issues?")
    assert intent["operation"] == "average"
    assert intent["field"] == "customer_rating"
    assert intent["filters"].get("category") == "Technical"
