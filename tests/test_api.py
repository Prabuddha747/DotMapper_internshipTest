from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app


def test_health_reports_data_loaded():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["tickets_loaded"] == 500


def test_no_endpoint_ever_leaks_the_raw_api_key():
    if not config.GROQ_API_KEY:
        pytest.skip("no GROQ_API_KEY configured in this environment")
    with TestClient(app) as client:
        with patch("app.main.llm.extract_intent", return_value={"operation": "count", "filters": {}}):
            responses = [
                client.get("/health"),
                client.get("/api/stats"),
                client.get("/api/anomalies"),
                client.post("/api/query", json={"question": "How many tickets are open?"}),
            ]
    for response in responses:
        assert config.GROQ_API_KEY not in response.text


def test_query_empty_question_returns_422():
    with TestClient(app) as client:
        response = client.post("/api/query", json={"question": "   "})
    assert response.status_code == 422


def test_query_missing_field_returns_422():
    with TestClient(app) as client:
        response = client.post("/api/query", json={})
    assert response.status_code == 422


def test_query_llm_error_returns_200_not_500():
    with TestClient(app) as client:
        with patch("app.main.llm.extract_intent", return_value={"operation": "error", "reason": "boom"}):
            response = client.post("/api/query", json={"question": "anything"})
    assert response.status_code == 200
    assert response.json()["operation"] == "error"


def test_query_valid_intent_returns_real_computed_answer():
    with TestClient(app) as client:
        with patch("app.main.llm.extract_intent", return_value={"operation": "count", "filters": {"status": "Open"}}):
            response = client.post("/api/query", json={"question": "How many open tickets?"})
    assert response.status_code == 200
    assert response.json()["data"]["count"] == 111  # verified real count, see docs/TRD.md


def test_query_response_carries_evidence_envelope():
    # source/rows_used/timestamp prove the answer came from a real pandas
    # computation, not the LLM — see docs/TRD.md §13.
    with TestClient(app) as client:
        with patch("app.main.llm.extract_intent", return_value={"operation": "count", "filters": {"status": "Open"}}):
            response = client.post("/api/query", json={"question": "How many open tickets?"})
    body = response.json()
    assert body["source"] == "in-memory pandas DataFrame (data/support_tickets.csv)"
    assert body["rows_used"] == 111
    assert body["timestamp"]  # ISO datetime string, exact value is not worth pinning


def test_query_error_response_still_carries_source_and_timestamp():
    with TestClient(app) as client:
        with patch("app.main.llm.extract_intent", return_value={"operation": "error", "reason": "boom"}):
            response = client.post("/api/query", json={"question": "..."})
    body = response.json()
    assert body["source"] and body["timestamp"]


def test_query_engine_error_returns_200_not_500():
    # Simulates an intent that slipped past llm.py's own validation (defense in depth).
    bad_intent = {"operation": "group_count", "group_by": "not_a_real_column", "filters": {}}
    with TestClient(app) as client:
        with patch("app.main.llm.extract_intent", return_value=bad_intent):
            response = client.post("/api/query", json={"question": "..."})
    assert response.status_code == 200
    assert response.json()["operation"] == "error"


def test_anomalies_endpoint_matches_verified_counts():
    with TestClient(app) as client:
        response = client.get("/api/anomalies")
    assert response.status_code == 200
    assert len(response.json()["anomalies"]) == 80 + 21  # Rule A + Rule B; Rule C is clean on real data


def test_stats_endpoint_shape():
    with TestClient(app) as client:
        response = client.get("/api/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["total_tickets"] == 500
    assert body["by_status"]["Resolved"] == 327
    assert body["anomaly_count"] == 80 + 21
