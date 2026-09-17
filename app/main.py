"""FastAPI app: wires ingestion, the query engine, anomaly detection, and
the Groq LLM layer together. See docs/flow.md for the request lifecycle.
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import anomaly, config, llm
from app import query_engine as qe
from app.ingestion import load_tickets
from app.models import QueryRequest

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "support_tickets.csv"
DATA_SOURCE = "in-memory pandas DataFrame (data/support_tickets.csv)"

# anomaly_summary needs the anomaly engine; wiring it here (instead of importing
# anomaly.py from inside query_engine.py) avoids a circular import between the two.
qe.OPERATIONS["anomaly_summary"] = anomaly.summary


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the CSV once into memory; app.state.df is the store for the process lifetime."""
    app.state.df = load_tickets(DATA_PATH)
    app.state.last_intent = None  # ponytail: single global slot, not per-client — fine for a local single-evaluator app; key by a client id if concurrent multi-user follow-ups are ever needed
    yield


app = FastAPI(title="Support Intelligence AI", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Service + data status, so a degraded /api/query (no Groq key) is visible at a glance."""
    df = getattr(app.state, "df", None)
    return {
        "status": "ok" if df is not None else "not_ready",
        "tickets_loaded": len(df) if df is not None else 0,
        "groq_configured": bool(config.GROQ_API_KEY),
    }


@app.post("/api/query")
def api_query(request: QueryRequest) -> dict:
    """NL question -> Groq intent -> deterministic query_engine execution.

    Every response carries an evidence trail (source, rows_used, timestamp) —
    the LLM only ever picked an operation + filters, so this envelope proves
    the answer text came from a real pandas computation, not the model.
    """
    base = {"question": request.question, "source": DATA_SOURCE, "timestamp": datetime.now(timezone.utc).isoformat()}
    intent = llm.extract_intent(request.question, previous=app.state.last_intent)
    operation = intent.pop("operation")

    if operation == "error":
        return {**base, "operation": "error", "answer": f"Couldn't process that question: {intent.get('reason')}"}

    try:
        result = qe.run(app.state.df, operation, **intent)
    except ValueError as exc:
        # Defensive: llm.py already validates fields/columns, but never turn
        # an unexpected data-shape issue into a raw 500.
        return {**base, "operation": "error", "answer": f"Couldn't compute that: {exc}"}

    app.state.last_intent = {"question": request.question, "intent": {"operation": operation, **intent}}
    return {**base, "operation": operation, **intent, **result}


@app.get("/api/anomalies")
def api_anomalies() -> dict:
    """All anomaly rule hits, tagged by rule. No LLM involved — fully deterministic."""
    return {"anomalies": anomaly.detect_anomalies(app.state.df)}


@app.get("/api/stats")
def api_stats() -> dict:
    """Dashboard summary for the UI's header cards."""
    df = app.state.df
    return {
        "total_tickets": len(df),
        "by_status": {k: int(v) for k, v in df["status"].value_counts().items()},
        "by_priority": {k: int(v) for k, v in df["priority"].value_counts().items()},
        "anomaly_count": len(anomaly.detect_anomalies(df)),
    }


# Mounted last: acts as a catch-all for "/" and static assets without
# shadowing the API routes above (Starlette matches routes in order).
app.mount("/", StaticFiles(directory=Path(__file__).resolve().parent.parent / "static", html=True), name="static")
