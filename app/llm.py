"""NL question -> Groq -> validated structured intent (docs/TRD.md §8).

The LLM only picks an operation + filters; it never computes numbers, so a
bad LLM response can pick the wrong operation but can never report a wrong
number for the right one. This function never raises — any failure (missing
key, network, timeout, bad JSON, invalid fields) degrades to
{"operation": "error", "reason": ...} so the API layer can respond 200.
"""
import json

from groq import BadRequestError, Groq, RateLimitError

from app import config
from app.anomaly import ENUM_VALUES
from app.ingestion import REQUIRED_COLUMNS
from app.query_engine import NUMERIC_FIELDS

OPERATIONS = ["count", "filter", "group_count", "average", "sum", "min", "max", "anomaly_summary", "equalize"]
GROUPABLE_COLUMNS = ["agent_id", "category", "priority", "status"]


def _build_system_prompt() -> str:
    """Embed the real schema/enum values so the prompt can't drift from the data."""
    enum_lines = "\n".join(f"- {col}: {sorted(values)}" for col, values in ENUM_VALUES.items())
    return (
        "You convert a support-ticket analytics question into ONE JSON object "
        "describing how to compute the answer. Never compute the answer yourself.\n\n"
        f"Allowed \"operation\" values: {OPERATIONS}\n"
        f"Allowed enum columns and exact values:\n{enum_lines}\n"
        f"Numeric fields (for average/sum/min/max): {NUMERIC_FIELDS}\n"
        f"Groupable columns (for group_count): {GROUPABLE_COLUMNS}\n\n"
        "Respond with JSON only, matching one of:\n"
        '{"operation": "count", "filters": {...}}\n'
        '{"operation": "filter", "filters": {...}, "limit": 50}\n'
        '{"operation": "group_count", "group_by": "<column>", "filters": {...}}\n'
        '{"operation": "average"|"sum"|"min"|"max", "field": "<numeric field>", "filters": {...}}\n'
        '{"operation": "anomaly_summary"}\n'
        '{"operation": "equalize", "group_by": "<column>"}\n'
        '{"operation": "error", "reason": "<why this question can\'t be answered>"}\n\n'
        '"filters" maps column -> value, column -> [values] (isin), or for a '
        'numeric field column -> {"gt"|"gte"|"lt"|"lte": number} (e.g. '
        '{"resolution_time_hrs": {"gt": 12}} for "resolution over 12 hours"). '
        "Omit filters that don't apply. Use the exact column names and enum "
        "casing given above. A resolution_time_hrs/customer_rating comparison "
        "only matches already-resolved tickets (those fields are null until "
        "resolved) — say so in your answer if the question is about unresolved "
        "tickets aging past a threshold; prefer \"anomaly_summary\" for that case.\n\n"
        "If the question is not about this ticket dataset (small talk, asking who "
        "you are, anything unrelated to tickets/agents/priority/status/ratings/"
        "resolution time), you MUST still respond with the JSON error form above — "
        "never prose, never an apology, never plain text. Always JSON, no exceptions.\n\n"
        "CHECK THIS FIRST, before anything below: does the question ask to "
        "equalize/balance/even out a column's categories, or ask the minimum/"
        "smallest number or percentage of tickets that must move or be "
        "transferred between categories so each ends up equal? If yes, "
        "ALWAYS use {\"operation\": \"equalize\", \"group_by\": \"<column>\"} "
        "— even if the question also talks about first swapping, rotating, or "
        "moving categories around. Ignore that swap/rotate framing completely: "
        "equalizing real current counts to total/#categories is plain "
        "arithmetic, not a simulation, and swapping which label owns which "
        "count never changes that arithmetic. Do NOT use the error form for "
        "these questions just because they mention \"swap\" or \"if\".\n\n"
        "Only if the question is NOT asking to equalize/balance, but instead "
        "describes an unrelated what-if (e.g. \"if X tickets became Y status, "
        "what would the new counts/percentages be\") and expects a new "
        "distribution back: these operations only read the real, current "
        "data and cannot simulate that, so respond with the JSON error form, "
        "reason: this system only answers questions about the actual current "
        "data, it can't simulate hypothetical changes."
    )


_SYSTEM_PROMPT = _build_system_prompt()


def _canonical_enum_value(column: str, value):
    """Case-insensitive match to the real enum casing; None if truly unknown.

    Live-tested finding: the model returns lowercase values ("critical")
    even when told the exact casing, so this can't be a strict equality check.
    """
    if column not in ENUM_VALUES:
        return value  # not an enum column (e.g. agent_id) — pass through
    for canonical in ENUM_VALUES[column]:
        if str(value).lower() == canonical.lower():
            return canonical
    return None


def _validate_filters(filters) -> tuple[dict, list[str]]:
    """Keep only known columns/values; report the rest instead of silently dropping them."""
    clean, dropped = {}, []
    for column, value in (filters or {}).items():
        if column not in REQUIRED_COLUMNS:
            dropped.append(f"unknown column '{column}'")
            continue
        was_list = isinstance(value, list)
        resolved = []
        for v in value if was_list else [value]:
            canonical = _canonical_enum_value(column, v)
            if canonical is None:
                dropped.append(f"unknown value '{v}' for {column}")
            else:
                resolved.append(canonical)
        if resolved:
            clean[column] = resolved if was_list else resolved[0]
    return clean, dropped


def _validate_intent(raw: dict) -> dict:
    """Turn a parsed-but-untrusted LLM JSON object into a safe query_engine call."""
    operation = raw.get("operation")

    if operation == "error":
        # The model itself flagged an out-of-scope/unanswerable question — keep its
        # reason if it gave a real string, otherwise fall back to a generic one.
        reason = raw.get("reason")
        reason = reason if isinstance(reason, str) and reason.strip() else "This question isn't about the ticket dataset."
        return {"operation": "error", "reason": reason}

    if operation not in OPERATIONS:
        return {"operation": "error", "reason": f"unknown operation from LLM: {operation!r}"}

    if operation == "anomaly_summary":
        return {"operation": "anomaly_summary"}

    filters, dropped = _validate_filters(raw.get("filters"))
    intent = {"operation": operation, "filters": filters}
    if dropped:
        intent["dropped_filters"] = dropped

    if operation in ("group_count", "equalize"):
        group_by = raw.get("group_by")
        if group_by not in GROUPABLE_COLUMNS:
            return {"operation": "error", "reason": f"invalid group_by from LLM: {group_by!r}"}
        intent["group_by"] = group_by

    if operation in ("average", "sum", "min", "max"):
        field = raw.get("field")
        if field not in NUMERIC_FIELDS:
            return {"operation": "error", "reason": f"invalid field from LLM: {field!r}"}
        intent["field"] = field

    if operation == "filter":
        intent["limit"] = raw.get("limit") if isinstance(raw.get("limit"), int) else 50

    return intent


def extract_intent(question: str) -> dict:
    """Question -> validated intent dict, or an {"operation": "error"} dict. Never raises."""
    if not config.GROQ_API_KEY:
        return {"operation": "error", "reason": "GROQ_API_KEY not configured"}

    try:
        # timeout + max_retries are the SDK's own handling — no hand-rolled retry loop needed.
        client = Groq(api_key=config.GROQ_API_KEY, timeout=5.0, max_retries=1)
        response = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"operation": "error", "reason": "LLM returned invalid JSON"}
    except RateLimitError:
        # Distinguished from a generic failure: this one is worth telling the user to retry.
        return {"operation": "error", "reason": "Groq free-tier rate limit hit — wait a moment and try again"}
    except BadRequestError:
        # Groq's own JSON-mode validator rejected the generation (e.g. the model
        # tried to answer an off-topic question in prose instead of JSON). The
        # system prompt now tells it to emit a JSON error instead — this is the
        # defensive fallback for when that steering still doesn't hold.
        return {"operation": "error", "reason": "This system only answers questions about the support ticket dataset."}
    except Exception as exc:  # noqa: BLE001 - any other Groq/network failure must degrade gracefully, not crash the API
        return {"operation": "error", "reason": f"Groq request failed: {exc}"}

    if not isinstance(raw, dict):
        return {"operation": "error", "reason": "LLM JSON was not an object"}

    return _validate_intent(raw)
