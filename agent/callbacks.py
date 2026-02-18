"""
Before- and after-model callbacks for the logistics agent.

- before_model_callback: Intercepts user input to block sensitive requests (e.g. delete all
  agreements, user list, agreement data) and returns safe responses or asks user to contact admin.
- after_model_callback: Sanitizes the model response so no sensitive information is exposed.

Blocked / redirected query types (user is directed to admin@nav-it.com where applicable):
  1. Delete/remove agreements from SAP  → blocked (no bulk delete)
  2. SAP user list or user details       → blocked (no user list/details)
  3. Data for a particular agreement ID → redirected to admin
"""

# Human-readable list of blocked query types (for docs and logging)
BLOCKED_QUERY_TYPES = [
    "Delete or remove (all) agreement(s) from SAP",
    "SAP user list, user details, or user info",
    "Data for / across / about a particular agreement (ID)",
]

import logging
import re
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sensitive-query detection (before_model_callback)
# ---------------------------------------------------------------------------

# Phrases that indicate request to delete agreements from SAP (block and respond)
DELETE_AGREEMENT_PATTERNS = [
    r"delete\s+(all\s+)?(the\s+)?agreement(s|id)?\s*(from\s+)?sap",
    r"remove\s+(all\s+)?(the\s+)?agreement(s|id)?\s*(from\s+)?sap",
    r"delete\s+.*agreement.*sap",
    r"remove\s+.*agreement.*sap",
]

# Phrases that indicate request for SAP user list/details (block and respond with admin email)
USER_LIST_PATTERNS = [
    r"user\s+list\s*(for\s+)?(the\s+)?sap",
    r"(give|get|show|list)\s+(me\s+)?(the\s+)?user(s)?\s*(list)?\s*(for\s+)?(the\s+)?sap",
    r"sap\s+user\s+list",
    r"sap\s+user\s+details",
    r"user\s+details\s*(for\s+)?(the\s+)?sap",
    r"(give|get|show|list)\s+(me\s+)?(the\s+)?sap\s+user",
    r"sap\s+user\s+(list|details|info)",
]

# Phrases that indicate request for data for a particular agreement ID (redirect to admin)
AGREEMENT_DATA_PATTERNS = [
    r"(give|get|show|fetch|provide)\s+(me\s+)?(the\s+)?data\s+(for|across|about)\s+(a\s+)?(particular\s+)?agreement\s*(\s+id)?",
    r"data\s+(for|across|about)\s+(a\s+)?(particular\s+)?agreement\s*(\s+id)?",
    r"agreement\s+(id\s+)?data",
]

CONTACT_ADMIN_MESSAGE = (
    "For this type of request, please contact the administrator at admin@nav-it.com. "
    "They will be able to assist you with the data or actions you need."
)

BLOCK_DELETE_MESSAGE = (
    "I cannot perform bulk delete operations on agreements in SAP. "
    "If you need to remove agreements, please contact your administrator at admin@nav-it.com."
)

BLOCK_USER_LIST_MESSAGE = (
    "I cannot provide SAP user lists or user details. For user or access information, "
    "please contact your administrator at admin@nav-it.com."
)


def _get_last_user_message(llm_request: LlmRequest) -> str:
    """Extract the last user message text from the LLM request."""
    if not getattr(llm_request, "contents", None):
        return ""
    text_parts = []
    for content in reversed(llm_request.contents):
        role = getattr(content, "role", None)
        if role and str(role).lower() == "user":
            for part in getattr(content, "parts", []) or []:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)
            if text_parts:
                return " ".join(reversed(text_parts))
    return ""


def _matches_any(text: str, patterns: list[str]) -> bool:
    """Return True if the normalized user text matches any of the regex patterns."""
    normalized = " ".join(text.lower().split())
    for pat in patterns:
        if re.search(pat, normalized, re.IGNORECASE):
            return True
    return False


def _safe_log_preview(text: str, max_len: int = 80) -> str:
    """Return a safe preview of user/content for logging (truncated, no secrets)."""
    if not text:
        return ""
    cleaned = " ".join(text.split())[:max_len]
    return f"{cleaned}..." if len(text) > max_len else cleaned


def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[LlmResponse]:
    """
    Run before the model is called. Intercepts sensitive user intents and returns
    a safe response without calling the LLM (e.g. block deletes, user list, or
    redirect agreement-data requests to admin@nav-it.com).
    """
    agent_name = getattr(callback_context, "agent_name", "unknown")
    invocation_id = getattr(callback_context, "invocation_id", None)
    logger.info(
        "before_model_callback entered agent=%s invocation_id=%s",
        agent_name,
        invocation_id,
        extra={"agent": agent_name, "invocation_id": invocation_id},
    )

    last_message = _get_last_user_message(llm_request)
    if not last_message:
        logger.debug("before_model_callback: no user message in request, passing through")
        return None

    logger.debug(
        "before_model_callback: inspecting user message preview=%s",
        _safe_log_preview(last_message),
    )

    if _matches_any(last_message, DELETE_AGREEMENT_PATTERNS):
        logger.warning(
            "before_model_callback: blocked request to delete agreement(s) from SAP agent=%s invocation_id=%s",
            agent_name,
            invocation_id,
            extra={"agent": agent_name, "invocation_id": invocation_id},
        )
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=BLOCK_DELETE_MESSAGE)],
            )
        )

    if _matches_any(last_message, USER_LIST_PATTERNS):
        logger.warning(
            "before_model_callback: blocked request for SAP user list agent=%s invocation_id=%s",
            agent_name,
            invocation_id,
            extra={"agent": agent_name, "invocation_id": invocation_id},
        )
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=BLOCK_USER_LIST_MESSAGE)],
            )
        )

    if _matches_any(last_message, AGREEMENT_DATA_PATTERNS):
        logger.info(
            "before_model_callback: redirected agreement data request to admin@nav-it.com agent=%s invocation_id=%s",
            agent_name,
            invocation_id,
            extra={"agent": agent_name, "invocation_id": invocation_id},
        )
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=CONTACT_ADMIN_MESSAGE)],
            )
        )

    logger.debug("before_model_callback: no sensitive intent matched, passing to model")
    return None


# ---------------------------------------------------------------------------
# Response sanitization (after_model_callback)
# ---------------------------------------------------------------------------

# Patterns that may indicate sensitive data; we redact or replace.
SENSITIVE_PATTERNS = [
    # API keys / secrets
    (re.compile(r"\b(sk-[a-zA-Z0-9]{20,})\b", re.IGNORECASE), "[REDACTED_API_KEY]"),
    (re.compile(r"\b(api[_-]?key|apikey)\s*[:=]\s*['\"]?[^\s'\"]+['\"]?", re.IGNORECASE), "api_key=[REDACTED]"),
    (re.compile(r"\b(bearer)\s+[a-zA-Z0-9_\-.]+\b", re.IGNORECASE), "Bearer [REDACTED]"),
    # Internal paths that might reveal structure
    (re.compile(r"\b[A-Z]:\\[^\s]+\b"), "[REDACTED_PATH]"),
    (re.compile(r"/etc/[^\s]+"), "[REDACTED_PATH]"),
    (re.compile(r"/var/[^\s]+"), "[REDACTED_PATH]"),
]

# If the response looks like a raw stack trace or internal error, replace with generic message.
STACK_TRACE_STARTERS = ("Traceback (most recent call last):", "File \"", "  at ", "Exception:", "Error:")


def _sanitize_text(text: str, log_redactions: bool = True) -> tuple[str, list[str]]:
    """
    Redact sensitive patterns from model output.
    Returns (sanitized_text, list of redaction descriptions for logging).
    """
    if not text or not text.strip():
        return text, []
    out = text
    redactions: list[str] = []
    for pattern, replacement in SENSITIVE_PATTERNS:
        before_len = len(out)
        out = pattern.sub(replacement, out)
        if len(out) != before_len and log_redactions:
            redactions.append(replacement)
    # If it looks like a stack trace, don't expose it to the user
    lines = out.strip().split("\n")
    if len(lines) >= 2 and any(lines[0].strip().startswith(s) for s in STACK_TRACE_STARTERS):
        redactions.append("stack_trace_replaced")
        return (
            "An internal error occurred while processing your request. "
            "Please try again or contact admin@nav-it.com if the issue persists.",
            redactions,
        )
    return out, redactions


def after_model_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
    """
    Run after the model returns. Sanitizes the response so that sensitive
    information (API keys, paths, stack traces) is not exposed to the user.
    """
    agent_name = getattr(callback_context, "agent_name", "unknown")
    invocation_id = getattr(callback_context, "invocation_id", None)
    logger.info(
        "after_model_callback entered agent=%s invocation_id=%s",
        agent_name,
        invocation_id,
        extra={"agent": agent_name, "invocation_id": invocation_id},
    )

    if not llm_response or not getattr(llm_response, "content", None):
        logger.debug("after_model_callback: no content in response, passing through")
        return None

    content = llm_response.content
    parts = getattr(content, "parts", None) or []
    if not parts:
        logger.debug("after_model_callback: empty parts, passing through")
        return None

    # Don't modify function_call parts (tool calls)
    modified_parts = []
    changed = False
    all_redactions: list[str] = []
    for part in parts:
        if getattr(part, "function_call", None):
            modified_parts.append(part)
            continue
        if hasattr(part, "text") and part.text:
            sanitized, redactions = _sanitize_text(part.text, log_redactions=True)
            if sanitized != part.text:
                changed = True
                all_redactions.extend(redactions)
            modified_parts.append(types.Part(text=sanitized))
        else:
            modified_parts.append(part)

    if not changed:
        logger.debug("after_model_callback: no sanitization needed, passing through")
        return None

    logger.warning(
        "after_model_callback: sanitized response (sensitive content redacted) agent=%s invocation_id=%s redactions=%s",
        agent_name,
        invocation_id,
        all_redactions,
        extra={
            "agent": agent_name,
            "invocation_id": invocation_id,
            "redactions": all_redactions,
        },
    )
    return LlmResponse(
        content=types.Content(role="model", parts=modified_parts),
        grounding_metadata=getattr(llm_response, "grounding_metadata", None),
    )
