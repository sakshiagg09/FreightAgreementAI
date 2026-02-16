"""
LLM token usage tracking per request.
Stores prompt_tokens, completion_tokens, and total_tokens consumed during one /invoke call.
Uses session_id-keyed storage so usage is visible across threads (ADK runner may run LLM in another thread).
"""
import threading
from typing import Dict, List, Any

# session_id -> { "total": {...}, "by_step": [...] }; shared across threads
_usage_store: Dict[str, Dict[str, Any]] = {}
# Usage recorded when session_id was unknown (e.g. LLM ran in another thread); merged into next get_llm_usage()
_orphan_usage: Dict[str, Any] = {"total": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "by_step": []}
_store_lock = threading.Lock()

_DEFAULT_TOTAL = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _with_input_output_aliases(usage: Dict[str, Any]) -> Dict[str, Any]:
    """Add input_tokens/output_tokens aliases so response supports both naming conventions."""
    out = dict(usage)
    out["input_tokens"] = out.get("prompt_tokens", 0)
    out["output_tokens"] = out.get("completion_tokens", 0)
    return out


def reset_llm_usage(session_id: str) -> None:
    """Reset token counts and steps for this session. Call once at the start of each /invoke."""
    with _store_lock:
        _usage_store[session_id] = {
            "total": dict(_DEFAULT_TOTAL),
            "by_step": [],
        }
        _orphan_usage["total"] = dict(_DEFAULT_TOTAL)
        _orphan_usage["by_step"] = []


def add_llm_usage(
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    tool_calls: List[str] | None = None,
    session_id: str | None = None,
) -> None:
    """Add token counts from one LLM call. Uses session_id from session_context if not provided."""
    if session_id is None:
        try:
            from utils.session_context import get_session_id
            session_id = get_session_id()
        except Exception:
            session_id = None
    # If session is "default" or missing (e.g. LLM in another thread), record as orphan and merge on get
    if not session_id or session_id == "default":
        session_id = None

    step = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    if tool_calls:
        step["tool_calls"] = tool_calls

    with _store_lock:
        if session_id:
            entry = _usage_store.get(session_id)
            if entry is None:
                entry = {"total": dict(_DEFAULT_TOTAL), "by_step": []}
                _usage_store[session_id] = entry
            entry["total"]["prompt_tokens"] = entry["total"].get("prompt_tokens", 0) + prompt_tokens
            entry["total"]["completion_tokens"] = entry["total"].get("completion_tokens", 0) + completion_tokens
            entry["total"]["total_tokens"] = entry["total"].get("total_tokens", 0) + total_tokens
            entry["by_step"].append(step)
        else:
            o = _orphan_usage
            o["total"]["prompt_tokens"] = o["total"].get("prompt_tokens", 0) + prompt_tokens
            o["total"]["completion_tokens"] = o["total"].get("completion_tokens", 0) + completion_tokens
            o["total"]["total_tokens"] = o["total"].get("total_tokens", 0) + total_tokens
            o["by_step"].append(step)


def get_llm_usage(session_id: str) -> Dict[str, Any]:
    """Return accumulated token usage and per-step breakdown for this session (invoke). Merges any orphan usage."""
    with _store_lock:
        entry = _usage_store.get(session_id)
        if entry is None:
            entry = {"total": dict(_DEFAULT_TOTAL), "by_step": []}
        total = dict(entry["total"])
        steps = list(entry["by_step"])
        # Merge orphan (usage recorded without session, e.g. from another thread)
        ot = _orphan_usage["total"]
        total["prompt_tokens"] = total.get("prompt_tokens", 0) + ot.get("prompt_tokens", 0)
        total["completion_tokens"] = total.get("completion_tokens", 0) + ot.get("completion_tokens", 0)
        total["total_tokens"] = total.get("total_tokens", 0) + ot.get("total_tokens", 0)
        steps.extend(_orphan_usage["by_step"])
        _orphan_usage["total"] = dict(_DEFAULT_TOTAL)
        _orphan_usage["by_step"] = []
        # Expose input_tokens / output_tokens (aliases for prompt_tokens / completion_tokens)
        return {
            "total": _with_input_output_aliases(total),
            "by_step": [_with_input_output_aliases(s) for s in steps],
        }
