"""
Run create_freight_agreement and create_rate_table_validity.
- Sync (run_in_background=False): runs in sequence and returns full result including Agreement ID.
- Async (run_in_background=True): runs in background thread and returns a short message without Agreement ID.
"""
import logging
import re
import threading

from google.adk.tools import FunctionTool
from utils.session_context import get_session_id

from tools.create_freight_agreement import create_freight_agreement
from tools.create_rate_table_validity import create_rate_table_validity

logger = logging.getLogger(__name__)

# Message returned when running in background (no agreement ID available yet)
ASYNC_RETURN_MESSAGE = (
    "Your freight agreement has been created in SAP. "
    "Please upload your Excel rate card when you're ready for the batch upload."
)


def _run_agreement_and_validity(
    agreement_description: str,
    start_date: str,
    end_date: str,
    currency: str,
    session_id: str,
) -> None:
    """Run create_freight_agreement then create_rate_table_validity (no output to user)."""
    desc_short = agreement_description[:50] + "..." if len(agreement_description) > 50 else agreement_description
    logger.info(
        "[Background] Starting agreement and validity creation: session_id=%s, description=%s, %s to %s",
        session_id, desc_short, start_date, end_date,
    )
    try:
        create_freight_agreement(
            agreement_description=agreement_description,
            start_date=start_date,
            end_date=end_date,
            currency=currency,
            session_id=session_id,
        )
        create_rate_table_validity(session_id=session_id)
    except Exception as e:
        logger.exception("Background agreement/validity creation failed: %s", e)
    else:
        logger.info("[Background] Agreement and validity creation finished for session_id=%s", session_id)


def _format_sync_success(result_fa: str, result_rtv: str) -> str:
    """Format sync result for user: 'Your freight agreement has been created in SAP. Agreement ID: ... Please upload...'"""
    # Extract Agreement ID from create_freight_agreement result (e.g. "- Agreement ID: 8100000375")
    agreement_id_match = re.search(r"Agreement ID:\s*(\S+)", result_fa, re.IGNORECASE)
    agreement_id = agreement_id_match.group(1) if agreement_id_match else None
    if agreement_id:
        return (
            "Your freight agreement has been created in SAP.\n\n"
            f"**Agreement ID:** {agreement_id}\n\n"
            "Please upload your Excel rate card when you're ready for the batch upload."
        )
    return f"{result_fa}\n\n{result_rtv}"


def create_agreement_and_validity_async(
    agreement_description: str,
    start_date: str,
    end_date: str,
    currency: str = "EUR",
    run_in_background: bool = False,
) -> str:
    """
    Create Freight Agreement and Rate Table Validity in SAP.

    - If run_in_background=False (default): runs synchronously and returns the full result
      including Agreement ID. Use when you want to show the Agreement ID to the user.
    - If run_in_background=True: starts creation in a background thread and returns immediately
      with a short message (no Agreement ID). Use when agreement details are already confirmed
      and you do not need to show SAP details to the user.

    Args:
        agreement_description: Description (e.g. from extracted PDF).
        start_date: YYYY-MM-DD.
        end_date: YYYY-MM-DD.
        currency: Optional, default EUR.
        run_in_background: If True, run in background and return short message without Agreement ID.

    Returns:
        Sync: Full success message including Agreement ID, or error.
        Async: Short message that agreement was created and user can upload Excel.
    """
    session_id = get_session_id()
    logger.info(
        "create_agreement_and_validity_async: session_id=%s, start=%s, end=%s, run_in_background=%s",
        session_id, start_date, end_date, run_in_background,
    )

    if run_in_background:
        thread = threading.Thread(
            target=_run_agreement_and_validity,
            kwargs={
                "agreement_description": agreement_description,
                "start_date": start_date,
                "end_date": end_date,
                "currency": currency,
                "session_id": session_id,
            },
            daemon=True,
        )
        thread.start()
        return ASYNC_RETURN_MESSAGE

    # Sync: run and return full result with Agreement ID
    try:
        result_fa = create_freight_agreement(
            agreement_description=agreement_description,
            start_date=start_date,
            end_date=end_date,
            currency=currency,
            session_id=session_id,
        )
        if result_fa.strip().lower().startswith("error"):
            return result_fa
        result_rtv = create_rate_table_validity(session_id=session_id)
        if result_rtv.strip().lower().startswith("error"):
            return f"{result_fa}\n\nRate Table Validity failed:\n{result_rtv}"
        return _format_sync_success(result_fa, result_rtv)
    except Exception as e:
        logger.exception("Agreement/validity creation failed: %s", e)
        return f"Error: Agreement and validity creation failed - {e}"


create_agreement_and_validity_async_tool = FunctionTool(create_agreement_and_validity_async)
