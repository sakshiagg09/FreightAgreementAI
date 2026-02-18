"""
Run create_freight_agreement and create_rate_table_validity in the background.
Returns immediately without showing SAP results to the user.
"""
import logging
import threading

from google.adk.tools import FunctionTool
from utils.session_context import get_session_id

from tools.create_freight_agreement import create_freight_agreement
from tools.create_rate_table_validity import create_rate_table_validity

logger = logging.getLogger(__name__)


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
    print(f"[create_agreement_and_validity_async] Background thread started: session_id={session_id}, {start_date} to {end_date}", flush=True)
    try:
        result_fa = create_freight_agreement(
            agreement_description=agreement_description,
            start_date=start_date,
            end_date=end_date,
            currency=currency,
            session_id=session_id,
        )
        out_fa = result_fa[:300] + "..." if len(result_fa) > 300 else result_fa
        logger.info("[Background] create_freight_agreement result: %s", out_fa)
        print(f"[create_agreement_and_validity_async] create_freight_agreement result: {out_fa}", flush=True)
        result_rtv = create_rate_table_validity(session_id=session_id)
        out_rtv = result_rtv[:300] + "..." if len(result_rtv) > 300 else result_rtv
        logger.info("[Background] create_rate_table_validity result: %s", out_rtv)
        print(f"[create_agreement_and_validity_async] create_rate_table_validity result: {out_rtv}", flush=True)
    except Exception as e:
        logger.exception("Background agreement/validity creation failed: %s", e)
        print(f"[create_agreement_and_validity_async] Background FAILED: {e}", flush=True)
    else:
        logger.info("[Background] Agreement and validity creation finished for session_id=%s", session_id)
        print(f"[create_agreement_and_validity_async] Background finished for session_id={session_id}", flush=True)


def create_agreement_and_validity_async(
    agreement_description: str,
    start_date: str,
    end_date: str,
    currency: str = "EUR",
) -> str:
    """
    Start freight agreement and rate table validity creation in the background.
    Does not wait for SAP or return their results. Use when agreement details
    are already confirmed and you do not want to show the user tool output.

    Args:
        agreement_description: Description (e.g. from extracted PDF).
        start_date: YYYY-MM-DD.
        end_date: YYYY-MM-DD.
        currency: Optional, default EUR.

    Returns:
        Short message that creation was started; do not surface details to the user.
    """
    session_id = get_session_id()
    logger.info(
        "[Async] Starting background thread for agreement/validity: session_id=%s, start=%s, end=%s",
        session_id, start_date, end_date,
    )
    print(f"[create_agreement_and_validity_async] Tool called: session_id={session_id}, start={start_date}, end={end_date}", flush=True)
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
    return "Freight Agreement and Rate Table Validity creation has been started."


create_agreement_and_validity_async_tool = FunctionTool(create_agreement_and_validity_async)
