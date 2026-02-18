"""
Run create_freight_agreement and create_rate_table_validity in sequence.
Returns the full result (including Agreement ID and UUID) so the agent can show it to the user.
"""
import logging

from google.adk.tools import FunctionTool
from utils.session_context import get_session_id

from tools.create_freight_agreement import create_freight_agreement
from tools.create_rate_table_validity import create_rate_table_validity

logger = logging.getLogger(__name__)


def create_agreement_and_validity_async(
    agreement_description: str,
    start_date: str,
    end_date: str,
    currency: str = "EUR",
) -> str:
    """
    Create Freight Agreement and Rate Table Validity in SAP.
    Runs create_freight_agreement then create_rate_table_validity and returns
    the full result including Agreement ID and Agreement UUID. Always include
    the Agreement ID and Agreement UUID in your response to the user.

    Args:
        agreement_description: Description (e.g. from extracted PDF).
        start_date: YYYY-MM-DD.
        end_date: YYYY-MM-DD.
        currency: Optional, default EUR.

    Returns:
        Full success or error message from SAP, including Agreement ID and UUID.
    """
    session_id = get_session_id()
    logger.info(
        "Creating agreement and validity: session_id=%s, start=%s, end=%s",
        session_id, start_date, end_date,
    )
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
        return f"{result_fa}\n\n{result_rtv}"
    except Exception as e:
        logger.exception("Agreement/validity creation failed: %s", e)
        return f"Error: Agreement and validity creation failed - {e}"


create_agreement_and_validity_async_tool = FunctionTool(create_agreement_and_validity_async)
