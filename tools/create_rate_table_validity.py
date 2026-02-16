"""
Create Rate Table Validity - retrieves TranspRateTableValidityUUID from Freight Agreement.

When create_freight_agreement creates an agreement, SAP automatically creates the rate table
validity structure. This tool calls the GET FreightAgreement API with expand to retrieve
the TranspRateTableValidityUUID from the nested structure, then stores it for batch upload.
"""
import logging
import requests
from typing import Optional
from google.adk.tools import FunctionTool
from utils.sap_client import SAPClient, SAP_FREIGHT_AGREEMENT_BASE
from utils.session_context import get_session_id, get_agreement_id, get_agreement_uuid, store_uuid

logger = logging.getLogger(__name__)

FREIGHT_AGREEMENT_ENDPOINT = f"{SAP_FREIGHT_AGREEMENT_BASE}/FreightAgreement"

# OData expand path to get _FrtAgrmtRateTableValidity (contains TranspRateTableValidityUUID)
EXPAND_PARAMS = (
    "_FreightAgreementItem($expand=_FreightAgrmtCalcSheet($expand=_FrtAgrmtCalcSheetItem"
    "($expand=_FreightAgreementRateTable($expand=_FrtAgrmtRateTableScaleRef,"
    "_FrtAgrmtRateTableValidity($expand=_FrtAgrmtRateTableCalcRule)))))"
)


def _extract_validity_uuid_from_response(data: dict) -> Optional[str]:
    """
    Extract first TranspRateTableValidityUUID from nested Freight Agreement GET response.
    Path: value[0]._FreightAgreementItem[0]._FreightAgrmtCalcSheet[0]._FrtAgrmtCalcSheetItem[0]
          ._FreightAgreementRateTable[0]._FrtAgrmtRateTableValidity[0].TranspRateTableValidityUUID
    """
    value_list = data.get("value", [])
    if not value_list:
        return None
    for agreement in value_list:
        items = agreement.get("_FreightAgreementItem", [])
        for item in items:
            calc_sheets = item.get("_FreightAgrmtCalcSheet", [])
            for sheet in calc_sheets:
                calc_items = sheet.get("_FrtAgrmtCalcSheetItem", [])
                for calc_item in calc_items:
                    rate_tables = calc_item.get("_FreightAgreementRateTable", [])
                    for rt in rate_tables:
                        validities = rt.get("_FrtAgrmtRateTableValidity", [])
                        for validity in validities:
                            uuid_val = validity.get("TranspRateTableValidityUUID")
                            if uuid_val:
                                return uuid_val
    return None


def create_rate_table_validity(
    agreement_id: str = "",
    agreement_uuid: str = "",
    session_id: str = None,
    start_date: str = "",
    end_date: str = "",
    timezone: str = "CET",
) -> str:
    """
    Retrieve TranspRateTableValidityUUID from the Freight Agreement created by create_freight_agreement.

    Calls the GET FreightAgreement API with $filter and $expand to fetch the nested structure,
    then extracts the first TranspRateTableValidityUUID and stores it for batch upload.
    Validity dates are already set when the freight agreement is created.

    Args:
        agreement_id: TransportationAgreement ID (e.g. 8100000087). Auto-retrieved from session if empty.
        agreement_uuid: TransportationAgreementUUID. Used as fallback if agreement_id not in session.
        session_id: Session ID for CSRF/cookie reuse.
        start_date: Optional, ignored (validity comes from freight agreement).
        end_date: Optional, ignored.
        timezone: Optional, ignored.

    Returns:
        str: Success message with TranspRateTableValidityUUID, or error message.
    """
    if session_id is None:
        session_id = get_session_id()

    # Resolve agreement_id for GET filter (TransportationAgreement eq '8100000087')
    if not agreement_id or agreement_id.strip() == "":
        agreement_id = get_agreement_id(session_id)
    if not agreement_id or agreement_id.strip() == "":
        # Fallback: we cannot GET without agreement_id; suggest user run create_freight_agreement first
        agreement_uuid_val = agreement_uuid or get_agreement_uuid(session_id)
        if agreement_uuid_val:
            return (
                "Error: TransportationAgreement ID is required for GET. "
                "It is stored when create_freight_agreement runs. "
                "Please run create_freight_agreement first, then call create_rate_table_validity again."
            )
        return (
            "Error: No Agreement ID or UUID in session. "
            "Please run create_freight_agreement first, then call create_rate_table_validity."
        )

    params = {
        "$filter": f"TransportationAgreement eq '{agreement_id}'",
        "$expand": EXPAND_PARAMS,
    }

    try:
        logger.info(f"GET Freight Agreement to extract TranspRateTableValidityUUID: filter={agreement_id}")
        response = SAPClient.get_with_session(
            session_id=session_id,
            endpoint=FREIGHT_AGREEMENT_ENDPOINT,
            params=params,
            timeout=(10, 60),
        )

        response.raise_for_status()
        data = response.json()

        validity_uuid = _extract_validity_uuid_from_response(data)
        if not validity_uuid:
            return (
                f"Error: Could not find TranspRateTableValidityUUID in Freight Agreement response. "
                f"Agreement ID: {agreement_id}. Response structure may have changed. "
                f"Raw keys: {list(data.keys())}"
            )

        # Store for batch upload
        store_uuid(session_id, "validity_uuid", validity_uuid)

        success_msg = (
            f"SUCCESS: Rate Table Validity UUID retrieved from Freight Agreement\n"
            f"- TranspRateTableValidityUUID: {validity_uuid}\n"
            f"- Agreement ID: {agreement_id}\n\n"
            f"⚠️ IMPORTANT: This Validity UUID is stored for batch upload. "
            f"You can now call batch_upload_to_sap with your Excel rate card."
        )
        logger.info(f"TranspRateTableValidityUUID retrieved: {validity_uuid}")
        return success_msg

    except requests.exceptions.HTTPError as e:
        error_msg = (
            f"HTTP Error fetching Freight Agreement: {e}\n"
            f"Response: {e.response.text if e.response else 'No response'}"
        )
        logger.error(error_msg)
        return f"Error: {error_msg}"
    except requests.exceptions.RequestException as e:
        error_msg = f"Request failed: {str(e)}"
        logger.error(error_msg)
        return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return f"Error: {error_msg}"


create_rate_table_validity_tool = FunctionTool(create_rate_table_validity)
