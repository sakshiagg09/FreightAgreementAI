"""
Create Freight Agreement in SAP
Based on Postman collection structure
"""
import json
import logging
import requests
from google.adk.tools import FunctionTool
from utils.sap_client import SAPClient, SAP_FREIGHT_AGREEMENT_BASE, SAP_PURCHASING_ORG, SAP_BUSINESS_PARTNER
from utils.session_context import get_session_id, store_uuid, get_charge_type, get_column_mapping
from utils.system_field_mapping import SYSTEM_FIELD_MAPPING

logger = logging.getLogger(__name__)

FREIGHT_AGREEMENT_ENDPOINT = f"{SAP_FREIGHT_AGREEMENT_BASE}/FreightAgreement"

def create_freight_agreement(
    agreement_description: str,
    start_date: str,  # Format: "YYYY-MM-DD"
    end_date: str,    # Format: "YYYY-MM-DD"
    currency: str = "EUR",
    agreement_type: str = "ZRF1",
    agreement_status: str = "01",  # 01 = Active, 02 = Inactive
    charge_type: str | None = None,
    session_id: str = None,
) -> str:
    """
    Create a Freight Agreement in SAP using the extracted agreement data.

    Args:
        agreement_description: Description of the agreement
        start_date: Agreement start date (YYYY-MM-DD)
        end_date: Agreement end date (YYYY-MM-DD)
        currency: Currency code (default: EUR)
        agreement_type: Agreement type code (default: ZRF1)
        agreement_status: Agreement status (default: 01 = Active)
        session_id: Session ID for CSRF token caching

    Returns:
        str: Success message with Agreement UUID, or error message
    """
    logger.info(f"Creating Freight Agreement: {agreement_description}")

    # Validate required fields before calling SAP
    if not agreement_description or not str(agreement_description).strip():
        return (
            "Error: Agreement description is required and cannot be empty. "
            "Please provide a description for the freight agreement (e.g. from the extracted PDF or user input)."
        )
    agreement_description = str(agreement_description).strip()

    if not start_date or not str(start_date).strip():
        return "Error: Start date is required (YYYY-MM-DD format)."
    start_date = str(start_date).strip()

    if not end_date or not str(end_date).strip():
        return "Error: End date is required (YYYY-MM-DD format)."
    end_date = str(end_date).strip()

    # Get session_id from context if not provided
    if session_id is None:
        session_id = get_session_id()

    # Resolve charge type:
    # 1) explicit parameter if provided
    # 2) value stored by fetch_charge_type_tool for this session
    # 3) fallback default
    effective_charge_type = (charge_type or "").strip() if charge_type else ""
    if not effective_charge_type:
        session_charge_type = get_charge_type(session_id)
        if session_charge_type:
            effective_charge_type = session_charge_type
        else:
            effective_charge_type = "Z_BASE_FRGHT_RF"

    logger.info(f"Using charge_type for Freight Agreement: {effective_charge_type}")

    # Resolve TransportationCalculationBase values from stored column_mapping + SYSTEM_FIELD_MAPPING
    column_mapping = get_column_mapping(session_id) or {}

    def _pick_calc_base(candidates: list[str], fallback: str) -> str:
        for key in candidates:
            if key in column_mapping:
                return key
        return fallback

    source_candidates = [
        key
        for key, (_scale, desc) in SYSTEM_FIELD_MAPPING.items()
        if "Source Location" in desc
    ]
    dest_candidates = [
        key
        for key, (_scale, desc) in SYSTEM_FIELD_MAPPING.items()
        if "Destination Location" in desc
    ]
    weight_candidates = [
        key
        for key, (_scale, desc) in SYSTEM_FIELD_MAPPING.items()
        if "gross weight" in desc.lower()
    ]

    source_calc_base = _pick_calc_base(source_candidates, "SOURCELOC_ZONE")
    dest_calc_base = _pick_calc_base(dest_candidates, "DESTLOC_ZONE")
    weight_calc_base = _pick_calc_base(weight_candidates, "GROSS_WEIGHT")
    # service_calc_base = _pick_calc_base(weight_candidates, "SERVICE_CODE")

    logger.info(
        "Using TransportationCalculationBase values from column_mapping: "
        f"source={source_calc_base}, dest={dest_calc_base}, weight={weight_calc_base}"
    )

    # Build the SAP API payload (aligned with working CURL from COMPLETE_WORKFLOW_CURLS.md)
    # Use simplified structure - SAP may reject extra/optional fields
    payload = {
        "TransportationAgreementDesc": agreement_description,
        "TransportationAgreementType": agreement_type,
        "TranspAgreementValidFrom": start_date,
        "TranspAgreementValidTo": end_date,
        "TranspAgreementTimeZone": "CET",
        "TransportationAgreementDocCrcy": currency,
        "TransportationShippingType": "",
        "TransportationMode": "",
        "TransportationAgreementStatus": agreement_status,
        "_FreightAgreementItem": [
            {
                "TransportationCalculationSheet": "",
                "_FreightAgrmtCalcSheet": [
                    {
                        "TransportationCalculationSheet": "",
                        "_FrtAgrmtCalcSheetItem": [
                            {
                                "TranspChargeType": effective_charge_type,
                                "TranspCalcResolutionBase": "ROOT",
                                "TranspCalcSheetItemCurrency": currency,
                                "TranspCalcSheetItemAmount": 0.0,
                                "TranspChargeInstrnType": "STND",
                                "TranspChargeIsDependent": False,
                                "TranspRateTableID": "",
                                "_FreightAgreementRateTable": [
                                    {
                                        "TranspRateTableID": "",
                                        "TranspRateTableValueType": "A",
                                        "TranspChargeType": effective_charge_type,
                                        "TranspRateTableSignType": "+",
                                        "TranspRateTableTimeZone": "CET",
                                        "_FrtAgrmtRateTableScaleRef": [
                                            {
                                                "TranspRateTableDimensionIndex": "2",
                                                "TransportationCalculationBase": source_calc_base,
                                                "TranspRateTblScaleRefScaleType": "X",
                                                "TranspRateTblScaleRefCalcType": "A",
                                            },
                                            {
                                                "TranspRateTableDimensionIndex": "1",
                                                "TransportationCalculationBase": dest_calc_base,
                                                "TranspRateTblScaleRefScaleType": "X",
                                                "TranspRateTblScaleRefCalcType": "A",
                                            },
                                            {
                                                "TranspRateTableDimensionIndex": "3",
                                                "TransportationCalculationBase": weight_calc_base,
                                                "TranspRateTblScaleRefScaleType": "B",
                                                "TranspRateTblScaleRefQtyUnit": "KG",
                                                "TranspRateTblScaleRefCalcType": "A",
                                            },
                                            # {
                                            #     "TranspRateTableDimensionIndex": "4",
                                            #     "TransportationCalculationBase": service_calc_base,
                                            #     "TranspRateTblScaleRefScaleType": "X",
                                            #     "TranspRateTblScaleRefCalcType": "A",
                                            # },
                                        ],
                                        # Omit _FrtAgrmtRateTableValidity - SAP auto-creates it from
                                        # TranspAgreementValidFrom/TranspAgreementValidTo. Sending it
                                        # causes "Property 'TranspRateTableValidFrom' is invalid" error.
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
        "_FrtAgrmtOrganization": [
            {"FreightAgreementPurchasingOrg": SAP_PURCHASING_ORG}
        ],
        "_FreightAgreementParty": [
            {"BusinessPartner": SAP_BUSINESS_PARTNER}
        ],
    }
    try:
        # Use SAPClient helper which:
        # - fetches CSRF token
        # - reuses cookies (SAP_SESSIONID, etc.) from the token request
        # This mirrors the working Postman request you shared.
        logger.info(f"Calling SAP API: {FREIGHT_AGREEMENT_ENDPOINT}")
        print(payload)
        response = SAPClient.post_with_csrf(
            session_id=session_id,
            endpoint=FREIGHT_AGREEMENT_ENDPOINT,
            json=payload,
            timeout=(10, 60),
        )

        response.raise_for_status()
        result = response.json()

        # Extract Agreement UUID from response (SAP may wrap in "d" or return entity directly)
        d = result.get("d", result)
        agreement_uuid = d.get("TransportationAgreementUUID", "")
        agreement_id = d.get("TransportationAgreement", "")

        if not agreement_uuid:
            return f"Error: Could not get Agreement UUID from SAP response: {json.dumps(result)}"

        success_msg = (
            f"SUCCESS: Freight Agreement created in SAP\n"
            f"- Agreement ID: {agreement_id}\n"
            f"- Agreement UUID: {agreement_uuid}\n"
            f"- Description: {agreement_description}\n"
            f"- Valid From: {start_date}\n"
            f"- Valid To: {end_date}\n"
            f"- Currency: {currency}\n\n"
            f"⚠️ IMPORTANT: Save this Agreement UUID for next steps: {agreement_uuid}"
        )

        logger.info(f"Freight Agreement created successfully: {agreement_uuid}")
        
        # Store Agreement UUID and ID for later use (e.g., in create_rate_table_validity GET)
        store_uuid(session_id, "agreement_uuid", agreement_uuid)
        if agreement_id:
            store_uuid(session_id, "agreement_id", agreement_id)
        
        return success_msg

    except requests.exceptions.HTTPError as e:
        response_body = "No response body"
        if e.response is not None:
            response_body = e.response.text
            if not response_body and e.response.content:
                response_body = e.response.content.decode("utf-8", errors="replace")
            response_body = response_body or "Empty body"
            if e.response.headers.get("Content-Type", "").startswith("application/json") and response_body:
                try:
                    err_json = json.loads(response_body)
                    response_body = json.dumps(err_json, indent=2)
                except json.JSONDecodeError:
                    pass
        error_msg = (
            f"HTTP Error creating Freight Agreement: {e}\n"
            f"Status: {e.response.status_code if e.response else 'N/A'}\n"
            f"Response: {response_body}"
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

create_freight_agreement_tool = FunctionTool(create_freight_agreement)
