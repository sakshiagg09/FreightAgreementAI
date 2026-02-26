"""
Session Context Management
Allows tools to access session_id from the current request context
Also stores UUIDs (Agreement UUID, Rate Table Validity UUID) per session
"""
import contextvars
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Context variable to store current session_id
_current_session_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('session_id', default=None)

# Session-level UUID storage: session_id -> {uuid_type: uuid_value}
# Example: {"test_session_123": {"agreement_uuid": "abc-123", "validity_uuid": "xyz-789"}}
_uuid_storage: Dict[str, Dict[str, str]] = {}

 # Session-level column mapping from select_rate_card_fields: session_id -> {system_field_key: excel_column_name}
_column_mapping_storage: Dict[str, dict] = {}

# Session-level charge type from resolve_charge_type_tool: session_id -> charge_type_code
_charge_type_storage: Dict[str, str] = {}

# Session-level rate card Excel path from select_rate_card_fields: session_id -> excel_file_path
_rate_card_path_storage: Dict[str, str] = {}

# Session-level Excel header row (0-based) for "header far down" templates; None = use default (2)
_excel_header_row_storage: Dict[str, Optional[int]] = {}

# Full field selection result from select_rate_card_fields for structured API response:
# session_id -> {"column_mapping": [...], "fields": [...], "weight_columns": [...]}
_field_selection_result_storage: Dict[str, Dict[str, Any]] = {}

# Flag: user requested retry for field selection (e.g. clicked Retry button in UI)
# When set, select_rate_card_fields will use retry logic regardless of enable_retries param.
_field_selection_retry_requested: Dict[str, bool] = {}

# Unselected field mappings: when user validates with a subset of checkboxes,
# mappings that were not selected are stored here (list of {excel_column, system_field_key, system_field_description}).
_unselected_field_mappings_storage: Dict[str, list] = {}

def set_session_id(session_id: str):
    """Set the current session ID in context"""
    _current_session_id.set(session_id)

def get_session_id() -> str:
    """Get the current session ID from context, or default"""
    session_id = _current_session_id.get()
    return session_id if session_id is not None else "default"

def store_uuid(session_id: str, uuid_type: str, uuid_value: str):
    """
    Store a UUID for a session.
    
    Args:
        session_id: Session identifier
        uuid_type: Type of UUID ('agreement_uuid' or 'validity_uuid')
        uuid_value: The UUID value to store
    """
    if session_id not in _uuid_storage:
        _uuid_storage[session_id] = {}
    _uuid_storage[session_id][uuid_type] = uuid_value
    logger.info(f"Stored {uuid_type} for session {session_id}: {uuid_value[:20]}...")

def get_uuid(session_id: str, uuid_type: str) -> Optional[str]:
    """
    Get a stored UUID for a session.
    
    Args:
        session_id: Session identifier
        uuid_type: Type of UUID ('agreement_uuid' or 'validity_uuid')
    
    Returns:
        UUID value if found, None otherwise
    """
    uuid_value = _uuid_storage.get(session_id, {}).get(uuid_type)
    if uuid_value:
        logger.debug(f"Retrieved {uuid_type} for session {session_id}: {uuid_value[:20]}...")
    return uuid_value

def get_agreement_uuid(session_id: Optional[str] = None) -> Optional[str]:
    """
    Convenience method to get Agreement UUID for current or specified session.
    
    Args:
        session_id: Optional session ID. If not provided, uses current session from context.
    
    Returns:
        Agreement UUID if found, None otherwise
    """
    if session_id is None:
        session_id = get_session_id()
    return get_uuid(session_id, "agreement_uuid")

def get_agreement_id(session_id: Optional[str] = None) -> Optional[str]:
    """
    Convenience method to get TransportationAgreement ID (e.g. 8100000087) for current or specified session.
    Used for GET FreightAgreement filter.
    
    Args:
        session_id: Optional session ID. If not provided, uses current session from context.
    
    Returns:
        Agreement ID if found, None otherwise
    """
    if session_id is None:
        session_id = get_session_id()
    return get_uuid(session_id, "agreement_id")

def get_validity_uuid(session_id: Optional[str] = None) -> Optional[str]:
    """
    Convenience method to get Rate Table Validity UUID for current or specified session.
    
    Args:
        session_id: Optional session ID. If not provided, uses current session from context.
    
    Returns:
        Validity UUID if found, None otherwise
    """
    if session_id is None:
        session_id = get_session_id()
    return get_uuid(session_id, "validity_uuid")


def store_charge_type(session_id: str, charge_type: str) -> None:
    """
    Store resolved SAP charge_type code for a session.
    Typically set by fetch_charge_type_tool.resolve_charge_type.
    """
    charge_type = (charge_type or "").strip()
    if not charge_type:
        return
    _charge_type_storage[session_id] = charge_type
    logger.info(f"Stored charge_type for session {session_id}: {charge_type}")


def get_charge_type(session_id: Optional[str] = None) -> Optional[str]:
    """
    Get stored SAP charge_type code for current or specified session.
    Returns None if no charge_type stored.
    """
    if session_id is None:
        session_id = get_session_id()
    return _charge_type_storage.get(session_id)


def store_column_mapping(session_id: str, mapping: dict):
    """
    Store column mapping (system_field_key -> excel_column_name) for a session.
    Set by select_rate_card_fields; used by batch_upload_to_sap.
    """
    _column_mapping_storage[session_id] = dict(mapping) if mapping else {}
    logger.info(f"Stored column mapping for session {session_id}: {list(_column_mapping_storage[session_id].keys())}")


def store_excel_header_row(session_id: str, header_row: int) -> None:
    """
    Store the 0-based Excel header row for "header far down" templates.
    Used by batch_upload_to_sap so it reads with header=header_row.
    """
    _excel_header_row_storage[session_id] = header_row
    logger.info(f"Stored Excel header row for session {session_id}: {header_row}")


def get_excel_header_row(session_id: Optional[str] = None) -> Optional[int]:
    """
    Get the stored Excel header row (0-based). Returns None if not set;
    caller should use default (e.g. 2) when None.
    """
    if session_id is None:
        session_id = get_session_id()
    return _excel_header_row_storage.get(session_id)


def get_column_mapping(session_id: Optional[str] = None) -> Optional[Dict]:
    """
    Get stored column mapping for current or specified session.
    Returns None if no mapping stored.
    """
    if session_id is None:
        session_id = get_session_id()
    return _column_mapping_storage.get(session_id) or None


def store_field_selection_result(session_id: str, result: Dict[str, Any]) -> None:
    """
    Store the full field selection result (column_mapping list, fields, weight_columns)
    for structured API responses. Set by select_rate_card_fields.
    """
    if not result:
        return
    _field_selection_result_storage[session_id] = {
        "column_mapping": result.get("column_mapping", []),
        "column_mapping_dict": result.get("column_mapping_dict", {}),
        "fields": result.get("fields", []),
        "weight_columns": result.get("weight_columns", []),
    }
    logger.info(f"Stored field selection result for session {session_id}")


def get_field_selection_result(session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get the stored field selection result for current or specified session.
    Returns None if none stored.
    """
    if session_id is None:
        session_id = get_session_id()
    return _field_selection_result_storage.get(session_id) or None


def store_unselected_field_mappings(session_id: str, mappings: list) -> None:
    """
    Store the list of field mappings that the user did not select when validating.
    Each item is a dict with excel_column, system_field_key, system_field_description.
    """
    _unselected_field_mappings_storage[session_id] = list(mappings) if mappings else []
    logger.info("Stored %d unselected field mapping(s) for session %s", len(_unselected_field_mappings_storage[session_id]), session_id)


def get_unselected_field_mappings(session_id: Optional[str] = None) -> Optional[list]:
    """
    Get the list of field mappings that were unselected at last validate.
    Returns None if none stored.
    """
    if session_id is None:
        session_id = get_session_id()
    return _unselected_field_mappings_storage.get(session_id)


def store_rate_card_path(session_id: str, excel_file_path: str) -> None:
    """
    Store the last used Excel rate card path for a session.
    This allows tools like batch_upload_to_sap to infer the Excel path
    even if the caller does not pass it explicitly.
    """
    excel_file_path = (excel_file_path or "").strip()
    if not excel_file_path:
        return
    _rate_card_path_storage[session_id] = excel_file_path
    logger.info(f"Stored rate card path for session {session_id}: {excel_file_path}")


def get_rate_card_path(session_id: Optional[str] = None) -> Optional[str]:
    """
    Get the stored Excel rate card path for the current or specified session.
    Returns None if no path is stored.
    """
    if session_id is None:
        session_id = get_session_id()
    return _rate_card_path_storage.get(session_id)


def set_field_selection_retry_requested(session_id: str) -> None:
    """
    Mark that the user requested a retry for field selection (e.g. clicked Retry in UI).
    select_rate_card_fields will use retry logic when this flag is set.
    """
    _field_selection_retry_requested[session_id] = True
    logger.info("Field selection retry requested for session %s", session_id)


def get_and_clear_field_selection_retry_requested(session_id: Optional[str] = None) -> bool:
    """
    Return True if retry was requested for this session, and clear the flag (one-shot).
    Used by select_rate_card_fields to enable retries without relying on the agent passing enable_retries.
    """
    if session_id is None:
        session_id = get_session_id()
    requested = _field_selection_retry_requested.pop(session_id, False)
    if requested:
        logger.info("Using retry logic for field selection (session %s)", session_id)
    return requested


def clear_session_uuids(session_id: str):
    """
    Clear all stored UUIDs for a session.
    
    Args:
        session_id: Session identifier
    """
    if session_id in _uuid_storage:
        del _uuid_storage[session_id]
        logger.info(f"Cleared UUID storage for session {session_id}")

def list_all_stored_uuids() -> Dict[str, Dict[str, str]]:
    """
    Get all stored UUIDs across all sessions (for debugging).
    
    Returns:
        Dictionary mapping session_id to dict of uuid_type -> uuid_value
    """
    return _uuid_storage.copy()
