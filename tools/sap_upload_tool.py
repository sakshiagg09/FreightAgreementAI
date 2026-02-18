"""
Batch Upload Rate Data to SAP (STRICT VERSION)
•  No alias-based auto detection
•  No system key guessing
•  Uses only stored mapping (select_rate_card_fields) OR explicit column parameters

SAP PREREQUISITE (fix "all columns show <= 50 KG"):
  The rate table's GROSS_WEIGHT scale must have one scale item per weight break.
  Before or after the first upload, in SAP go to: Manage Rate Tables → your table
  → Scale Items (or Select Column Scale) → dimension 3 (GROSS_WEIGHT).
  Create scale items with values matching the Excel weight brackets (e.g. 50, 75, 100,
  150, 200, 300, ... kg). Without this, every rate is stored under the single existing
  scale item and the UI shows multiple columns with the same label and wrong layout.
"""

import json
import logging
import os
import re
import time
import pandas as pd
import requests
from google.adk.tools import FunctionTool
from utils.sap_client import SAPClient, SAP_RATE_TABLE_BASE
from utils.session_context import (
    get_session_id,
    get_validity_uuid,
    get_column_mapping,
    get_excel_header_row,
    get_rate_card_path,
)
from utils.system_field_mapping import SYSTEM_FIELD_MAPPING

logger = logging.getLogger(__name__)

# Log prefix for easy filtering: grep "[sap_upload]" to find all SAP upload logs.
# Errors include status_code, response_body, get_url, validity_uuid for debugging.
SAP_UPLOAD_LOG_PREFIX = "[sap_upload]"


def _safe_response_text(resp, max_len: int = 500) -> str:
    """Safely get response text for logging; truncate if too long."""
    try:
        text = getattr(resp, "text", None) or str(resp)
        if len(text) > max_len:
            return text[:max_len] + f"... (truncated, total {len(text)} chars)"
        return text
    except Exception:
        return "<unable to read response>"


# No trailing slash: SAP returns 400 "URI has a slash as last character" otherwise
SAP_BATCH_ENDPOINT = f"{SAP_RATE_TABLE_BASE.rstrip('/')}/$batch"

# Optional: try to create scale items so each weight gets its own column (endpoint may vary by system)
SAP_SCALE_ITEM_RELATIVE = "TranspRateTableValidity(TranspRateTableValidityUUID='{uuid}')/_TranspRateTableScaleItem"

# Path to list rates for a validity (for clear-before-upload)
# Use unquoted UUID to match batch POST format (create_agreement_and_batch_rates.sh)
def _rates_collection_path(validity_uuid: str) -> str:
    u = (validity_uuid or "").strip().strip("'\"")
    return f"TranspRateTableValidity(TranspRateTableValidityUUID={u})/_TranspRateTableRate"

# Single batch for all entries so SAP shows 1 row per lane (not 1 row per chunk).
BATCH_CHUNK_SIZE = 210
DELETE_BATCH_CHUNK_SIZE = 25
BATCH_MAX_RETRIES = 3
BATCH_RETRY_DELAY_SEC = 2
STOP_ON_FIRST_BATCH_FAILURE = True

INVALID_AMOUNT_STRINGS = (
    "upon request", "upon_request", "nan", "", "n/a", "na", "no", "null",
    "0", "0.", "0.0", "0.00", "0,00", "0,0", ".0", ",0",
)

# Module import log (debug level to avoid noise in production)
logger.debug("Loaded sap_upload_tool module.")


def _normalize_zone(value: str) -> str:
    """Strip and uppercase for consistent de-dup. Treat 0/empty as blank."""
    if value is None:
        return ""
    s = str(value).strip().upper()
    if not s or s in ("0", "0.0", "0,0", "0,00", "000"):
        return ""
    return s


def _parse_amount(raw) -> float | None:
    """Parse rate amount; handle comma decimal. Returns None if invalid or zero."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip().replace(",", ".")
    if not s or s.lower() in INVALID_AMOUNT_STRINGS:
        return None
    try:
        v = float(s)
        return v if v > 0 else None
    except ValueError:
        return None


# -------------------------------------------------------
# Helper: Resolve exact DataFrame column (case-insensitive)
# -------------------------------------------------------
def _resolve_excel_column_to_df_column(df: pd.DataFrame, excel_col_name: str) -> str | None:
    if not excel_col_name:
        return None

    excel_norm = excel_col_name.strip().lower()
    for col in df.columns:
        if str(col).strip().lower() == excel_norm:
            return col
    return None


# -------------------------------------------------------
# Helper: Resolve mapping column to actual DataFrame column (handles composite headers)
# -------------------------------------------------------
def _resolve_mapping_column_to_df_column(df: pd.DataFrame, excel_col_name: str) -> str | None:
    """
    Resolve the stored mapping column name (from select_rate_card_fields) to an actual
    column in the DataFrame. Handles composite headers where the stored value is a
    concatenation of multiple header cells (e.g. "Origin country Minimum € / km ...")
    but the Excel was read with a single header row so df.columns has "Origin country",
    "Minimum", "€ / km", etc.
    """
    if not excel_col_name or not isinstance(df.columns, pd.Index):
        return None

    # 1) Exact or case-insensitive match
    resolved = _resolve_excel_column_to_df_column(df, excel_col_name)
    if resolved is not None:
        return resolved

    # 2) Stored value not in df.columns (e.g. composite from merged multi-row header).
    #    Find the longest df column that is a prefix of the stored value (normalized).
    excel_stripped = excel_col_name.strip()
    excel_lower = excel_stripped.lower()
    candidates = [
        col for col in df.columns
        if col and excel_lower.startswith(str(col).strip().lower())
    ]
    if candidates:
        # Prefer longest match so "Origin country" wins over "Origin"
        return max(candidates, key=lambda c: len(str(c).strip()))

    # 3) Stored value might start with the logical column (e.g. "Origin country" in sheet,
    #    but stored as "Origin country Minimum ..."). Already covered by (2).

    return None



# -------------------------------------------------------
# Read Excel
# -------------------------------------------------------
def _read_excel(excel_file_path: str, header_row: int | None = None) -> pd.DataFrame:
    """
    Read the Excel rate card. Uses header_row (0-based) when provided;
    otherwise default 2 (third physical row). For "header far down" templates,
    select_rate_card_fields stores the detected row and it is passed here.
    """
    header = header_row if header_row is not None else 2
    df = pd.read_excel(excel_file_path, header=header, sheet_name=0, engine=None)
    logger.debug(
        "Read Excel file for SAP upload.",
        extra={
            "excel_file_path": os.path.abspath(os.path.expanduser(str(excel_file_path))),
            "header_row": header,
            "columns": list(df.columns),
            "row_count": len(df),
        },
    )
    return df


# -------------------------------------------------------
# Deduplicate DataFrame by (origin, destination) so each lane appears once
# -------------------------------------------------------
def _deduplicate_wide_by_lane(
        df: pd.DataFrame,
        origin_col: str,
        dest_col: str,
        weight_excel_columns: list[str],
        currency_col: str | None,
) -> pd.DataFrame:
    """
    Collapse duplicate (origin, destination) rows into one row per lane.
    For each weight column, use the first non-zero rate in the group so we
    get one logical row per lane (e.g. one 'AT AT' instead of many).
    """
    norm_origin = df[origin_col].astype(str).apply(_normalize_zone)
    norm_dest = df[dest_col].astype(str).apply(_normalize_zone)
    valid = (norm_origin != "") & (norm_dest != "")
    df = df.loc[valid].copy()
    norm_origin = norm_origin[valid]
    norm_dest = norm_dest[valid]
    if df.empty:
        return df
    df = df.copy()
    df["_lane_origin"] = norm_origin.values
    df["_lane_dest"] = norm_dest.values

    grouped = df.groupby(["_lane_origin", "_lane_dest"], as_index=False)
    result = grouped.first()  # first row per group for origin, dest, currency
    for wcol in weight_excel_columns:
        if wcol not in df.columns:
            continue
        # Build a plain list of scalars (one per group) to avoid 2D array / dimension errors
        values = []
        for _, grp in grouped:
            col_vals = grp[wcol]
            chosen = None
            for v in col_vals:
                if _parse_amount(v) is not None:
                    chosen = v
                    break
            if chosen is None and len(col_vals):
                chosen = col_vals.iloc[0]
            values.append(chosen)
        result[wcol] = values
    result = result.drop(columns=["_lane_origin", "_lane_dest"], errors="ignore")
    return result


def _deduplicate_long_by_lane(
        df: pd.DataFrame,
        origin_col: str,
        dest_col: str,
        weight_col: str,
        amount_col: str,
        currency_col: str | None,
) -> pd.DataFrame:
    """
    Collapse duplicate (origin, destination, weight) rows into one per lane+weight.
    Keeps the first non-zero amount per (origin, dest, weight).
    """
    norm_origin = df[origin_col].astype(str).apply(_normalize_zone)
    norm_dest = df[dest_col].astype(str).apply(_normalize_zone)
    valid = (norm_origin != "") & (norm_dest != "")
    df = df.loc[valid].copy()
    if df.empty:
        return df
    df = df.copy()
    df["_no"] = norm_origin.values
    df["_nd"] = norm_dest.values
    try:
        weight_vals = pd.to_numeric(df[weight_col], errors="coerce")
    except Exception:
        weight_vals = pd.Series(dtype=float)
    if len(weight_vals) != len(df):
        return df
    df["_nw"] = weight_vals
    df = df[df["_nw"] > 0]
    if df.empty:
        return df
    keep_cols = [c for c in df.columns if c not in ("_no", "_nd", "_nw")]
    seen: set[tuple] = set()
    rows = []
    for _, row in df.iterrows():
        key = (row["_no"], row["_nd"], float(row["_nw"]))
        if key in seen:
            continue
        if _parse_amount(row.get(amount_col)) is None:
            continue
        seen.add(key)
        rows.append({c: row[c] for c in keep_cols})
    if not rows:
        return pd.DataFrame(columns=keep_cols)
    return pd.DataFrame(rows)


# -------------------------------------------------------
# Helper: Resolve origin/destination columns from mapping
# WITHOUT hardcoding system field keys
# -------------------------------------------------------
def _resolve_column_by_description(column_mapping: dict, description_substrings: list[str]) -> str | None:
    """
    Given a column_mapping of {system_field_key: excel_column_name},
    find the first excel_column_name whose system field description
    (from SYSTEM_FIELD_MAPPING) contains any of the provided substrings.
    """
    if not column_mapping:
        return None

    lowered_substrings = [s.lower() for s in description_substrings]

    for system_field_key, excel_column in column_mapping.items():
        mapping_entry = SYSTEM_FIELD_MAPPING.get(system_field_key)
        if not mapping_entry:
            # Weight slabs and other non-system keys won't be in SYSTEM_FIELD_MAPPING
            continue

        _, description = mapping_entry
        desc_lower = str(description).lower()
        if any(sub in desc_lower for sub in lowered_substrings):
            return excel_column

    return None


# -------------------------------------------------------
# Extract numeric weight from column name (wide format)
# -------------------------------------------------------
def extract_weight_from_column(weight_column: str) -> float:
    """
    Parse weight (upper bound in kg) from Excel column names like:
    - "30-50" -> 50, "51-75" -> 75
    - "> 20 001" or "> 20001" -> 20001
    - "1001-1250" -> 1250
    - "FTL\\n33 PALLETS" -> 33 (or use a high weight break)
    """
    weight_str = str(weight_column).strip().replace("\n", " ").replace("\r", " ")

    # Range: "30-50", "51-75", "1001-1250" -> upper bound
    range_match = re.search(r"(\d[\d\s]*)\s*-\s*(\d[\d\s]*)", weight_str)
    if range_match:
        upper = range_match.group(2).replace(" ", "")
        if upper:
            return float(upper)

    # "> 20 001", "> 20001" -> single number
    gt_match = re.search(r">\s*(\d[\d\s]*)", weight_str)
    if gt_match:
        num_str = gt_match.group(1).replace(" ", "")
        if num_str:
            return float(num_str)

    # Single number or last number in string (e.g. "33 PALLETS" -> 33)
    numeric_match = re.search(r"(\d+\.?\d*)", weight_str)
    if numeric_match:
        return float(numeric_match.group(1))

    return 0.0


def _is_likely_rate_bucket_column(col_name: str, reserved: set) -> bool:
    """True if column name looks like a weight/rate bucket and is not reserved."""
    if not col_name:
        return False
    col_lower = str(col_name).strip().lower()
    reserved_lower = {str(r).strip().lower() for r in reserved if r}
    if col_lower in reserved_lower:
        return False
    s = str(col_name).strip().upper()
    # Must contain digits to be a weight/rate bucket
    if not re.search(r"\d", s):
        return False
    # Exclude known non-rate columns
    if re.match(r"^(LOT|SHADOW|LANE\s*NUMBER|SITE\s*NAME|SERVICE\s*TYPE|LEAD\s*TIME|CURRENCY)$", s, re.I):
        return False
    return True


def _auto_detect_weight_columns(
        df: pd.DataFrame,
        origin_col: str,
        dest_col: str,
        currency_col: str | None,
        column_mapping: dict,
) -> list[tuple[str, str]]:
    """
    When column_mapping has no weight slabs, detect rate columns from Excel headers.
    Returns list of (mapping_key, excel_column) for columns that look like weight/rate buckets.
    """
    reserved = {origin_col, dest_col}
    if currency_col:
        reserved.add(currency_col)
    for excel_col in (column_mapping or {}).values():
        reserved.add(str(excel_col).strip())

    weight_columns = []
    for col in df.columns:
        col_str = str(col).strip()
        if not _is_likely_rate_bucket_column(col_str, reserved):
            continue
        # Use actual df column reference for indexing; synthetic key for mapping
        weight_columns.append(("WEIGHT_SLAB", col))
    return weight_columns


def format_gross_weight_for_sap(weight_value: float) -> str:
    """
    Format weight (kg) for SAP TransportationScaleItem03Value.
    Use integer when whole so SAP can match existing scale items (e.g. 50, 75, 100);
    otherwise use 6 decimals so scale items are distinct.
    """
    w = float(weight_value)
    if w == int(w):
        return str(int(w))
    return f"{w:.6f}"


def _try_create_scale_items(
        session_id: str,
        rate_table_validity_uuid: str,
        weight_values_kg: list[float],
) -> None:
    """
    Try to create scale items for dimension 3 (GROSS_WEIGHT) via a single $batch
    request so CSRF is fetched once for the batch endpoint (avoids "URI has a slash
    as last character" when fetching CSRF for the validity entity URL).
    """
    if not weight_values_kg:
        return
    batch_boundary = "batch_scale_001"
    changeset_boundary = "changeset_scale"
    scale_item_path = f"TranspRateTableValidity(TranspRateTableValidityUUID={rate_table_validity_uuid})/_TranspRateTableScaleItem"
    chunk_parts = []
    for idx, w in enumerate(weight_values_kg, 1):
        scale_value = format_gross_weight_for_sap(w)
        payload = {
            "TranspRateTableDimensionIndex": "3",
            "TransportationScaleItem03Value": scale_value,
        }
        chunk_parts.append(
            f"""--{changeset_boundary}
Content-Type: application/http
Content-Transfer-Encoding: binary
Content-ID: {idx}

POST {scale_item_path} HTTP/1.1
OData-Version: 4.0
OData-MaxVersion: 4.0
Content-Type: application/json
Accept: application/json

{json.dumps(payload, indent=2)}
"""
        )
    batch_body = f"--{batch_boundary}\n"
    batch_body += "Content-Type: multipart/mixed; boundary=changeset_scale\n\n"
    batch_body += "\n".join(chunk_parts)
    batch_body += f"\n--{changeset_boundary}--\n"
    batch_body += f"--{batch_boundary}--"
    endpoint = SAP_BATCH_ENDPOINT.rstrip("/")
    try:
        resp = SAPClient.post_with_csrf_raw(
            session_id=session_id,
            endpoint=endpoint,
            data=batch_body,
            content_type=f"multipart/mixed;boundary={batch_boundary}",
            timeout=(10, 60),
            session_base=SAP_BATCH_ENDPOINT,
        )
        if resp.status_code in (200, 202):
            logger.info(
                "Scale items batch sent (weight dimension); check SAP for result.",
            )
        else:
            logger.debug(
                "Scale items batch returned %s; API may not support creation.",
                resp.status_code,
            )
    except Exception as exc:
        logger.debug(
            "Could not create scale items via batch (endpoint may not exist): %s",
            exc,
        )


def _clear_existing_rates_for_validity(
    session_id: str,
    rate_table_validity_uuid: str,
) -> tuple[str, str]:
    """
    Fetch all existing rate records for this validity and DELETE them in batch,
    so the subsequent upload replaces the table instead of appending (avoids duplicate rows).

    Returns:
        ("cleared", str(count)) if rates were deleted,
        ("skipped", reason) if no rates or clear was skipped,
        ("error", message) if GET/DELETE failed (upload still proceeds).
    """
    base = SAP_RATE_TABLE_BASE.rstrip("/")
    path = _rates_collection_path(rate_table_validity_uuid)
    get_url = f"{base}/{path}"
    logger.info(
        f"{SAP_UPLOAD_LOG_PREFIX} Clear step: GET existing rates",
        extra={
            "validity_uuid": rate_table_validity_uuid,
            "get_url": get_url,
            "session_id": session_id,
        },
    )
    try:
        # Use session_base so GET shares cookies with subsequent batch POST (matches curl)
        response = SAPClient.get_with_session(
            session_id=session_id,
            endpoint=get_url,
            timeout=(10, 60),
            session_base=SAP_BATCH_ENDPOINT,
        )
        if response.status_code != 200:
            reason = f"GET existing rates returned status {response.status_code}"
            logger.warning(
                f"{SAP_UPLOAD_LOG_PREFIX} Could not fetch existing rates for clear; skipping clear. %s | URL=%s | response_body=%s",
                reason,
                get_url,
                _safe_response_text(response),
                extra={
                    "status_code": response.status_code,
                    "get_url": get_url,
                    "validity_uuid": rate_table_validity_uuid,
                },
            )
            return ("skipped", reason)
        data = response.json()
        values = data.get("value") or []
        logger.info(
            f"{SAP_UPLOAD_LOG_PREFIX} Clear step: GET returned %s existing rate(s) for validity.",
            len(values),
            extra={"validity_uuid": rate_table_validity_uuid, "existing_rate_count": len(values)},
        )
        if not values:
            return ("skipped", "no existing rates")
        # Collect delete paths: use @odata.id or build from TranspRateTableRateUUID
        delete_paths = []
        for item in values:
            odata_id = item.get("@odata.id") or item.get("@odata.editLink")
            if odata_id:
                # Extract path after service root (e.g. .../0001/TranspRateTableRate('x') or .../TranspRateTableValidity('v')/_TranspRateTableRate('x'))
                if "/0001/" in odata_id:
                    rel = odata_id.split("/0001/", 1)[-1].lstrip("/")
                else:
                    rel = odata_id.split("/")[-2] + "/" + odata_id.split("/")[-1] if "/" in odata_id else odata_id
                delete_paths.append(rel)
            else:
                rate_uuid = item.get("TranspRateTableRateUUID")
                if rate_uuid:
                    v_uuid = (rate_table_validity_uuid or "").strip().strip("'\"")
                    # Use unquoted UUIDs to match batch POST format
                    delete_paths.append(f"TranspRateTableValidity(TranspRateTableValidityUUID={v_uuid})/_TranspRateTableRate(TranspRateTableRateUUID={rate_uuid})")
        if not delete_paths:
            reason = "could not determine delete paths from API response"
            logger.warning(
                f"{SAP_UPLOAD_LOG_PREFIX} Skipping clear: %s | GET returned %s items; sample keys: %s",
                reason,
                len(values),
                list(values[0].keys())[:10] if values else [],
                extra={"value_count": len(values), "validity_uuid": rate_table_validity_uuid},
            )
            return ("skipped", reason)
        logger.info(
            "Clearing %s existing rate(s) for this validity before upload.",
            len(delete_paths),
            extra={"count": len(delete_paths)},
        )
        batch_boundary = "batch_delete_001"
        changeset_boundary = "changeset_delete"
        crlf = "\n"
        for chunk_start in range(0, len(delete_paths), DELETE_BATCH_CHUNK_SIZE):
            chunk_paths = delete_paths[chunk_start : chunk_start + DELETE_BATCH_CHUNK_SIZE]
            chunk_parts = []
            for idx, rel_path in enumerate(chunk_paths, 1):
                chunk_parts.append(
                    f"--{changeset_boundary}{crlf}"
                    f"Content-Type: application/http{crlf}"
                    f"Content-Transfer-Encoding: binary{crlf}"
                    f"Content-ID: {idx}{crlf}{crlf}"
                    f"DELETE {rel_path} HTTP/1.1{crlf}"
                    f"OData-Version: 4.0{crlf}"
                    f"OData-MaxVersion: 4.0{crlf}"
                )
            batch_body = (
                f"--{batch_boundary}{crlf}"
                f"Content-Type: multipart/mixed; boundary={changeset_boundary}{crlf}{crlf}"
                + crlf.join(chunk_parts)
                + f"{crlf}--{changeset_boundary}--{crlf}"
                + f"--{batch_boundary}--"
            )
            try:
                del_resp = SAPClient.post_with_csrf_raw(
                    session_id=session_id,
                    endpoint=SAP_BATCH_ENDPOINT.rstrip("/"),
                    data=batch_body,
                    content_type=f"multipart/mixed;boundary={batch_boundary}",
                    timeout=(10, 120),
                    session_base=SAP_BATCH_ENDPOINT,
                )
                if del_resp.status_code not in (200, 202):
                    logger.warning(
                        f"{SAP_UPLOAD_LOG_PREFIX} Delete batch returned %s | response_body=%s",
                        del_resp.status_code,
                        _safe_response_text(del_resp),
                        extra={
                            "status_code": del_resp.status_code,
                            "chunk_start": chunk_start,
                            "delete_count": len(chunk_paths),
                        },
                    )
            except Exception as del_exc:
                logger.warning(
                    f"{SAP_UPLOAD_LOG_PREFIX} Could not delete existing rates batch: %s",
                    del_exc,
                    exc_info=True,
                    extra={"chunk_start": chunk_start, "delete_count": len(chunk_paths)},
                )
        return ("cleared", str(len(delete_paths)))
    except Exception as e:
        logger.warning(
            f"{SAP_UPLOAD_LOG_PREFIX} Could not clear existing rates (GET or DELETE failed): %s. Proceeding with upload (may append).",
            e,
            exc_info=True,
            extra={"get_url": get_url, "validity_uuid": rate_table_validity_uuid},
        )
        return ("error", str(e))


# -------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------
def batch_upload_to_sap(
        excel_file_path: str = "",
        rate_table_validity_uuid: str = "",
        origin_column: str = "",
        destination_column: str = "",
        currency_column: str = "",
        session_id: str = None,
        max_weight_breaks: int | None = None,
) -> str:
    """
    Upload rate card data from Excel to SAP. By default uploads all weight columns from the Excel
    (e.g. 50, 75, 100, 150, 200, ... 20001 kg). Use max_weight_breaks=N to limit to first N weights
    if your SAP rate table has fewer scale items.
    """
    if session_id is None:
        session_id = get_session_id()

    # If caller did not provide an Excel path, fall back to the last
    # rate card used for this session (set by select_rate_card_fields).
    if not excel_file_path:
        stored_path = get_rate_card_path(session_id)
        if stored_path:
            excel_file_path = stored_path

    abs_excel = os.path.abspath(os.path.expanduser(str(excel_file_path))) if excel_file_path else ""
    logger.info(
        f"{SAP_UPLOAD_LOG_PREFIX} Starting SAP batch upload.",
        extra={
            "session_id": session_id,
            "excel_file_path": abs_excel,
            "rate_table_validity_uuid": rate_table_validity_uuid or "<auto>",
            "origin_column_param": origin_column,
            "destination_column_param": destination_column,
            "currency_column_param": currency_column,
            "max_weight_breaks_param": max_weight_breaks,
        },
    )

    if not rate_table_validity_uuid:
        rate_table_validity_uuid = get_validity_uuid(session_id)
        logger.info(
            f"{SAP_UPLOAD_LOG_PREFIX} Validity UUID from session: %s",
            "resolved" if rate_table_validity_uuid else "not found",
            extra={"session_id": session_id, "validity_uuid_truncated": (rate_table_validity_uuid[:36] + "...") if rate_table_validity_uuid and len(rate_table_validity_uuid) > 36 else rate_table_validity_uuid},
        )
        if not rate_table_validity_uuid:
            logger.error(
                f"{SAP_UPLOAD_LOG_PREFIX} Missing rate_table_validity_uuid and unable to resolve from session.",
                extra={"session_id": session_id},
            )
            return "Error: rate_table_validity_uuid is required."

    if not excel_file_path:
        logger.error(
            f"{SAP_UPLOAD_LOG_PREFIX} Excel file path is missing for SAP batch upload and no stored rate card path was found.",
            extra={"session_id": session_id},
        )
        return (
            "Error: Excel file path is required for batch upload. "
            "Please upload the rate card and run select_rate_card_fields first, "
            "or pass excel_file_path explicitly."
        )

    excel_file_path = os.path.abspath(os.path.expanduser(str(excel_file_path)))
    if not os.path.exists(excel_file_path):
        logger.error(
            f"{SAP_UPLOAD_LOG_PREFIX} Excel file not found for SAP batch upload.",
            extra={"excel_file_path": excel_file_path},
        )
        return f"Error: Excel file not found: {excel_file_path}"

    try:
        excel_header_row = get_excel_header_row(session_id)
        df = _read_excel(excel_file_path, header_row=excel_header_row)
        logger.info(
            f"{SAP_UPLOAD_LOG_PREFIX} Excel loaded: %s rows, %s columns.",
            len(df),
            len(df.columns),
            extra={"rows": len(df), "columns_sample": list(df.columns)[:15]},
        )

        column_mapping = get_column_mapping(session_id)
        logger.debug(
            f"{SAP_UPLOAD_LOG_PREFIX} Column mapping from session.",
            extra={"session_id": session_id, "column_mapping_keys": list(column_mapping.keys()) if column_mapping else [], "mapping_count": len(column_mapping) if column_mapping else 0},
        )

        if not column_mapping and not (origin_column and destination_column):
            logger.error(
                f"{SAP_UPLOAD_LOG_PREFIX} Column mapping not found and required columns not passed explicitly.",
                extra={"session_id": session_id, "column_mapping": bool(column_mapping)},
            )
            return (
                "Error: Column mapping not found. "
                "Run select_rate_card_fields first "
                "or pass column names explicitly."
            )

        # -------------------------------------
        # Resolve required columns
        # -------------------------------------
        # Prefer explicit parameters if provided.
        # Otherwise, infer origin/destination from the descriptions in SYSTEM_FIELD_MAPPING
        # so we are not tied to any specific system field key (no hardcoded SOURCELOC/DESTLOC).
        origin_col = origin_column or _resolve_column_by_description(
            column_mapping,
            ["Source Location", "Transportation Zone of Source Location"],
        )
        dest_col = destination_column or _resolve_column_by_description(
            column_mapping,
            ["Destination Location", "Transportation Zone of Destination Location"],
        )
        # Resolve to actual DataFrame column (handles composite headers from merged rows,
        # e.g. "Origin country Minimum € / km ..." -> "Origin country")
        if origin_col:
            origin_col = _resolve_mapping_column_to_df_column(df, origin_col) or origin_col
        if dest_col:
            dest_col = _resolve_mapping_column_to_df_column(df, dest_col) or dest_col

        weight_col_long = column_mapping.get("GROSS_WEIGHT")
        amount_col_long = column_mapping.get("GOODS_VALUE")
        currency_col = currency_column or column_mapping.get("CURRENCY")
        # Resolve to actual DataFrame columns (handles composite headers)
        if weight_col_long:
            weight_col_long = _resolve_mapping_column_to_df_column(df, weight_col_long) or weight_col_long
        if amount_col_long:
            amount_col_long = _resolve_mapping_column_to_df_column(df, amount_col_long) or amount_col_long
        if currency_col:
            currency_col = _resolve_mapping_column_to_df_column(df, currency_col) or currency_col

        # Validate required: origin is mandatory
        if not origin_col or origin_col not in df.columns:
            logger.error(
                f"{SAP_UPLOAD_LOG_PREFIX} Origin column not found in DataFrame.",
                extra={"origin_col": origin_col, "df_columns": list(df.columns)},
            )
            return f"Error: Origin column not found in the Excel file. The system expected a column for 'Transportation Zone of Source Location' (e.g. 'Origin country'). Please ensure the rate card has an origin column and that column mapping was run (select_rate_card_fields)."

        # Destination optional: if rate card has no destination column (origin-only matrix),
        # use origin as destination so each row becomes (origin, origin) with the rate.
        if not dest_col or dest_col not in df.columns:
            dest_col = origin_col
            logger.info(
                f"{SAP_UPLOAD_LOG_PREFIX} No destination column in Excel; using origin as destination (origin-only rate card).",
                extra={"origin_col": origin_col},
            )

        # Detect format
        is_long_format = bool(weight_col_long and amount_col_long)

        logger.info(
            "Determined rate table format for SAP upload.",
            extra={
                "is_long_format": is_long_format,
                "rows_in_df": len(df),
                "origin_col": origin_col,
                "dest_col": dest_col,
                "weight_col_long": weight_col_long,
                "amount_col_long": amount_col_long,
                "currency_col": currency_col,
            },
        )

        all_entries = []
        # Track unique entries to avoid creating duplicate entities in SAP
        seen_entry_keys: set[tuple] = set()

        def _add_sap_entry(origin, destination, weight_value, rate_amount, currency):
            """
            Add a single SAP rate entry, de-duplicating by (origin, destination, weight).
            Only one rate per (origin, dest, weight) is sent; we keep the first amount seen.
            Never add entries with zero or negative amount (avoids "AT AT 0 0" duplicate rows).
            Origin/destination are normalized (strip + upper) so duplicates are caught.
            """
            parsed = _parse_amount(rate_amount)
            if parsed is None:
                logger.debug(
                    "Skipping entry with zero/invalid amount; not sent to SAP.",
                    extra={"origin": origin, "destination": destination, "weight": weight_value},
                )
                return

            normalized_amount = round(parsed, 6)
            norm_origin = _normalize_zone(origin)
            norm_dest = _normalize_zone(destination)
            if not norm_origin or not norm_dest:
                logger.debug(
                    "Skipping entry with blank origin/destination.",
                    extra={"origin": origin, "destination": destination},
                )
                return

            normalized_currency = (currency or "EUR").strip() or "EUR"
            normalized_weight = format_gross_weight_for_sap(float(weight_value))

            # One entity per (origin, dest, weight) — normalized so "AT"/"at"/"AT " are same
            entry_key = (norm_origin, norm_dest, normalized_weight)

            if entry_key in seen_entry_keys:
                logger.debug(
                    "Skipping duplicate (origin, dest, weight); keeping first rate.",
                    extra={
                        "origin": norm_origin,
                        "destination": norm_dest,
                        "weight": weight_value,
                        "amount": rate_amount,
                    },
                )
                return

            seen_entry_keys.add(entry_key)

            sap_entry = {
                "TransportationCalcBase01": "SOURCELOC_ZONE",
                "TransportationCalcBase02": "DESTLOC_ZONE",
                "TransportationCalcBase03": "GROSS_WEIGHT",
                "TransportationRateCurrency": normalized_currency,
                "TransportationRateAmount": normalized_amount,
                "TranspRateTableDimensionIndex": "3",
                "TransportationScaleItem01Value": norm_origin,
                "TransportationScaleItem02Value": norm_dest,
                "TransportationScaleItem03Value": normalized_weight,
            }

            all_entries.append({
                "sap_entry": sap_entry,
                "origin": norm_origin,
                "destination": norm_dest,
                "weight": weight_value,
                "amount": normalized_amount,
                "currency": normalized_currency,
            })

            logger.debug(
                "Prepared SAP rate entry.",
                extra={
                    "origin": norm_origin,
                    "destination": norm_dest,
                    "weight": weight_value,
                    "amount": normalized_amount,
                    "currency": normalized_currency,
                },
            )

        # -------------------------------------
        # LONG FORMAT
        # -------------------------------------
        if is_long_format:

            if weight_col_long not in df.columns:
                logger.error(
                    "Configured long-format weight column not found in DataFrame.",
                    extra={"weight_col_long": weight_col_long, "df_columns": list(df.columns)},
                )
                return f"Error: Weight column '{weight_col_long}' not found."
            if amount_col_long not in df.columns:
                logger.error(
                    "Configured long-format amount column not found in DataFrame.",
                    extra={"amount_col_long": amount_col_long, "df_columns": list(df.columns)},
                )
                return f"Error: Amount column '{amount_col_long}' not found."

            # One row per (origin, destination, weight): collapse duplicates
            df_long = _deduplicate_long_by_lane(
                df, origin_col, dest_col, weight_col_long, amount_col_long, currency_col
            )
            if len(df_long) < len(df):
                logger.info(
                    "Deduplicated long format by lane+weight: %s -> %s rows.",
                    len(df),
                    len(df_long),
                    extra={"before": len(df), "after": len(df_long)},
                )
            df = df_long

            logger.info(
                "Processing DataFrame in long format for SAP upload.",
                extra={
                    "rows_in_df": len(df),
                    "weight_col_long": weight_col_long,
                    "amount_col_long": amount_col_long,
                },
            )

            for _, row in df.iterrows():
                try:
                    origin = _normalize_zone(row.get(origin_col, ""))
                    destination = _normalize_zone(row.get(dest_col, ""))

                    if not origin or not destination:
                        continue

                    rate_amount = _parse_amount(row.get(amount_col_long))
                    if rate_amount is None:
                        continue

                    try:
                        weight_value = float(row[weight_col_long])
                    except (TypeError, ValueError):
                        continue
                    if weight_value <= 0:
                        continue

                    currency = (
                        str(row[currency_col]).strip()
                        if currency_col and currency_col in df.columns
                        else "EUR"
                    )

                    _add_sap_entry(origin, destination, weight_value, rate_amount, currency)

                except Exception as row_exc:
                    logger.debug(
                        "Skipping row during long-format processing due to exception.",
                        exc_info=row_exc,
                    )
                    continue

        # -------------------------------------
        # WIDE FORMAT
        # -------------------------------------
        else:

            weight_columns = []

            if column_mapping:
                for key, excel_col in column_mapping.items():
                    # Treat columns with digits in either the mapping key OR
                    # the actual Excel column name as weight slabs.
                    if (
                            (re.search(r"\d+", str(key)) or re.search(r"\d+", str(excel_col)))
                            and excel_col in df.columns
                    ):
                        weight_columns.append((key, excel_col))

            # If no weight columns from mapping (e.g. LLM omitted weight buckets),
            # auto-detect from Excel: columns that look like weight/rate buckets
            # (e.g. "30-50", "51-75", "> 20 001", "1/2 FTL", "FTL 33 PALLETS").
            if not weight_columns:
                weight_columns = _auto_detect_weight_columns(
                    df, origin_col, dest_col, currency_col, column_mapping
                )
                logger.info(
                    "Auto-detected weight/rate columns from Excel headers (not in mapping).",
                    extra={
                        "auto_detected_count": len(weight_columns),
                        "columns": [wc[1] for wc in weight_columns],
                    },
                )

            # One column per weight value so SAP does not show the same weight twice (e.g. two "<= 150 KG")
            seen_weights: set[float] = set()
            deduped: list[tuple[str, str]] = []
            for mapping_key, excel_col in weight_columns:
                parsed = extract_weight_from_column(str(excel_col))
                if parsed <= 0:
                    continue
                if parsed in seen_weights:
                    logger.debug(
                        "Skipping duplicate weight column (same kg as another): %s -> %s kg",
                        excel_col,
                        parsed,
                    )
                    continue
                seen_weights.add(parsed)
                deduped.append((mapping_key, excel_col))
            weight_columns = deduped

            # Log resolved weight columns and their interpreted upper-limit weights
            if weight_columns:
                debug_weight_info = []
                for mapping_key, excel_col in weight_columns:
                    parsed_weight = extract_weight_from_column(str(excel_col))
                    debug_weight_info.append(
                        {
                            "mapping_key": mapping_key,
                            "excel_column": excel_col,
                            "parsed_weight": parsed_weight,
                        }
                    )
                logger.info(
                    "Resolved wide-format weight columns.",
                    extra={"weight_columns": debug_weight_info},
                )

            if not weight_columns:
                logger.error(
                    f"{SAP_UPLOAD_LOG_PREFIX} No weight columns found for wide-format processing (mapping or auto-detect).",
                    extra={
                        "column_mapping_keys": list(column_mapping.keys()) if column_mapping else [],
                        "df_columns_sample": list(df.columns)[:30],
                    },
                )
                return (
                    "Error: No weight/rate columns found for wide format. "
                    "Ensure your Excel has columns like '30-50', '51-75', etc. for weight brackets, "
                    "or add them to the rate card field mapping."
                )

            # One row per (origin, destination): collapse duplicates so e.g. "AT AT" appears once
            weight_excel_cols = [wcol for _, wcol in weight_columns]
            df_wide = _deduplicate_wide_by_lane(
                df, origin_col, dest_col, weight_excel_cols, currency_col
            )
            if len(df_wide) < len(df):
                logger.info(
                    "Deduplicated wide format by lane: %s -> %s rows (one per origin–destination).",
                    len(df),
                    len(df_wide),
                    extra={"before": len(df), "after": len(df_wide)},
                )
            df = df_wide

            logger.info(
                "Processing DataFrame in wide format for SAP upload.",
                extra={
                    "rows_in_df": len(df),
                    "weight_column_count": len(weight_columns),
                },
            )

            for _, row in df.iterrows():
                try:
                    origin = _normalize_zone(row.get(origin_col, ""))
                    destination = _normalize_zone(row.get(dest_col, ""))

                    if not origin or not destination:
                        logger.debug(
                            "Skipping row: missing or blank origin/destination.",
                            extra={"origin": origin, "destination": destination},
                        )
                        continue

                    currency = (
                        str(row[currency_col]).strip()
                        if currency_col and currency_col in df.columns
                        else "EUR"
                    )

                    for mapping_key, wcol in weight_columns:
                        rate_value = row[wcol]
                        rate_amount = _parse_amount(rate_value)
                        if rate_amount is None:
                            logger.debug(
                                "Skipping cell: zero/invalid/NaN rate value.",
                                extra={
                                    "origin": origin,
                                    "destination": destination,
                                    "excel_weight_column": wcol,
                                },
                            )
                            continue
                        # Always derive the weight from the Excel column header
                        # (wcol), so headers like "30-50" are interpreted using
                        # their upper limit (50), regardless of the mapping key.
                        weight_value = extract_weight_from_column(str(wcol))
                        if weight_value == 0:
                            logger.debug(
                                "Skipping cell: could not parse weight from column header.",
                                extra={
                                    "origin": origin,
                                    "destination": destination,
                                    "excel_weight_column": wcol,
                                    "raw_header": str(wcol),
                                },
                            )
                            continue

                        logger.debug(
                            "Adding SAP rate entry (wide format).",
                            extra={
                                "origin": origin,
                                "destination": destination,
                                "excel_weight_column": wcol,
                                "derived_weight": weight_value,
                                "rate_amount": rate_amount,
                                "currency": currency,
                            },
                        )

                        _add_sap_entry(origin, destination, weight_value, rate_amount, currency)

                except Exception as row_exc:
                    logger.debug(
                        "Skipping row during wide-format processing due to exception.",
                        exc_info=row_exc,
                    )
                    continue

        if not all_entries:
            logger.error(
                f"{SAP_UPLOAD_LOG_PREFIX} No valid rate entries prepared for SAP upload after processing DataFrame.",
                extra={"rows_in_df": len(df), "is_long_format": is_long_format},
            )
            return "Error: No valid rate entries found."

        # -------------------------------------------------------
        # Final pass: keep exactly one entry per (origin, dest, weight), no zeros.
        # Ensures we never send duplicates or "AT AT 0 0" even if something slipped through.
        # -------------------------------------------------------
        final_keys: set[tuple[str, str, str]] = set()
        filtered: list[dict] = []
        for e in all_entries:
            key = (e["origin"], e["destination"], format_gross_weight_for_sap(float(e["weight"])))
            if key in final_keys:
                continue
            amt = e.get("amount") or e["sap_entry"].get("TransportationRateAmount")
            if amt is None or float(amt) <= 0:
                continue
            final_keys.add(key)
            filtered.append(e)
        dropped = len(all_entries) - len(filtered)
        if dropped:
            logger.warning(
                "Dropped %s duplicate or zero-amount entries before sending; sending only unique (origin, dest, weight) with positive amount.",
                dropped,
                extra={"before": len(all_entries), "after": len(filtered)},
            )
        all_entries = filtered

        if not all_entries:
            logger.error(
                f"{SAP_UPLOAD_LOG_PREFIX} No rate entries left after removing duplicates and zeros.",
                extra={"rows_in_df": len(df)},
            )
            return "Error: No valid rate entries found after de-duplication."

        # -------------------------------------------------------
        # Sort by weight, then origin, then destination so SAP sees
        # scale values in ascending order (50, 75, 100, ...). This
        # can help systems that create or assign scale items by order.
        # -------------------------------------------------------
        all_entries.sort(key=lambda e: (float(e["weight"]), e["origin"], e["destination"]))

        # -------------------------------------------------------
        # Optional: limit weight breaks. Default: no limit (upload all weights from Excel).
        # Use max_weight_breaks=N or env SAP_MAX_WEIGHT_BREAKS=N to restrict (e.g. if SAP
        # rate table has fewer scale items). SAP requires one scale item per weight for clean display.
        # -------------------------------------------------------
        max_breaks = max_weight_breaks
        if max_breaks is None or max_breaks <= 0:
            try:
                env_val = os.environ.get("SAP_MAX_WEIGHT_BREAKS", "").strip()
                max_breaks = int(env_val) if env_val else 0
            except ValueError:
                max_breaks = 0
        logger.info(
            f"{SAP_UPLOAD_LOG_PREFIX} Weight break limit: param=%s env=%s resolved=%s (0=upload all).",
            max_weight_breaks,
            os.environ.get("SAP_MAX_WEIGHT_BREAKS", ""),
            max_breaks,
            extra={"max_weight_breaks_param": max_weight_breaks, "SAP_MAX_WEIGHT_BREAKS_env": os.environ.get("SAP_MAX_WEIGHT_BREAKS"), "resolved_max_breaks": max_breaks},
        )
        if max_breaks and max_breaks > 0:
            unique_weights_ordered = sorted(set(e["weight"] for e in all_entries))
            allowed_weights = set(unique_weights_ordered[:max_breaks])
            all_entries = [e for e in all_entries if e["weight"] in allowed_weights]
            logger.info(
                f"{SAP_UPLOAD_LOG_PREFIX} Limited to first %s weight breaks to match curl display.",
                max_breaks,
                extra={"allowed_weights": list(allowed_weights), "max_weight_breaks": max_breaks},
            )

        # -------------------------------------------------------
        # Summary: (origin, dest, weight) counts for verification
        # -------------------------------------------------------
        unique_lanes = set((e["origin"], e["destination"]) for e in all_entries)
        unique_weights = set(e["weight"] for e in all_entries)
        weight_breaks_sorted = sorted(unique_weights)
        summary = {
            "total_rate_rows": len(all_entries),
            "unique_origin_dest_pairs": len(unique_lanes),
            "unique_weight_breaks": len(unique_weights),
            "weight_breaks_kg": weight_breaks_sorted,
        }
        logger.info(
            "SAP upload summary (origin, dest, weight) — verify against Excel.",
            extra=summary,
        )
        # Do NOT create scale items here; it causes duplicate columns and 0,00 display.
        # The curl script (create_agreement_and_batch_rates.sh) gets correct display because
        # it creates the agreement with a rate table that has the right scale structure.
        # Our create_agreement_and_validity_async may create differently—check that flow.
        logger.info(
            "Uploading rates (no scale item creation). Weight breaks: %s.",
            weight_breaks_sorted,
        )

        # -------------------------------------------------------
        # CLEAR EXISTING RATES (replace instead of append — avoids duplicate rows)
        # -------------------------------------------------------
        logger.info(
            f"{SAP_UPLOAD_LOG_PREFIX} Clearing existing rates before upload.",
            extra={"validity_uuid": rate_table_validity_uuid},
        )
        clear_status, clear_detail = _clear_existing_rates_for_validity(
            session_id, rate_table_validity_uuid
        )
        logger.info(
            f"{SAP_UPLOAD_LOG_PREFIX} Clear step result: %s (%s)",
            clear_status,
            clear_detail,
            extra={"clear_status": clear_status, "validity_uuid": rate_table_validity_uuid},
        )

        # -------------------------------------------------------
        # BATCH SEND
        # -------------------------------------------------------
        batch_boundary = "batch_123453"
        changeset_boundary = "changeset_all"

        response = None

        logger.info(
            f"{SAP_UPLOAD_LOG_PREFIX} Beginning SAP batch upload over HTTP.",
            extra={
                "total_entries": len(all_entries),
                "batch_chunk_size": BATCH_CHUNK_SIZE,
                "max_retries": BATCH_MAX_RETRIES,
                "validity_uuid": rate_table_validity_uuid,
            },
        )

        # Normalize validity UUID (strip quotes) to match working curl format
        validity_uuid = (rate_table_validity_uuid or "").strip().strip("'\"")

        total_chunks = (len(all_entries) + BATCH_CHUNK_SIZE - 1) // BATCH_CHUNK_SIZE
        logger.info(
            f"{SAP_UPLOAD_LOG_PREFIX} Will upload %s entries in %s chunk(s) (chunk_size=%s).",
            len(all_entries),
            total_chunks,
            BATCH_CHUNK_SIZE,
            extra={"total_entries": len(all_entries), "total_chunks": total_chunks, "chunk_size": BATCH_CHUNK_SIZE},
        )
        for chunk_start in range(0, len(all_entries), BATCH_CHUNK_SIZE):
            chunk = all_entries[chunk_start:chunk_start + BATCH_CHUNK_SIZE]
            chunk_index = (chunk_start // BATCH_CHUNK_SIZE) + 1

            chunk_parts = []

            # Line endings: use \n to match create_agreement_and_batch_rates.sh (bash produces LF)
            crlf = "\n"
            for idx, entry in enumerate(chunk, 1):
                # Compact JSON (single line) to match create_agreement_and_batch_rates.sh exactly
                json_body = json.dumps(entry["sap_entry"], separators=(",", ":"))
                chunk_parts.append(
                    f"--{changeset_boundary}{crlf}"
                    f"Content-Type: application/http{crlf}"
                    f"Content-Transfer-Encoding: binary{crlf}"
                    f"Content-ID: {idx}{crlf}{crlf}"
                    f"POST TranspRateTableValidity(TranspRateTableValidityUUID={validity_uuid})/_TranspRateTableRate HTTP/1.1{crlf}"
                    f"OData-Version: 4.0{crlf}"
                    f"OData-MaxVersion: 4.0{crlf}"
                    f"Content-Type: application/json{crlf}"
                    f"Accept: application/json{crlf}{crlf}"
                    f"{json_body}{crlf}"
                )
            # Structure exactly matches create_agreement_and_batch_rates.sh BATCH_BODY
            batch_body = (
                f"--{batch_boundary}{crlf}"
                f"Content-Type: multipart/mixed; boundary=changeset_all{crlf}{crlf}"
                + crlf.join(chunk_parts)
                + f"{crlf}--{changeset_boundary}--{crlf}"
                + f"--{batch_boundary}--"
            )

            for attempt in range(1, BATCH_MAX_RETRIES + 1):
                try:
                    logger.info(
                        "Uploading SAP batch chunk %s of %s.",
                        chunk_index,
                        total_chunks,
                        extra={
                            "chunk_index": chunk_index,
                            "total_chunks": total_chunks,
                            "entries_in_chunk": len(chunk),
                            "attempt": attempt,
                        },
                    )

                    response = SAPClient.post_with_csrf_raw(
                        session_id=session_id,
                        endpoint=SAP_BATCH_ENDPOINT.rstrip("/"),
                        data=batch_body,
                        content_type=f"multipart/mixed;boundary={batch_boundary}",
                        timeout=(10, 600),
                        session_base=SAP_BATCH_ENDPOINT,
                    )

                    response.raise_for_status()

                    logger.info(
                        "Successfully uploaded SAP batch chunk %s of %s (status=%s).",
                        chunk_index,
                        total_chunks,
                        response.status_code,
                        extra={
                            "chunk_index": chunk_index,
                            "total_chunks": total_chunks,
                            "entries_in_chunk": len(chunk),
                            "status_code": response.status_code,
                        },
                    )
                    break

                except requests.RequestException as http_exc:
                    resp = getattr(http_exc, "response", None)
                    resp_text = _safe_response_text(resp) if resp else str(http_exc)
                    logger.warning(
                        f"{SAP_UPLOAD_LOG_PREFIX} Error uploading SAP batch chunk. Will retry if attempts remain. | status=%s | response_body=%s",
                        getattr(resp, "status_code", None) if resp else "N/A",
                        resp_text,
                        extra={
                            "chunk_index": chunk_index,
                            "attempt": attempt,
                            "max_retries": BATCH_MAX_RETRIES,
                            "validity_uuid": validity_uuid,
                        },
                        exc_info=True,
                    )

                    if attempt == BATCH_MAX_RETRIES:
                        logger.error(
                            f"{SAP_UPLOAD_LOG_PREFIX} Exhausted all retry attempts for SAP batch chunk. | last_status=%s | response_body=%s",
                            getattr(resp, "status_code", None) if resp else "N/A",
                            resp_text,
                            extra={
                                "chunk_index": chunk_index,
                                "entries_in_chunk": len(chunk),
                                "validity_uuid": validity_uuid,
                            },
                        )
                        if STOP_ON_FIRST_BATCH_FAILURE:
                            raise
                    else:
                        time.sleep(BATCH_RETRY_DELAY_SEC)

        logger.info(
            "Completed SAP batch upload.",
            extra={"total_entries_uploaded": len(all_entries)},
        )

        lane_word = "lane" if len(unique_lanes) == 1 else "lanes"
        weight_word = "weight break" if len(unique_weights) == 1 else "weight breaks"
        summary_line = (
            f"{len(all_entries)} rate rows ({len(unique_lanes)} origin–destination {lane_word} × "
            f"{len(unique_weights)} {weight_word})."
        )
        # SAP stores one rate record per (origin, dest, weight), so the UI may show one row per weight for the same lane; use a view that groups by lane / shows weight as columns if available.
        logger.info(
            "One SAP rate record per (origin, dest, weight). Use rate table view that groups by lane to see one row per origin–destination with weight as columns.",
            extra={"unique_lanes": len(unique_lanes), "unique_weights": len(unique_weights)},
        )
        weights_list = ", ".join(str(w) for w in weight_breaks_sorted[:20])
        if len(weight_breaks_sorted) > 20:
            weights_list += f", ... (+{len(weight_breaks_sorted) - 20} more)"
        sap_note = f" Weight breaks: {weights_list}. Same payload structure as create_agreement_and_batch_rates.sh."
        if clear_status == "cleared":
            clear_note = f" Cleared {clear_detail} existing rate(s) before upload so each lane appears once."
        else:
            clear_note = (
                f" Clear before upload was not run ({clear_detail}). "
                "If you see duplicate rows (e.g. multiple AT AT), in SAP use the Delete button to remove "
                "all existing rates for this validity, then run the upload again once."
            )
        view_note = (
            " SAP requires one scale item per weight break for clean display. Add scale items in "
            "Manage Rate Tables → Scale Items (dimension 3 GROSS_WEIGHT) if columns are duplicated or show 0,00."
        )
        return f"SUCCESS: Uploaded {len(all_entries)} rate entries to SAP. Summary: {summary_line}.{sap_note}{clear_note}{view_note}"

    except Exception as e:
        logger.exception(
            f"{SAP_UPLOAD_LOG_PREFIX} Unhandled exception during SAP batch upload: %s",
            e,
            extra={
                "session_id": session_id,
                "excel_file_path": excel_file_path,
                "rate_table_validity_uuid": rate_table_validity_uuid,
                "exception_type": type(e).__name__,
            },
        )
        return f"Error uploading batch: {str(e)}"


sap_upload_tool = FunctionTool(batch_upload_to_sap)