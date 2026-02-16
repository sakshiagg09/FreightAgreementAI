import json
import logging
import re
import os
import pandas as pd
import requests
from google.adk.tools import FunctionTool
from utils.system_field_mapping import SYSTEM_FIELD_MAPPING
from utils.session_context import get_session_id, store_column_mapping, store_rate_card_path
from config.dev_config import (
    AZURE_OPENAI_CHAT_URL,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT,
)

# Import Phoenix tracing
from utils.phoenix_tracing import spa_upload_tracer
from opentelemetry.trace import get_current_span, Status, StatusCode

logger = logging.getLogger(__name__)

# Extract field descriptions from SYSTEM_FIELD_MAPPING
FIELD_DESCRIPTIONS = [desc for _, desc in SYSTEM_FIELD_MAPPING.values()]

DESCRIPTION_TO_FIELD = {}
for field_key, (scale_base, description) in SYSTEM_FIELD_MAPPING.items():

    DESCRIPTION_TO_FIELD[description] = field_key
    if "/" in description:
        for part in description.split("/"):
            part = part.strip()
            if part and part not in DESCRIPTION_TO_FIELD:
                DESCRIPTION_TO_FIELD[part] = field_key

def get_system_field_key_from_description(description: str) -> str | None:
    """
    Helper function to get the system field key from a description.
    Returns the exact field key (e.g., "SOURCE_LOC_ZONE") from SYSTEM_FIELD_MAPPING.
    """
    return DESCRIPTION_TO_FIELD.get(description)

@spa_upload_tracer.tool(name="select_rate_card_fields", description="Matches Excel rate card columns with DHL system field descriptions")
def select_rate_card_fields(file_path: str) -> str:
    """Matches Excel columns with DHL field descriptions."""
    logger.info(f"Tool select_rate_card_fields called for: {file_path}")

    # Get current span for tracing
    span = get_current_span()

    try:
        # Validate file path
        if not file_path or not isinstance(file_path, str):
            error_msg = "Error: Invalid file path provided."
            if span and span.is_recording():
                span.set_attribute("output.error", error_msg)
                span.set_status(Status(StatusCode.ERROR, error_msg))
            return error_msg

        # Normalize file path (handle spaces and special characters)
        file_path = os.path.abspath(os.path.expanduser(file_path))

        # Check if file exists
        if not os.path.exists(file_path):
            error_msg = f"Error: File not found at path: {file_path}"
            logger.error(error_msg)
            if span and span.is_recording():
                span.set_attribute("output.error", error_msg)
                span.set_status(Status(StatusCode.ERROR, error_msg))
            return error_msg

        # Validate file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in ['.xls', '.xlsx']:
            error_msg = (
                f"Error: Invalid file format. Expected Excel file (.xls or .xlsx), "
                f"but received: {file_ext}. Please upload a valid Excel document."
            )
            logger.error(error_msg)
            if span and span.is_recording():
                span.set_attribute("output.error", error_msg)
                span.set_status(Status(StatusCode.ERROR, error_msg))
            return error_msg

        # Check file size (basic corruption check)
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            error_msg = "Error: The uploaded file is empty or corrupted. Please upload a valid Excel file."
            logger.error(error_msg)
            if span and span.is_recording():
                span.set_attribute("output.error", error_msg)
                span.set_status(Status(StatusCode.ERROR, error_msg))
            return error_msg

        # Check for problematic characters in filename (log warning but proceed)
        filename = os.path.basename(file_path)
        problematic_chars = ['<', '>', ':', '"', '|', '?', '*']
        if any(char in filename for char in problematic_chars):
            logger.warning(f"Filename contains special characters that may cause issues: {filename}")

        # Trace: Log file validation
        if span and span.is_recording():
            span.set_attribute("input.file_path", file_path)
            span.set_attribute("input.file_size", file_size)
            span.set_attribute("input.file_extension", file_ext)
            span.set_attribute("input.filename", filename)

        # Load Excel with better error handling.
        # We treat the **third physical row in Excel (index 2)** as the header row,
        # mirroring the behavior of your working prototype:
        #     df = pd.read_excel(file_path, header=2)
        try:
            df = pd.read_excel(file_path, header=2, engine=None)  # engine=None lets pandas auto-detect
        except Exception as read_error:
            # More specific error handling for Excel reading
            error_type = type(read_error).__name__
            logger.error(f"Failed to read Excel file: {error_type}: {read_error}")

            # Provide user-friendly error message
            if "No such file" in str(read_error) or "FileNotFoundError" in error_type:
                error_msg = f"Error: File not found or cannot be accessed: {file_path}"
            elif "Permission" in error_type or "permission" in str(read_error).lower():
                error_msg = f"Error: Permission denied. Cannot read file: {file_path}"
            elif "corrupt" in str(read_error).lower() or "invalid" in str(read_error).lower():
                error_msg = (
                    "Error: The uploaded file appears to be corrupted or contains an invalid format. "
                    "Please verify the file is a valid Excel document (.xls or .xlsx) and try uploading again."
                )
            elif "xlrd" in str(read_error).lower() or "openpyxl" in str(read_error).lower():
                error_msg = (
                    "Error: Unable to parse Excel file. The file may be corrupted, password-protected, "
                    "or in an unsupported format. Please ensure the file is a valid Excel document (.xls or .xlsx)."
                )
            else:
                error_msg = (
                    f"Error: Failed to read Excel file. {str(read_error)}. "
                    "Please verify the file is a valid Excel document (.xls or .xlsx) and not corrupted."
                )

            if span and span.is_recording():
                span.set_attribute("output.error", error_msg)
                span.set_attribute("output.error_type", error_type)
                span.set_status(Status(StatusCode.ERROR, error_msg))

            return error_msg

        # Extract cleaned column names, ignoring empty, 'Unnamed', and 'nan' headers
        cols = [
            str(c).strip()
            for c in df.columns.tolist()
            if str(c).strip()
               and not str(c).startswith("Unnamed")
               and str(c).lower() != "nan"
        ]

        # Trace: Log input parameters
        if span and span.is_recording():
            span.set_attribute("input.file_path", file_path)
            span.set_attribute("input.total_columns", len(cols))
            span.set_attribute("input.columns", json.dumps(cols[:50]))  # First 50 columns
            span.set_attribute("input.field_descriptions_count", len(FIELD_DESCRIPTIONS))
            span.set_attribute("input.field_descriptions_sample", json.dumps(FIELD_DESCRIPTIONS[:20]))  # First 20 descriptions

        logger.info(f"Found {len(cols)} columns in Excel file")
        logger.debug(f"FIELD_DESCRIPTIONS count: {len(FIELD_DESCRIPTIONS)}")
        logger.debug(f"Sample FIELD_DESCRIPTIONS: {FIELD_DESCRIPTIONS[:10]}")

        if not cols:
            if span and span.is_recording():
                span.set_attribute("output.error", "No valid columns found")
                span.set_status(Status(StatusCode.ERROR, "No valid columns found"))
            return "Error: No valid columns found in the Excel file. Please check the file format."

        # Create a structured list of field mappings for the LLM
        # Format: "FIELD_KEY: Description"
        field_mappings = []
        for field_key, (scale_base, description) in SYSTEM_FIELD_MAPPING.items():
            field_mappings.append(f"{field_key}: {description}")

        # Limit to first 500 fields to avoid token limits (prioritize most relevant)
        # You can adjust this or use a smarter filtering mechanism
        field_mappings_str = "\n".join(field_mappings[:500])

        prompt = f"""
        Analyze these Excel column names: {cols}

        Match them against these DHL system field mappings (format: FIELD_KEY: Description):
        {field_mappings_str}

        MATCHING RULES (MANDATORY):
         - You MUST map an Excel column when there is a clear semantic equivalence between the column name and a DHL field description, even if the words differ slightly.
         - If there is no clear semantic match, DO NOT map that Excel column.
         - NEVER return null, empty strings, or placeholder values.
         - NEVER include obviously weak or purely guessed matches.


        OUTPUT FORMAT (JSON ONLY):
        {{
          "matched_fields": [
            {{
              "excel_column": "<exact Excel column name>",
              "system_field_key": "<exact DHL system field key>"
            }}
          ]
        }}


        Weight ranges, FTL, pallet counts, pricing buckets, LOT, SHADOW, etc.
        usually do NOT have DHL system field mappings — OMIT THEM COMPLETELY from matched_fields.
        If there are no valid matches, return:
         {{ "matched_fields": [] }}
        """

        # Build Azure OpenAI payload to mirror the working Postman request
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        print("here")
        print(payload)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "api-key": AZURE_OPENAI_API_KEY,
        }

        # Make API request to Azure OpenAI
        response = requests.post(
            AZURE_OPENAI_CHAT_URL,
            headers=headers,
            json=payload,
            timeout=900,
        )
        if response.status_code >= 400:
            logger.error(
                "Azure OpenAI HTTP %s error response (field selector): %s",
                response.status_code,
                response.text,
            )
            response.raise_for_status()

        resp = response.json()
        logger.debug(f"Azure OpenAI API response: {resp}")

        # Azure OpenAI chat completions: choices[0].message.content is a JSON string per your spec
        choices = resp.get("choices", [])
        if not choices:
            raise ValueError("Azure OpenAI response missing 'choices'")

        message = choices[0].get("message", {}) or {}
        content_str = message.get("content", "{}") or "{}"

        # Strip possible markdown code fences like ```json ... ``` to get clean JSON.
        stripped = content_str.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            content_str = "\n".join(lines).strip()

        # Parse the model response content as JSON, with robust fallback handling.
        try:
            # First, try to parse the content directly as JSON
            data = json.loads(content_str)
        except json.JSONDecodeError as e:
            logger.warning(
                "Model content is not clean JSON, attempting to extract JSON object. "
                "Error: %s | Raw content: %r",
                e,
                content_str,
            )

            # Try to locate a JSON object substring within the content
            json_candidate = None
            try:
                start = content_str.index("{")
                end = content_str.rindex("}") + 1
                json_candidate = content_str[start:end]
                data = json.loads(json_candidate)
            except Exception as inner_e:
                # If we still cannot parse, raise a clear error so the caller
                # gets a meaningful message instead of a cryptic JSON error.
                error_msg = (
                    "Model returned a non-JSON or malformed JSON response for field mapping. "
                    "Please retry the operation. "
                    f"Underlying error: {str(inner_e)}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg) from inner_e

        # Extract message content
        matched_cols = data.get("matched_fields", [])
        print(matched_cols)
        # Include weight patterns (e.g. 30-50, FTL)
        weight_pattern = re.compile(r'^(\d+-\d+|\d+\s*-\s*\d+|>\s*\d+|FTL|1/2\s*FTL)', re.IGNORECASE)
        matched_weight_cols = []

        for c in cols:
            if weight_pattern.match(c) and c not in matched_weight_cols:
                matched_weight_cols.append(c)

        # Preserve original Excel order for weight columns
        ordered_weight_cols = [c for c in cols if c in matched_weight_cols]

        # Extract system_field_keys
        system_field_keys = [
            item["system_field_key"]
            for item in matched_cols
            if "system_field_key" in item
        ]

        # Combine system fields + weight slabs
        final_fields = system_field_keys + ordered_weight_cols

        # Build and store column mapping: system_field_key -> excel_column (for SAP upload)
        column_mapping = {}
        for item in matched_cols:
            if "system_field_key" in item and "excel_column" in item:
                column_mapping[item["system_field_key"]] = item["excel_column"]
        for col in ordered_weight_cols:
            column_mapping[col] = col  # Weight bracket columns: key = Excel header
        # If Excel has a CURRENCY column, add it so SAP upload can resolve from mapping
        for col in cols:
            if str(col).strip().upper() in ("CURRENCY", "CURRENCY CODE", "CCY") and "CURRENCY" not in column_mapping:
                column_mapping["CURRENCY"] = col
                break
        session_id = get_session_id()
        # Remember both the column mapping and the Excel path for this session
        store_column_mapping(session_id, column_mapping)
        store_rate_card_path(session_id, file_path)
        logger.info(
            f"Stored column mapping and rate card path for session {session_id}: "
            f"{len(column_mapping)} entries, path={file_path}"
        )

        # Build response string
        return json.dumps(final_fields)

    except Exception as e:
        logger.error(f"Field selection error: {e}")
        return f"Error matching fields: {str(e)}"

field_selection_tool = FunctionTool(select_rate_card_fields)