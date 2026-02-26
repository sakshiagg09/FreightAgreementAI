import json
import os
import logging
import re
from typing import Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, APIRouter
from google.genai import types  # type: ignore
from agent.logistics_agent import runner, session_service
from utils.session_context import (
    set_session_id,
    store_rate_card_path,
    set_field_selection_retry_requested,
    get_field_selection_result,
    store_field_selection_result,
    store_column_mapping,
    store_unselected_field_mappings,
    get_unselected_field_mappings,
)
from utils.llm_usage import reset_llm_usage, get_llm_usage
from utils.system_field_mapping import SYSTEM_FIELD_MAPPING
from config.dev_config import TEMP_UPLOADS_DIR, APP_NAME, DEFAULT_USER_ID

logger = logging.getLogger(__name__)

# Define the router for the logistics API
router = APIRouter()

def _parse_selected_field_keys(value: Optional[str]) -> set[str]:
    """Parse selected_field_keys from JSON array or comma-separated string. Returns set of system_field_key."""
    if not value or not value.strip():
        return set()
    raw = value.strip()
    # Handle form values like ["TSP","LANE"] or [\"TSP\",\"LANE\"] (Postman/form escaped quotes)
    if raw.startswith("["):
        normalized = raw.replace('\\"', '"')
        try:
            keys = json.loads(normalized)
            return set(str(k).strip() for k in keys if k)
        except (json.JSONDecodeError, TypeError):
            pass
        # Fallback: strip [ ], split by ",", strip quotes from each
        inner = raw.strip("[]").strip()
        if inner:
            parts = [p.strip().strip('"').strip("'").strip() for p in inner.split(",") if p.strip()]
            return set(p for p in parts if p)
    return set(k.strip().strip('"').strip("'") for k in raw.split(",") if k.strip())


@router.post("/invoke")
async def invoke(
    session_id: str = Form(...),
    message: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    selected_field_keys: Optional[str] = Form(None),
):
    user_id = DEFAULT_USER_ID
    actual_message = message or ""
    
    if file:
        try:
            os.makedirs(TEMP_UPLOADS_DIR, exist_ok=True)
            
            # Sanitize filename to handle special characters and spaces
            original_filename = file.filename or "uploaded_file"
            # Replace problematic characters but keep spaces (they're usually fine)
            # Only remove truly problematic characters: < > : " | ? * \
            sanitized_name = re.sub(r'[<>:"|?*\\]', '_', original_filename)
            # Ensure filename is not empty after sanitization
            if not sanitized_name or sanitized_name.strip() == '':
                sanitized_name = "uploaded_file"
            
            # Preserve file extension
            file_ext = os.path.splitext(original_filename)[1]
            if not sanitized_name.endswith(file_ext):
                sanitized_name = os.path.splitext(sanitized_name)[0] + file_ext
            
            file_path = os.path.abspath(os.path.join(TEMP_UPLOADS_DIR, f"{session_id}_{sanitized_name}"))
            
            # Validate file size before writing (basic check)
            file_content = await file.read()
            if len(file_content) == 0:
                logger.warning(f"Uploaded file '{original_filename}' is empty")
                # Still save it, but log a warning
            
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            logger.info(f"File uploaded successfully: {file_path} (original: {original_filename})")
            # Store path for this session so batch_upload_to_sap can use it when called without args
            if file_ext.lower() in (".xls", ".xlsx"):
                store_rate_card_path(session_id, file_path)
            actual_message += f" [FILE_UPLOADED: {file_path}]"
        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}", exc_info=True)
            # Don't fail the entire request, but log the error
            actual_message += f" [FILE_UPLOAD_ERROR: Failed to save file - {str(e)}]"

    if not actual_message:
        actual_message = "Hello"

    # When user clicks Retry (e.g. "Retry" or "retry field selection"), set session flag
    # so select_rate_card_fields uses retry logic even if the agent doesn't pass enable_retries
    _msg_lower = actual_message.strip().lower()
    if _msg_lower in ("retry", "retry field selection", "retry with retries") or (
        _msg_lower.startswith("retry") and len(actual_message.strip()) < 80
    ):
        set_field_selection_retry_requested(session_id)

    # When user clicks Validate with a subset of checkboxes, keep only selected in column_mapping and store unselected
    unselected_field_mappings: Optional[list] = None
    validated_field_selection_result: Optional[dict[str, Any]] = None
    if (
        _msg_lower in ("validate", "validate mapping", "validate mappings")
        or (_msg_lower.startswith("validate") and len(actual_message.strip()) < 60)
    ) and selected_field_keys:
        selected_keys = _parse_selected_field_keys(selected_field_keys)
        if selected_keys:
            current = get_field_selection_result(session_id=session_id)
            # Tool may have stored under "default" when running in another thread; use as fallback
            if not current and session_id != "default":
                current = get_field_selection_result(session_id="default")
            # Build mapping list from stored result; support legacy storage that only has column_mapping_dict
            mapping_list = (current.get("column_mapping") or []) if current else []
            if not mapping_list and current and current.get("column_mapping_dict"):
                cm_dict = current.get("column_mapping_dict") or {}
                mapping_list = []
                for sys_key, excel_col in cm_dict.items():
                    if not sys_key:
                        continue
                    entry = SYSTEM_FIELD_MAPPING.get(sys_key)
                    description = entry[1] if entry else sys_key
                    mapping_list.append({
                        "excel_column": excel_col or "",
                        "system_field_key": sys_key,
                        "system_field_description": description,
                    })
            if current and mapping_list:
                selected_list = [m for m in mapping_list if (m.get("system_field_key") or "").strip() in selected_keys]
                unselected_list = [m for m in mapping_list if (m.get("system_field_key") or "").strip() not in selected_keys]
                if selected_list is not mapping_list:
                    new_dict = {m["system_field_key"]: m.get("excel_column") or "" for m in selected_list}
                    new_fields = (current.get("fields") or [])[:]
                    weight_cols = set(current.get("weight_columns") or [])
                    new_fields = [f for f in new_fields if f in new_dict or f in weight_cols]
                    store_column_mapping(session_id, new_dict)
                    store_field_selection_result(
                        session_id,
                        {
                            "column_mapping": selected_list,
                            "column_mapping_dict": new_dict,
                            "fields": new_fields,
                            "weight_columns": current.get("weight_columns") or [],
                        },
                    )
                    store_unselected_field_mappings(session_id, unselected_list)
                    unselected_field_mappings = unselected_list
                    validated_field_selection_result = {
                        "column_mapping": selected_list,
                        "column_mapping_dict": new_dict,
                        "fields": new_fields,
                        "weight_columns": current.get("weight_columns") or [],
                    }
                    logger.info(
                        "Validate with selection: kept %d mapping(s), stored %d unselected for session %s",
                        len(selected_list),
                        len(unselected_list),
                        session_id,
                    )

    final_response = ""
    try:
        set_session_id(session_id)
        reset_llm_usage(session_id)
        session = await session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
        if not session:
            await session_service.create_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)

        user_message = types.Content(role="user", parts=[types.Part.from_text(text=actual_message)])
        events = runner.run(user_id=user_id, session_id=session_id, new_message=user_message)

        # Extract field selection from tool response in events (session context may not propagate to tools)
        field_selection_result: Optional[dict[str, Any]] = None
        for event in events:
            if hasattr(event, "content") and hasattr(event.content, "parts"):
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        final_response += part.text
                    # Capture select_rate_card_fields tool output so we can return it for the UI
                    if hasattr(part, "function_response") and part.function_response:
                        name = getattr(part.function_response, "name", "") or ""
                        if name == "select_rate_card_fields":
                            raw = getattr(part.function_response, "response", None)
                            if isinstance(raw, str):
                                try:
                                    field_selection_result = json.loads(raw)
                                except json.JSONDecodeError:
                                    pass
                            elif isinstance(raw, dict):
                                field_selection_result = raw
            elif hasattr(event, "text"):
                final_response += event.text

        usage = get_llm_usage(session_id)
        # Prefer validated result from this request's Validate step, then from events, then session storage.
        if validated_field_selection_result is not None:
            field_selection_result = validated_field_selection_result
        elif field_selection_result is None:
            field_selection_result = get_field_selection_result(session_id=session_id)
        elif field_selection_result:
            store_field_selection_result(session_id, field_selection_result)
        # If tool/ADK returned {"result": "<json string>"}, parse the inner string so we expose real dicts
        if isinstance(field_selection_result, dict) and "result" in field_selection_result:
            raw_result = field_selection_result.get("result")
            if isinstance(raw_result, str) and raw_result.strip():
                try:
                    parsed = json.loads(raw_result)
                    if isinstance(parsed, dict):
                        field_selection_result = parsed
                        store_field_selection_result(session_id, field_selection_result)
                except json.JSONDecodeError:
                    pass
        # Include unselected list when we have it (e.g. after Validate with selection)
        if unselected_field_mappings is None:
            unselected_field_mappings = get_unselected_field_mappings(session_id=session_id)
        # Expose column_mapping_dict (and full field selection) as top-level "result" for clients
        result_payload = None
        if field_selection_result and isinstance(field_selection_result, dict):
            result_payload = field_selection_result.get("column_mapping_dict")
            if result_payload is None and field_selection_result:
                result_payload = field_selection_result
        return {
            "session_id": session_id,
            "response": final_response or "I'm ready to help. Please provide the document.",
            "result": result_payload,
            "unselected_field_mappings": unselected_field_mappings,
            "usage": usage,
        }
    except Exception as e:
        logger.error(f"Invoke API failed: {e}", exc_info=True)
        return {"error": str(e)}


