import os
import logging
import re
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, APIRouter
from google.genai import types  # type: ignore
from agent.logistics_agent import runner, session_service
from utils.session_context import set_session_id, store_rate_card_path
from utils.llm_usage import reset_llm_usage, get_llm_usage
from config.dev_config import TEMP_UPLOADS_DIR, APP_NAME, DEFAULT_USER_ID

# Import Phoenix tracing
from utils.phoenix_tracing import AgentTracer, get_tracer
from opentelemetry.trace import get_current_span, Status, StatusCode

logger = logging.getLogger(__name__)

# Define the router for the logistics API
router = APIRouter()

@router.post("/invoke")
async def invoke(
    session_id: str = Form(...),
    message: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
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

    # Trace the API invocation itself - use context manager properly
    tracer = get_tracer()
    final_response = ""
    
    if tracer:
        # Create a span for the API invocation - this replaces the "invocation" unknown span
        with tracer.start_as_current_span(
            "spa_upload.api.invoke",
            openinference_span_kind="chain",
        ) as api_span_context:
            api_span = get_current_span()
            if api_span and api_span.is_recording():
                api_span.set_attribute("openinference.span.kind", "CHAIN")
                api_span.set_attribute("name", "spa_upload.api.invoke")
                api_span.set_attribute("display_name", "API Invoke")
                api_span.set_attribute("input.value", f"session_id={session_id}, message={actual_message}")
            
            try:
                # Set session context for tools to access
                set_session_id(session_id)
                reset_llm_usage(session_id)

                # Get or create session
                session = await session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
                if not session:
                    await session_service.create_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)

                # Wrap message for ADK Runner
                user_message = types.Content(role="user", parts=[types.Part.from_text(text=actual_message)])
                
                # Run agent with Phoenix tracing - use clear agent name
                with AgentTracer(agent_type="LogisticsAgent") as agent_tracer:
                    # Set input for agent span
                    agent_tracer.set_input(f"session_id={session_id}, message={actual_message}")
                    
                    events = runner.run(user_id=user_id, session_id=session_id, new_message=user_message)
                    
                    # Collect response
                    for event in events:
                        if hasattr(event, 'content') and hasattr(event.content, 'parts'):
                            for part in event.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    final_response += part.text
                        elif hasattr(event, 'text'):
                            final_response += event.text
                    
                    # Set output for agent span
                    agent_tracer.set_output(final_response or "I'm ready to help. Please provide the document.")
                
                # Set output and status for API span - MUST be set before context exits
                if api_span and api_span.is_recording():
                    api_span.set_attribute("output.value", final_response or "I'm ready to help. Please provide the document.")
                    api_span.set_status(Status(StatusCode.OK))

                usage = get_llm_usage(session_id)
                return {
                    "session_id": session_id,
                    "response": final_response or "I'm ready to help. Please provide the document.",
                    "usage": usage,
                }
            except Exception as e:
                logger.error(f"Invoke API failed: {e}", exc_info=True)
                # Set error status on API span - MUST be set before context exits
                if api_span and api_span.is_recording():
                    api_span.record_exception(e)
                    api_span.set_status(Status(StatusCode.ERROR))
                return {"error": str(e)}
    else:
        # No tracer available - execute without tracing
        try:
            set_session_id(session_id)
            reset_llm_usage(session_id)
            session = await session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
            if not session:
                await session_service.create_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)

            user_message = types.Content(role="user", parts=[types.Part.from_text(text=actual_message)])
            
            with AgentTracer(agent_type="LogisticsAgent") as agent_tracer:
                agent_tracer.set_input(f"session_id={session_id}, message={actual_message}")
                events = runner.run(user_id=user_id, session_id=session_id, new_message=user_message)
                
                for event in events:
                    if hasattr(event, 'content') and hasattr(event.content, 'parts'):
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                final_response += part.text
                    elif hasattr(event, 'text'):
                        final_response += event.text
                
                agent_tracer.set_output(final_response or "I'm ready to help. Please provide the document.")

            usage = get_llm_usage(session_id)
            return {
                "session_id": session_id,
                "response": final_response or "I'm ready to help. Please provide the document.",
                "usage": usage,
            }
        except Exception as e:
            logger.error(f"Invoke API failed: {e}", exc_info=True)
            return {"error": str(e)}


