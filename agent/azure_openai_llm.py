"""
Azure OpenAI LLM - Custom ADK LLM implementation using Azure OpenAI with Tool Calling.
"""

import json
import logging
import re
import time
from typing import Any, AsyncGenerator

import requests
from google.adk.models import BaseLlm, LlmResponse
from google.genai import types

from config.dev_config import OLLAMA_TIMEOUT, MAX_RETRIES, RETRY_DELAY
from utils.llm_usage import add_llm_usage

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)
AZURE_TIMEOUT = OLLAMA_TIMEOUT

# -----------------------------------------------------------------------------
# Azure OpenAI LLM
# -----------------------------------------------------------------------------


class AzureOpenAILlm(BaseLlm):
    """Custom LLM implementation for ADK using Azure OpenAI with Tool Calling."""
    api_url: str
    api_key: str

    def __init__(self, model: str, api_url: str, api_key: str):
        super().__init__(model=model, api_url=api_url, api_key=api_key)

    # -------------------------------------------------------------------------
    # Message building helpers
    # -------------------------------------------------------------------------

    def _extract_system_instruction(self, request: Any) -> str:
        """Extract system instruction from request config."""
        if not (hasattr(request, "config") and request.config and hasattr(request.config, "system_instruction")):
            return ""
        instr = request.config.system_instruction
        if isinstance(instr, str):
            return instr
        if hasattr(instr, "parts"):
            return " ".join(p.text for p in instr.parts if hasattr(p, "text"))
        return ""

    def _content_part_to_text(self, part: Any) -> str | None:
        """Convert a content part to text. Returns None if not a text/tool part."""
        if hasattr(part, "text") and part.text:
            return part.text
        if hasattr(part, "function_response") and part.function_response:
            try:
                tool_name = getattr(part.function_response, "name", "tool")
                tool_resp = getattr(part.function_response, "response", None)
                tool_resp_str = json.dumps(tool_resp, ensure_ascii=False) if tool_resp is not None else ""
                return f"[Tool {tool_name} result]: {tool_resp_str}"
            except Exception:
                return str(part.function_response)
        return None

    def _build_messages(self, request: Any) -> list[dict]:
        """Build Azure OpenAI messages from ADK request."""
        messages = []

        # System instruction
        system_instruction = self._extract_system_instruction(request)
        if system_instruction:
            system_instruction += "\n\nAvailable tools are provided. If you need to extract information from a file path provided in the user message, YOU MUST call the appropriate tool."
            messages.append({"role": "system", "content": system_instruction})

        # Chat history
        if not hasattr(request, "contents"):
            return messages

        for content in request.contents:
            role = getattr(content, "role", "user")
            if role == "model":
                role = "assistant"

            parts_text = []
            if hasattr(content, "parts"):
                for part in content.parts:
                    text = self._content_part_to_text(part)
                    if text:
                        parts_text.append(text)

            if parts_text:
                messages.append({"role": role, "content": "\n".join(parts_text)})

        return messages

    def _get_params_schema(self, fd: Any) -> dict:
        """Get params schema from ADK function declaration (same as Ollama)."""
        params = getattr(fd, "parameters", None)
        if hasattr(params, "model_dump"):
            params = params.model_dump(exclude_none=True)
        if isinstance(params, dict) and params.get("type") == "object":
            return params
        return {"type": "object", "properties": {}}

    def _build_tools(self, request: Any) -> list[dict]:
        """Build Azure OpenAI tools array from ADK request config."""
        tools = []
        if not (hasattr(request, "config") and request.config and hasattr(request.config, "tools")):
            return tools

        for tool in request.config.tools:
            if not (hasattr(tool, "function_declarations") and tool.function_declarations):
                continue
            for fd in tool.function_declarations:
                params_schema = self._get_params_schema(fd)
                tools.append({
                    "type": "function",
                    "function": {
                        "name": fd.name,
                        "description": fd.description or "",
                        "parameters": params_schema,
                    },
                })
        return tools

    def _call_azure_api(self, payload: dict) -> dict:
        """Call Azure OpenAI API with retry logic."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "api-key": self.api_key,
        }

        for attempt in range(MAX_RETRIES + 1):
            if attempt > 0:
                wait_time = RETRY_DELAY * (2 ** (attempt - 1))
                logger.info(f"Retrying Azure OpenAI (attempt {attempt + 1}/{MAX_RETRIES + 1}) after {wait_time}s...")
                time.sleep(wait_time)

            logger.info(
                f"Calling Azure OpenAI with {len(payload['messages'])} messages "
                f"(attempt {attempt + 1}/{MAX_RETRIES + 1})..."
            )
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=AZURE_TIMEOUT,
                )
                if response.status_code >= 400:
                    logger.error("Azure OpenAI HTTP %s: %s", response.status_code, response.text)
                    response.raise_for_status()
                return response.json()
            except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout, requests.exceptions.Timeout):
                if attempt < MAX_RETRIES:
                    logger.warning(f"Azure OpenAI timeout (attempt {attempt + 1}), retrying...")
                    continue
                raise
            except requests.exceptions.ConnectionError:
                if attempt < MAX_RETRIES:
                    logger.warning(f"Azure OpenAI connection failed (attempt {attempt + 1}), retrying...")
                    continue
                raise

        raise Exception("Failed to get response from Azure OpenAI after retries")

    def _record_usage(self, data: dict) -> None:
        """Extract and accumulate token usage from Azure response; record tool_calls for this step (e.g. agreement tool)."""
        usage = data.get("usage") or {}
        tool_calls: list[str] = []
        try:
            msg = (data.get("choices") or [{}])[0].get("message") or {}
            for tc in msg.get("tool_calls") or []:
                fn = (tc or {}).get("function") or {}
                name = fn.get("name")
                if name:
                    tool_calls.append(name)
        except (IndexError, KeyError, TypeError):
            pass

        if isinstance(usage, dict):
            prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            completion = usage.get("completion_tokens") or usage.get("output_tokens") or 0
            total = usage.get("total_tokens") or (prompt + completion)
        else:
            prompt = completion = total = 0

        if not (prompt or completion or total):
            logger.warning(
                "Azure OpenAI response had no token usage (missing or zero). "
                "Response keys: %s; usage=%s",
                list(data.keys()),
                data.get("usage"),
            )

        add_llm_usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            tool_calls=tool_calls if tool_calls else None,
        )

    def _parse_azure_response(self, data: dict) -> LlmResponse:
        """Parse Azure OpenAI response into ADK LlmResponse."""
        choices = data.get("choices", [])
        if not choices:
            raise Exception("Azure OpenAI response missing 'choices'")

        msg = choices[0].get("message", {}) or {}
        text = msg.get("content", "") or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        parts = []
        if text:
            parts.append(types.Part.from_text(text=text))

        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            fn_name = fn.get("name")
            fn_args = fn.get("arguments", {})
            if isinstance(fn_args, str):
                try:
                    fn_args = json.loads(fn_args)
                except json.JSONDecodeError:
                    pass
            logger.info(f"Model requested tool call: {fn_name}({fn_args})")
            parts.append(types.Part(function_call=types.FunctionCall(name=fn_name, args=fn_args)))

        return LlmResponse(content=types.Content(role="model", parts=parts))

    def _error_response(self, message: str, error: Exception) -> LlmResponse:
        """Build error LlmResponse."""
        return LlmResponse(
            content=types.Content(role="model", parts=[types.Part.from_text(text=message)]),
            errorMessage=str(error),
        )

    def generate_content(self, request: Any, **kwargs) -> LlmResponse:
        """Generate content via Azure OpenAI with tool calling support."""
        try:
            messages = self._build_messages(request)
            tools = self._build_tools(request)

            payload = {"messages": messages, "temperature": 0.0}
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            data = self._call_azure_api(payload)
            self._record_usage(data)
            return self._parse_azure_response(data)

        except (requests.exceptions.Timeout, requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout) as e:
            connect_timeout, read_timeout = AZURE_TIMEOUT
            timeout_type = (
                "connection" if isinstance(e, requests.exceptions.ConnectTimeout)
                else "read" if isinstance(e, requests.exceptions.ReadTimeout)
                else "general"
            )
            logger.error(f"Azure OpenAI timed out ({timeout_type}) after {MAX_RETRIES + 1} attempts: {e}")
            error_msg = (
                f"Model request timed out ({timeout_type}) after {MAX_RETRIES + 1} attempts. "
                f"Connect: {connect_timeout}s, Read: {read_timeout}s. "
                f"Check Azure OpenAI health, network, and try smaller requests."
            )
            return self._error_response(error_msg, e)

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Azure OpenAI connection failed: {e}")
            return self._error_response(
                f"Could not connect to Azure OpenAI at {self.api_url}. Verify endpoint and network.",
                e,
            )

        except Exception as e:
            logger.error(f"Azure OpenAI generate failed: {e}", exc_info=True)
            return self._error_response(f"Error calling model: {str(e)}", e)

    async def generate_content_async(self, request: Any, **kwargs) -> AsyncGenerator[LlmResponse, None]:
        yield self.generate_content(request, **kwargs)
