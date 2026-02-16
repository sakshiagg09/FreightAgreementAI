"""
Phoenix Tracing and Observability Module
Provides tracing for LLM, tools, and agent with project_name "spa_upload".
Uses env-based config (FLASK_ENV), PHOENIX_COLLECTOR_ENDPOINT/PHOENIX_API_KEY,
and optional OpenInference SpanAttributes + add_attributes() for session/input.
"""
import os
import json
import logging
from typing import Optional, Any, Dict
from functools import wraps

logger = logging.getLogger(__name__)

# Lazy import to handle OpenTelemetry/Phoenix version incompatibilities
register = None
Status = None
StatusCode = None
get_current_span = None
SpanAttributes = None
_PHOENIX_AVAILABLE = False

try:
    from phoenix.otel import register
    from opentelemetry.trace import Status, StatusCode, get_current_span
    _PHOENIX_AVAILABLE = True
except ImportError as e:
    logger.warning(
        "Phoenix/OpenTelemetry import failed (%s). Tracing disabled. "
        "To enable tracing, upgrade: pip install 'opentelemetry-sdk>=1.39' 'opentelemetry-exporter-otlp-proto-grpc>=1.39'",
        e,
    )

try:
    from openinference.semconv.trace import SpanAttributes
except ImportError:
    # Fallback if openinference-semantic-conventions not installed
    class _SpanAttributes:
        INPUT_VALUE = "input.value"
        SESSION_ID = "session_id"
    SpanAttributes = _SpanAttributes

# Project name for Phoenix tracing
PROJECT_NAME = "spa_upload"

# ---------------------------------------------------------------------------
# Env-based Phoenix config (reference pattern: FLASK_ENV, llm_prod/llm_dev)
# ---------------------------------------------------------------------------
def _get_phoenix_config():
    """Resolve phoenix_url and phoenix_system_api_key from FLASK_ENV and config."""
    env = os.getenv("FLASK_ENV", "development")
    if env == "production":
        try:
            from config.llm_prod import phoenix_url, phoenix_system_api_key
            return phoenix_url, phoenix_system_api_key
        except ImportError:
            pass
    else:
        try:
            from config.llm_dev import phoenix_url, phoenix_system_api_key
            return phoenix_url, phoenix_system_api_key
        except ImportError:
            pass
    # Fallback: dev_config or env
    try:
        from config.dev_config import PHOENIX_URL, PHOENIX_SYSTEM_API_KEY
        return PHOENIX_URL, PHOENIX_SYSTEM_API_KEY or ""
    except ImportError:
        return os.getenv("PHOENIX_URL", "http://localhost:6006"), os.getenv("PHOENIX_SYSTEM_API_KEY", "")


# Initialize Phoenix tracer with project_name "spa_upload"
def initialize_phoenix_tracer():
    """
    Initialize Phoenix OpenTelemetry tracer with project_name "spa_upload".
    
    Returns:
        Tracer instance or None if initialization fails
    """
    tracer_provider = get_tracer_provider()
    if tracer_provider is None:
        return None
    
    try:
        tracer = tracer_provider.get_tracer(__name__)
        return tracer
    except Exception as e:
        logger.warning(f"Failed to get tracer: {e}. Tracing will be disabled.")
        return None

# Create a tracer instance for using decorators like tracer.tool, tracer.llm, etc.
_spa_upload_tracer = None

class NoOpTracer:
    """No-op tracer that provides the same interface but does nothing when tracing is disabled."""
    def tool(self, *args, **kwargs):
        def decorator(func):
            return func  # Return function unchanged
        return decorator
    
    def llm(self, *args, **kwargs):
        def decorator(func):
            return func  # Return function unchanged
        return decorator
    
    def agent(self, *args, **kwargs):
        def decorator(func):
            return func  # Return function unchanged
        return decorator

def get_spa_upload_tracer():
    """Get tracer instance for using decorators like tracer.tool(), tracer.llm(), tracer.agent()."""
    global _spa_upload_tracer
    if _spa_upload_tracer is None:
        tracer_provider = get_tracer_provider()
        if tracer_provider is not None:
            _spa_upload_tracer = tracer_provider.get_tracer("spa_upload_tracer")
        else:
            # Return no-op tracer if provider is not available
            _spa_upload_tracer = NoOpTracer()
    return _spa_upload_tracer

# Global tracer instance
_tracer = None
_tracer_provider = None

def get_tracer():
    """Get or initialize Phoenix tracer."""
    global _tracer
    if _tracer is None:
        _tracer = initialize_phoenix_tracer()
    return _tracer

def get_tracer_provider():
    """Get or initialize Phoenix tracer provider (reference pattern: endpoint and env vars)."""
    global _tracer_provider
    if not _PHOENIX_AVAILABLE:
        return None
    if _tracer_provider is None:
        try:
            phoenix_url, phoenix_system_api_key = _get_phoenix_config()
            os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = f"{phoenix_url.rstrip('/')}/projects"
            if phoenix_system_api_key:
                os.environ["PHOENIX_API_KEY"] = phoenix_system_api_key

            _tracer_provider = register(
                project_name=PROJECT_NAME,
                endpoint=f"{phoenix_url.rstrip('/')}/v1/traces",
            )
            logger.info(
                f"Phoenix tracer provider initialized: project_name={PROJECT_NAME}, endpoint={phoenix_url}/v1/traces"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Phoenix tracer provider: {e}. Tracing will be disabled.")
            _tracer_provider = None
    return _tracer_provider


def add_attributes(
    uuid: Optional[str] = None,
    email_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    query: Optional[str] = None,
    data: Optional[Any] = None,
) -> None:
    """
    Set custom attributes on the current span (session, workflow, input).
    Use inside a traced span (e.g. tool, LLM, or agent).
    """
    if not _PHOENIX_AVAILABLE or get_current_span is None:
        return
    span = get_current_span()
    if not span.is_recording():
        return
    if data is not None:
        data_json = json.dumps(data) if not isinstance(data, str) else data
        span.set_attribute(getattr(SpanAttributes, "INPUT_VALUE", "input.value"), data_json)
    for key, value in (
        ("uuid", uuid),
        ("email_id", email_id),
        ("workflow_id", workflow_id),
        ("query", query),
    ):
        if value is not None:
            span.set_attribute(key, value)
    if uuid is not None:
        span.set_attribute(getattr(SpanAttributes, "SESSION_ID", "session_id"), str(uuid))


# Optional: instrument OpenAI if openinference-instrumentation-openai is installed
def _instrument_openai_if_available():
    if not _PHOENIX_AVAILABLE:
        return
    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor
        tracer_provider = get_tracer_provider()
        if tracer_provider is not None:
            OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
            logger.info("OpenAI instrumentation enabled for Phoenix tracing.")
    except ImportError:
        pass


# Get the tracer for decorators - this is the main tracer instance
spa_upload_tracer = None

def _get_decorator_tracer():
    """Get tracer instance for decorators."""
    global spa_upload_tracer
    if spa_upload_tracer is None:
        spa_upload_tracer = get_spa_upload_tracer()
    return spa_upload_tracer

# Legacy decorator for LLM calls - kept for backward compatibility
def trace_llm(func=None, *, name: Optional[str] = None):
    """
    Decorator to trace LLM calls with @spa_upload.llm annotation.
    
    Usage:
        @trace_llm
        def generate_content(self, request):
            ...
        
        or
        
        @trace_llm(name="custom_llm_name")
        def generate_content(self, request):
            ...
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            if tracer is None:
                logger.debug("Phoenix tracer not available, skipping LLM tracing")
                return f(*args, **kwargs)
            
            # Extract model name if available
            model_name = "unknown"
            if args and hasattr(args[0], "model"):
                model_name = args[0].model
            elif "model" in kwargs:
                model_name = kwargs["model"]
            
            # Create descriptive span name with model
            if name:
                span_name = name
            else:
                # Use model name in span name for clarity
                model_short = model_name.split(":")[0] if ":" in model_name else model_name
                span_name = f"{PROJECT_NAME}.llm.{model_short}" if model_name != "unknown" else f"{PROJECT_NAME}.llm"
            
            logger.info(f"🤖 Creating LLM span: {span_name} with model: {model_name}")
            
            with tracer.start_as_current_span(
                span_name,
                openinference_span_kind="llm",
            ):
                span = get_current_span()
                
                # Set input attributes
                if span.is_recording():
                    # Set OpenInference LLM attributes
                    span.set_attribute("llm.model_name", model_name)
                    span.set_attribute("llm.provider", "ollama")
                    span.set_attribute("openinference.span.kind", "LLM")
                    # Set clear name attribute for filtering - this is critical for Phoenix to show the name
                    span.set_attribute("name", span_name)
                    span.set_attribute("llm.name", span_name)
                    # Also set as display name
                    span.set_attribute("display_name", f"LLM ({model_name})")
                    
                    # Try to extract input messages if available
                    input_value = None
                    if args and len(args) > 1:
                        try:
                            request = args[1]
                            # Try to extract messages from request
                            messages_list = []
                            if hasattr(request, "contents"):
                                for content in request.contents:
                                    role = getattr(content, 'role', 'user')
                                    if role == 'model':
                                        role = 'assistant'
                                    if hasattr(content, 'parts'):
                                        for part in content.parts:
                                            if hasattr(part, 'text') and part.text:
                                                messages_list.append({"role": role, "content": part.text[:500]})
                            if messages_list:
                                input_value = json.dumps(messages_list, indent=2)[:2000]
                            else:
                                input_value = str(request)[:2000]
                        except Exception as e:
                            logger.debug(f"Failed to extract input: {e}")
                            if args:
                                input_value = str(args[1])[:2000] if len(args) > 1 else str(args)[:2000]
                    
                    if input_value:
                        span.set_attribute(getattr(SpanAttributes, "INPUT_VALUE", "input.value"), input_value)
                
                try:
                    # Call the original function
                    result = f(*args, **kwargs)
                    
                    # Extract response details if available
                    if span.is_recording():
                        if hasattr(result, "content"):
                            output_value = str(result.content)[:2000]
                            span.set_attribute("output.value", output_value)
                            span.set_attribute("llm.response_length", len(str(result.content)))
                        # Explicitly set OK status - this is critical for Phoenix to show status
                        span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    if span.is_recording():
                        span.record_exception(e)
                        # Explicitly set ERROR status
                        span.set_status(Status(StatusCode.ERROR))
                    raise
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)

# Legacy decorator for tool calls - kept for backward compatibility
def trace_tool(func=None, *, name: Optional[str] = None, description: Optional[str] = None):
    """
    Decorator to trace tool calls with @spa_upload.tool annotation.
    
    Usage:
        @trace_tool
        def extract_agreement_info(file_path: str):
            ...
        
        or
        
        @trace_tool(name="custom_tool", description="Tool description")
        def extract_agreement_info(file_path: str):
            ...
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            if tracer is None:
                logger.debug("Phoenix tracer not available, skipping tool tracing")
                return f(*args, **kwargs)
            
            tool_name = name or f.__name__
            tool_description = description or (f.__doc__ or "No description")
            
            # Create descriptive span name with actual tool name
            span_name = f"{PROJECT_NAME}.tool.{tool_name}"
            logger.info(f"🔧 Creating tool span: {span_name} for tool: {tool_name}")
            
            with tracer.start_as_current_span(
                span_name,
                openinference_span_kind="tool",
            ):
                span = get_current_span()
                
                # Set tool attributes
                if span.is_recording():
                    # Set OpenInference Tool attributes
                    span.set_attribute("openinference.span.kind", "TOOL")
                    span.set_attribute("tool.name", tool_name)
                    span.set_attribute("tool.description", tool_description)
                    # Set clear name attribute for filtering - this is critical for Phoenix to show the name
                    span.set_attribute("name", span_name)
                    span.set_attribute("tool.full_name", span_name)
                    # Also set as display name
                    span.set_attribute("display_name", tool_name)
                    
                    # Set input - format as JSON for better readability
                    try:
                        input_dict = {}
                        if args:
                            # Skip 'self' if it's a method
                            arg_list = list(args)
                            if arg_list and hasattr(arg_list[0], '__class__') and arg_list[0].__class__.__name__ == f.__qualname__.split('.')[0]:
                                arg_list = arg_list[1:]  # Skip self
                            input_dict["args"] = [str(arg)[:500] for arg in arg_list]
                        if kwargs:
                            input_dict["kwargs"] = {k: str(v)[:500] for k, v in kwargs.items()}
                        input_str = json.dumps(input_dict, indent=2)[:2000]
                    except:
                        input_str = str(args) + str(kwargs)
                        if len(input_str) > 2000:
                            input_str = input_str[:2000] + "..."
                    
                    span.set_attribute(getattr(SpanAttributes, "INPUT_VALUE", "input.value"), input_str)
                    span.set_attribute("tool.parameters", input_str)
                
                try:
                    result = f(*args, **kwargs)
                    
                    # Set output
                    if span.is_recording():
                        result_str = str(result)
                        output_str = result_str[:2000] if len(result_str) <= 2000 else result_str[:2000] + "..."
                        span.set_attribute("output.value", output_str)
                        span.set_attribute("tool.output_length", len(result_str))
                        span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    if span.is_recording():
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR))
                    raise
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)

# Context manager for agent tracing with @spa_upload.agent annotation
class AgentTracer:
    """
    Context manager for tracing agent execution with @spa_upload.agent annotation.
    
    Usage:
        with AgentTracer() as tracer:
            result = runner.run(...)
        
        or
        
        with AgentTracer(name="LogisticsAgent") as tracer:
            result = runner.run(...)
    """
    def __init__(self, name: Optional[str] = None, agent_type: Optional[str] = None):
        # Use provided name, or construct from agent_type, or use default
        self.agent_type = agent_type or "LogisticsAgent"
        if name:
            self.name = name
        else:
            # Use clear, descriptive name: "LogisticsAgent" or "spa_upload.agent.LogisticsAgent"
            self.name = self.agent_type  # Simple name like "LogisticsAgent"
        self.tracer = get_tracer()
        self.span_context = None
    
    def __enter__(self):
        if self.tracer is None:
            return self
        
        self.span_context = self.tracer.start_as_current_span(
            self.name,
            openinference_span_kind="agent",
        )
        self.span_context.__enter__()
        
        # Get the current span and set agent attributes
        span = get_current_span()
        if span.is_recording():
            span.set_attribute("openinference.span.kind", "AGENT")
            span.set_attribute("agent.name", self.agent_type)
            span.set_attribute("agent.type", "LlmAgent")
            # Also set the name as an attribute for filtering - this is critical for Phoenix to show the name
            span.set_attribute("name", self.name)
            span.set_attribute("display_name", self.agent_type)
        
        return self
    
    def set_input(self, input_value: Any):
        """Set input for the agent span."""
        if self.tracer is None:
            return
        span = get_current_span()
        if span.is_recording():
            input_str = str(input_value)
            if len(input_str) > 2000:
                input_str = input_str[:2000] + "..."
            span.set_attribute(getattr(SpanAttributes, "INPUT_VALUE", "input.value"), input_str)
    
    def set_output(self, output_value: Any):
        """Set output for the agent span."""
        if self.tracer is None:
            return
        span = get_current_span()
        if span.is_recording():
            output_str = str(output_value)
            if len(output_str) > 2000:
                output_str = output_str[:2000] + "..."
            span.set_attribute("output.value", output_str)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span_context:
            span = get_current_span()
            if span and span.is_recording():
                # Set status BEFORE exiting the context
                if exc_type:
                    span.record_exception(exc_val)
                    span.set_status(Status(StatusCode.ERROR))
                else:
                    # Explicitly set OK status - this is critical for Phoenix to show status
                    span.set_status(Status(StatusCode.OK))
            # Exit the context - this will finalize the span
            self.span_context.__exit__(exc_type, exc_val, exc_tb)
        
        return False  # Don't suppress exceptions

# Initialize tracer on module import and export for direct use
# Usage: @spa_upload_tracer.tool(name="Tool Name")
#        @spa_upload_tracer.llm
#        @spa_upload_tracer.agent
try:
    _tracer_provider = get_tracer_provider()
    _tracer = initialize_phoenix_tracer()
    _spa_upload_tracer = get_spa_upload_tracer()
    _instrument_openai_if_available()
    # Export the tracer for direct use like: @spa_upload_tracer.tool(name="Tool Name")
    # This will always be available (NoOpTracer if tracing is disabled)
    spa_upload_tracer = _spa_upload_tracer
    if _tracer_provider is not None:
        logger.info("Phoenix tracer exported as 'spa_upload_tracer' - use @spa_upload_tracer.tool(), @spa_upload_tracer.llm(), etc.")
    else:
        logger.warning("Phoenix tracer provider not available - using no-op tracer (decorators will not trace)")
except Exception as e:
    logger.warning(f"Phoenix initialization failed: {e}. Using no-op tracer.")
    _tracer = None
    spa_upload_tracer = NoOpTracer()
