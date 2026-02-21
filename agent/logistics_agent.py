"""
Logistics Agent - ADK-based agent for freight agreement and rate card workflows.
"""

import logging
from pathlib import Path

from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService

logger = logging.getLogger(__name__)

from .azure_openai_llm import AzureOpenAILlm
from .callbacks import before_model_callback, after_model_callback
from config.dev_config import AZURE_OPENAI_CHAT_URL, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
from tools.create_agreement_and_validity_async import create_agreement_and_validity_async_tool
from tools.sap_upload_tool import sap_upload_tool
from tools.fetch_charge_type_tool import resolve_charge_type_tool
from tools.select_dhl_fields import field_selection_tool
from tools.upload_aggreement import agreement_tool

_INSTRUCTION_PATH = Path(__file__).parent / "instruction.txt"


def _load_instruction() -> str:
    """Load agent instruction from instruction.txt."""
    return _INSTRUCTION_PATH.read_text(encoding="utf-8").strip()


# -----------------------------------------------------------------------------
# Agent setup
# -----------------------------------------------------------------------------

model_instance = AzureOpenAILlm(
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_CHAT_URL,
    AZURE_OPENAI_API_KEY,
)
session_service = InMemorySessionService()

logistics_agent = LlmAgent(
    name="LogisticsAgent",
    model=model_instance,
    instruction=_load_instruction(),
    tools=[
        agreement_tool,
        resolve_charge_type_tool,
        field_selection_tool,
        create_agreement_and_validity_async_tool,
        sap_upload_tool,
    ],
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
)

runner = Runner(agent=logistics_agent, session_service=session_service, app_name="LogisticsApp")
logger.info(
    "LogisticsAgent initialized with before_model_callback and after_model_callback app_name=LogisticsApp"
)
