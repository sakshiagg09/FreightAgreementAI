# Expose root_agent for ADK web (adk web discovers agent.root_agent)
from .logistics_agent import logistics_agent

root_agent = logistics_agent

__all__ = ["root_agent", "logistics_agent"]
