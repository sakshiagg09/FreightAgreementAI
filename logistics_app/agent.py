# ADK web interface expects a module with root_agent.
# Expose the main logistics agent so "adk web" can discover and run it.
from agent.logistics_agent import logistics_agent

root_agent = logistics_agent
