import logging
import os

from google.adk.agents import Agent
from google.adk.apps.app import App
from google.adk.models import Gemini
from google.adk.tools import (
    AgentTool,
    FunctionTool,  # noqa: F401
    google_search,  # built-in Google Search tool
)
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = os.getenv("MODEL", "gemini-2.5-flash")
logger.info(f"Using model: {model}")

class SearchAgent(Agent):
    """Subclass to fix ADK app name mismatch warning."""
    pass


class RootAgent(Agent):
    """Subclass to fix ADK app name mismatch warning."""
    pass


search_agent = SearchAgent(
    model=model,
    name="SearchAgent",
    description="Agent to perform Google Search",
    instruction="You're a specialist in Google Search",
    tools=[google_search],
)

root_agent = RootAgent(
    name="root_agent",
    description="You are a helpful AI assistant designed to provide accurate and useful information",
    model=Gemini(
        model=model,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction="""You are a helpful AI assistant designed to provide accurate and useful information.
    If you don't know the answer, use the SearchAgent to perform a Google search.""",
    tools=[AgentTool(agent=search_agent)],
)

app = App(root_agent=root_agent, name="app")
