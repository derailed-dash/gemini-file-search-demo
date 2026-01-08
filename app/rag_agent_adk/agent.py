import functools
import logging
import os
from typing import Any

from google import genai
from google.adk.agents import Agent
from google.adk.apps.app import App
from google.adk.models import Gemini
from google.adk.tools import (
    AgentTool,
    google_search,
)
from google.genai import types

from .tools_custom import FileSearchTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_id = os.getenv("MODEL", "gemini-2.5-flash")
logger.info(f"Using model: {model_id}")

STORE_NAME = os.getenv("STORE_NAME")
logger.info(f"Store name: {STORE_NAME}")


class SearchAgent(Agent):
    """Subclass to fix ADK app name mismatch warning."""

    pass


class RootAgent(Agent):
    """Subclass to fix ADK app name mismatch warning."""

    pass


# Search Specialist Agent (Google Search only)
search_agent = SearchAgent(
    model=model_id,
    name="SearchAgent",
    description="Agent to perform Google Search",
    instruction="You're a specialist in Google Search",
    tools=[google_search],
)


@functools.cache
def get_store_name():
    """Retrieve the store name dynamically using a temp client."""
    try:
        # Assuming env vars are set aka GEMINI_API_KEY
        client = genai.Client()
        logger.info(f"Looking for File Search Store: {STORE_NAME}...")
        for a_store in client.file_search_stores.list():
            if a_store.display_name == STORE_NAME:
                logger.info(f"Found store: {a_store.name}")
                return a_store.name
    except Exception as e:
        logger.error(f"Error resolving store: {e}")
        return None


# Root Agent Configuration
def create_root_agent():
    store_name = get_store_name()

    tools: list[Any] = [AgentTool(agent=search_agent)]

    instruction = """You are a helpful AI assistant designed to provide accurate and useful information.
    If you don't know the answer to something, use the SearchAgent to perform a Google Search.
    """

    if store_name:
        logger.info(f"Attaching File Search Tool connected to {store_name}")
        # Add the custom FileSearchTool
        tools.append(FileSearchTool(file_search_store_names=[store_name]))

        instruction += """
        IMPORTANT: You MUST start by searching your reference materials
        using the 'file_search' tool for information relevant to the user's request.
        Always use the 'file_search' tool before answering bespoke questions about Dazbo or his ships.
        """
    else:
        logger.warning("No File Search Store found. Running without RAG.")

    return RootAgent(
        name="rag_agent_adk",
        description="A chatbot with access to Google Search and Bespoke File Search",
        model=Gemini(
            model=model_id,
            retry_options=types.HttpRetryOptions(attempts=3),
        ),
        instruction=instruction,
        tools=tools,
    )


root_agent = create_root_agent()
app = App(root_agent=root_agent, name="rag_agent_adk")
