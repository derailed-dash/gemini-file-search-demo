"""
This module defines a multi-agent system using the Google Agent Development Kit (ADK).

It implements a root agent ('rag_agent_adk') that orchestrates two specialized agents
wrapped as tools:
1. SearchAgent: A specialist for general information using Google Search.
2. RagAgent: A specialist for bespoke information using Gemini File Search.

Note: Sub-agents are wrapped in 'AgentTool' and passed via the 'tools' parameter rather
than the 'sub_agents' list. This is a required workaround in the ADK to support the use
of built-in tools (like google_search) within the agent hierarchy, which avoids the
"Tool use with function calling is unsupported" error. This is the "Agent-as-a-Tool"
pattern.
"""
import logging
import os

from google import genai
from google.adk.agents import Agent
from google.adk.tools import AgentTool, google_search
from google.genai import types

from .tools_custom import FileSearchTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_id = os.getenv("MODEL", "gemini-2.5-flash")
logger.info(f"Using model: {model_id}")

STORE_NAME = os.getenv("STORE_NAME")
logger.info(f"Store name: {STORE_NAME}")


# Search Specialist Agent (Google Search only)
search_agent = Agent(
    model=model_id,
    name="SearchAgent",
    description="Agent to perform Google Search",
    instruction="You're a specialist in Google Search",
    tools=[google_search],
)


def get_store_name():
    """Retrieve the store name dynamically using a temp client."""
    client = genai.Client()

    if not client:
        logger.error("Could not initialize GenAI client.")
        return None

    try:
        if not STORE_NAME:
            logger.warning("STORE_NAME env var is not set.")
            return None

        logger.info(f"Looking for File Search Store: {STORE_NAME}...")
        for a_store in client.file_search_stores.list():
            if a_store.display_name == STORE_NAME:
                logger.info(f"Found store: {a_store.name}")
                return a_store.name
        logger.warning(f"Store '{STORE_NAME}' not found.")
    except Exception as e:
        logger.error(f"Error resolving store: {e}")

    return None


# RAG Specialist Agent (File Search only)
def create_rag_agent() -> Agent:
    store_name = get_store_name()
    if store_name:
        logger.info(f"Creating RagAgent connected to {store_name}")
        instruction = """Use the file_search tool to retrieve information from the knowledge base."""

        return Agent(
            model=model_id,
            name="RagAgent",
            description="Agent to perform retrieval from bespoke file search store",
            instruction=instruction,
            tools=[FileSearchTool(file_search_store_names=[store_name])],
        )
    else:
        logger.warning("No File Search Store found. RagAgent will not be available.")
        return None


# Root Agent Configuration
def create_root_agent():

    instruction = """You are a helpful AI assistant designed to provide accurate and useful information.
    You have access to two specialist agents:
    1. RagAgent: For bespoke information from the internal knowledge base.
    2. SearchAgent: For general information from Google Search.

    Always try the RagAgent first. If this fails to yield a useful answer, then try the SearchAgent.
    """

    tools = [AgentTool(agent=search_agent)]
    rag_agent = create_rag_agent()
    if rag_agent:
        tools.append(AgentTool(agent=rag_agent))

    return Agent(
        name="rag_agent_adk",
        description="A chatbot with access to Google Search and Bespoke File Search",
        model=model_id,
        instruction=instruction,
        tools=tools,
        generate_content_config=types.GenerateContentConfig(temperature=1, top_p=1, max_output_tokens=8192),
    )

root_agent = create_root_agent()
