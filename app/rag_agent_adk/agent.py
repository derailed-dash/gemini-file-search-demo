import functools
import logging
import os

from google import genai
from google.adk.agents import Agent
from google.adk.tools import (
    AgentTool,
    google_search,
)
from google.genai import types

from .tools_custom import FileSearchTool

logging.getLogger("google_adk").setLevel(logging.ERROR)
logging.getLogger("google_genai").setLevel(logging.ERROR)
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


@functools.cache
def get_store_name():
    """Retrieve the store name dynamically using a temp client."""
    client = None
    try:
        # Try default client (supports API_KEY)
        client = genai.Client()
    except Exception:
        pass

    if not client:
        logger.error("Could not initialize GenAI client (checked API_KEY and Vertex credentials).")
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


# Root Agent Configuration
def create_root_agent():
    store_name = get_store_name()

    # tools = [AgentTool(agent=search_agent)]
    tools = []

    instruction = """You are a helpful AI assistant designed to provide accurate and useful information.
    If you don't know the answer to something, use the SearchAgent to perform a Google Search.
    """

    if store_name:
        logger.info(f"Attaching File Search Tool connected to {store_name}")

        instruction += """
        IMPORTANT: You MUST start by searching your reference materials
        using the 'file_search' tool for information relevant to the user's request.
        Always use the 'file_search' tool before answering.
        """

        tools.append(FileSearchTool(file_search_store_names=[store_name]))
    else:
        logger.warning("No File Search Store found. Running without RAG.")

    logger.info(f"Tools attached to root agent: {[t.name for t in tools]}")

    return Agent(
        name="rag_agent_adk",
        description="A chatbot with access to Google Search and Bespoke File Search",
        model=model_id,
        instruction=instruction,
        tools=tools,
        generate_content_config=types.GenerateContentConfig(temperature=1, top_p=1, max_output_tokens=8192),
    )

root_agent = create_root_agent()
