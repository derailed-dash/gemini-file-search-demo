"""
ADK Basic Agent (Google Search Only)

This module defines an agent using the Google Agent Development Kit (ADK).

WHAT:
A structured agent that uses Gemini and Google Search to answer user queries.
It features a "fail-fast" mechanism to prevent infinite search loops on fictional topics.

WHY:
To demonstrate the ADK's `Agent` and `App` architectural patterns. ADK provides
structure, state management, and easier deployment (via `adk run`) compared to
raw scripts.

HOW:
1.  Defines a `SearchAgent` subclass (mostly for declarative structure).
2.  Defines a `RootAgent` that delegates to the `SearchAgent`.
3.  Uses strict system instructions to "FAIL FAST" if search yields no results.
4.  Exposes an `app` object that the ADK runner discovers and serves.
"""

import logging
import os

from google.adk.agents import Agent
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


search_agent = Agent(
    model=model,
    name="SearchAgent",
    description="Agent to perform Google Search",
    instruction="You're a specialist in Google Search. Only perform one search. Fail fast if no relevant results are found.",
    tools=[google_search],
)

root_agent = Agent(
    name="basic_agent_adk",
    description="You are a helpful AI assistant designed to provide accurate and useful information",
    model=Gemini(
        model=model,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction="""You are a helpful AI assistant designed to provide accurate and useful information.
    If you don't know the answer, use the SearchAgent to perform a Google search.
    Do not attempt to search more than ONCE.
    If the search yields no relevant results or returns unrelated content, you MUST immediately respond with: "I could not find any information about that."
    Do NOT retry the search with different terms. Do NOT ask for clarification. FAIL FAST.""",
    tools=[AgentTool(agent=search_agent)],
)
