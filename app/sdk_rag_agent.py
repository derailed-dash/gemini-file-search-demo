"""
Native SDK RAG Agent with "Tool as an Agent" Pattern

Demonstrates how to combine Gemini's File Search tool with Google Search
using the Google Gen AI SDK.

PROBLEM:
The Gemini API currently does not support using the native `GoogleSearch` tool and
the `FileSearch` tool in the same request. Attempting to do so results in a
400 INVALID_ARGUMENT error: "Search as a tool and file search tool are not supported together".

SOLUTION:
To circumvent this limitation, we implement the "Tool as an Agent" pattern:
1.  We define a standard Python function `google_search_tool` that takes a query.
2.  Inside this function, we instantiate a *new* `genai.Client` (a sub-agent).
3.  This sub-agent is configured specifically to use the native `GoogleSearch` tool.
4.  The main agent sees this function as just another user-defined tool, allowing it
    to coexist happily with the native `FileSearch` tool.

This allows the main agent to decide when to query the File Store (native RAG)
and when to query the web (via the sub-agent function).
"""

import logging
import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

STORE_NAME = os.getenv("STORE_NAME")
logger.info(f"Store name: {STORE_NAME}")

model = os.getenv("MODEL", "gemini-2.5-flash")
logger.info(f"Using model: {model}")


def get_store(client: genai.Client, store_name: str) -> types.FileSearchStore | None:
    """Retrieve a store by display name"""
    try:
        for a_store in client.file_search_stores.list():
            if a_store.display_name == store_name:
                return a_store
    except Exception as e:
        logger.error(f"Error listing stores: {e}")
    return None


def google_search_tool(query: str) -> str:
    """
    Performs a Google Search using a sub-agent.
    ONLY use this tool if you cannot find the answer in the File Search store.
    Useful for finding current information, news, or topics NOT covered in the story.
    """
    logger.info(f"[Tool] Performing Google Search for: {query}")
    try:
        # Create a fresh client/chat for the sub-agent
        sub_client = genai.Client()
        response = sub_client.models.generate_content(
            model=os.getenv("MODEL", "gemini-2.5-flash"),
            contents=f"Please search for: {query}. Summarize the results found.",
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.0,  # Fact-focused
            ),
        )
        return response.text if response.text else "No results found."
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return f"Error performing search: {e}"


def main() -> None:
    """
    Runs a simple interactive chat with Gemini using the Google Search tool AND File Search.
    Uses 'Tool as an Agent' pattern to circumvent API exclusivity.
    """
    # 1. Initialize the client
    client = genai.Client()

    # 2. Retrieve the File Search Store
    logger.info(f"Looking for File Search Store: {STORE_NAME}...")

    store = None
    if STORE_NAME:
        store = get_store(client, STORE_NAME)
    else:
        logger.warning("STORE_NAME env var not set. Skipping store lookup.")

    # 3. Build Tools List
    # We ALWAYS add our custom google_search_tool (Tool as an Agent)
    tools_list: list[Any] = [google_search_tool]

    if store and store.name:
        logger.info(f"Found store: {store.name}")
        file_search_tool = types.Tool(file_search=types.FileSearch(file_search_store_names=[store.name]))
        tools_list.append(file_search_tool)
    else:
        logger.warning(f"Store '{STORE_NAME}' not found! RAG capabilities disabled.")

    # 4. Create the chat session
    # Note that we use the system_instruction to guide the agent on when to use the tools
    # We want to use the File Search tool FIRST, then fallback to Google Search if needed
    chat = client.chats.create(
        model=model,
        config=types.GenerateContentConfig(
            tools=tools_list,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False),
            temperature=0.7,
            system_instruction="""You are a helpful assistant with access to a story knowledge base (via File Search) and Google Search.

            PRIORITY:
            1. ALWAYS check your internal story knowledge base first. If the user asks about characters, ships, or events from the story (e.g., "Krellons", "Star-Eater", "Attitude Adjuster"), use the information retrieved from the file store.
            2. ONLY use the `google_search_tool` if the information is clearly NOT in the story (e.g., real-world stock prices, weather, historical facts).

            Do NOT attempt to "call" the File Search tool as a function. It is automatically applied to your context. Just use the information it provides.
            """,
        ),
    )

    print("--- Native SDK Agent (Tool-as-Agent Pattern) ---")
    if store:
        print(f"Connected to Store: {STORE_NAME}")
    print("Type 'exit' or 'quit' to stop.")
    print("------------------------------------------")

    while True:
        try:
            user_input = input("User: ").strip()
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
            if not user_input:
                continue

            # 4. Send message and get response
            response = chat.send_message(user_input)

            # 5. Print response
            if response.text:
                print(f"Agent: {response.text}")
                # Print citations if available to prove RAG is working
                if response.candidates and response.candidates[0].grounding_metadata:
                    gm = response.candidates[0].grounding_metadata
                    if gm.grounding_chunks:
                        print(f"\n[Grounding] Found {len(gm.grounding_chunks)} chunks from File Search.")
                    if gm.search_entry_point:
                        print(f"[Grounding] Search Entry Point: {gm.search_entry_point.rendered_content}")
            else:
                print("Agent: [No text response]")

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
