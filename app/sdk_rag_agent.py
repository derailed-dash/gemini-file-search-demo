import logging
import os
import typing

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

STORE_NAME = os.getenv("STORE_NAME")
logger.info(f"Store name: {STORE_NAME}")


def get_store(client: genai.Client, store_name: str) -> types.FileSearchStore | None:
    """Retrieve a store by display name"""
    try:
        for a_store in client.file_search_stores.list():
            if a_store.display_name == store_name:
                return a_store
    except Exception as e:
        logger.error(f"Error listing stores: {e}")
    return None


def main() -> None:
    """
    Runs a simple interactive chat with Gemini using the Google Search tool AND File Search.
    """
    # 1. Initialize the client
    # Note: Explicitly loading dot-env if needed, but assuming env is set or loaded by caller/IDE
    from dotenv import load_dotenv

    load_dotenv()
    client = genai.Client()

    # 2. Retrieve the File Search Store
    logger.info(f"Looking for File Search Store: {STORE_NAME}...")
    if not STORE_NAME:
        logger.warning("STORE_NAME env var not set. Skipping store lookup.")
        store = None
    else:
        store = get_store(client, STORE_NAME)

    tools_list: list[types.Tool] = [types.Tool(google_search=types.GoogleSearch())]

    if store:
        logger.info(f"Found store: {store.name}")
        # Add File Search Tool
        if not store.name:
            # Should not happen for a valid store, but satisfies mypy
            logger.warning("Store found but has no name, skipping File Search.")
        else:
            # Note: Using the plural 'file_search_store_names' based on recent SDK findings
            file_search_tool = types.Tool(file_search=types.FileSearch(file_search_store_names=[store.name]))
            tools_list.append(file_search_tool)
    else:
        logger.warning(f"Store '{STORE_NAME}' not found! Running with Google Search ONLY.")

    # 3. Create the chat session
    model_id = os.getenv("MODEL", "gemini-2.5-flash")
    logger.info(f"Starting chat with model: {model_id}")

    chat = client.chats.create(
        model=model_id,
        config=types.GenerateContentConfig(
            # cast to Any or the union to satisfy invariance
            tools=typing.cast(list[types.Tool | typing.Any], tools_list),
            temperature=0.7,
        ),
    )

    print(f"--- Simple Agent + File Search (Model: {model_id}) ---")
    if store:
        print(f"Connected to Store: {STORE_NAME}")
    else:
        print("!! NO STORE CONNECTED !!")

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
                        print(f"\n[Grounding] Found {len(gm.grounding_chunks)} chunks.")
                    if gm.search_entry_point:
                        print(f"[Grounding] Search Entry Point: {gm.search_entry_point.rendered_content}")
            else:
                print("Agent: [No text response]")

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
