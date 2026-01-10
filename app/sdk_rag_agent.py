"""
Native SDK RAG Agent with Gemini File Search Tool
"""

import logging
import os

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

def main() -> None:
    """
    Runs a simple interactive chat with Gemini using the File Search Tool.
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
    tools_list = []

    if store and store.name:
        logger.info(f"Found store: {store.name}")
        file_search_tool = types.Tool(file_search=types.FileSearch(file_search_store_names=[store.name]))
        tools_list.append(file_search_tool)
    else:
        logger.warning(f"Store '{STORE_NAME}' not found! RAG capabilities disabled.")

    # 4. Create the chat session
    chat = client.chats.create(
        model=model,
        config=types.GenerateContentConfig(
            tools=tools_list,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False),
            temperature=0.7,
            system_instruction="""You are a helpful assistant with access to a story knowledge base (via File Search).""",
        ),
    )

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
