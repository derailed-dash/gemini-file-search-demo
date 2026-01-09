"""
Native SDK Basic Agent (Google Search Only)

This script demonstrates a minimal "Agent" using the raw Google Gen AI SDK.

WHAT:
A simple CLI chatbot that uses Gemini 2.5 Flash and the native Google Search tool.

WHY:
To establish a baseline for "native" (non-framework) implementation. It highlights
how easy it is to attach the `GoogleSearch` tool to a model.

HOW:
1.  Instantiates `genai.Client()`.
2.  Creates a chat session with `config=types.GenerateContentConfig(tools=[...])`.
3.  Uses the built-in `types.GoogleSearch()` tool.
4.  Runs a simple REPL loop for user interaction.
"""

import logging
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """
    Runs a simple interactive chat with Gemini using the Google Search tool.
    """
    # 1. Initialize the client
    client = genai.Client()

    # 2. Add Google Search tool
    google_search_tool = types.Tool(google_search=types.GoogleSearch())

    # 3. Create the chat session
    model_id = os.getenv("MODEL", "gemini-2.5-flash")
    logger.info(f"Starting chat with model: {model_id}")

    chat = client.chats.create(
        model=model_id,
        config=types.GenerateContentConfig(
            tools=[google_search_tool],
            temperature=0.7,  # slightly creative but grounded
        ),
    )

    print(f"--- Simple Agent (Model: {model_id}) ---")
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
            else:
                print("Agent: [No text response]")

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
