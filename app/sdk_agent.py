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

    # 2. Define the tool
    # The 'google_search_retrieval' tool is a built-in tool in the Gemini API.
    # We configure it within the GenerateContentConfig.
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
