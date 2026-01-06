import logging
import os
import warnings

from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app

# Suppress Pydantic warnings triggered by google-adk internals
warnings.filterwarnings("ignore", module="pydantic._internal._generate_schema")

# Configure logging to ensure app-level INFO logs are captured
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
allow_origins = os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
session_service_uri = None # In-memory session configuration

app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    allow_origins=allow_origins,
    session_service_uri=session_service_uri,
)
app.title = "gemini-file-search-demo"
app.description = "API for interacting with the Agent gemini-file-search-demo"


# Main execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
