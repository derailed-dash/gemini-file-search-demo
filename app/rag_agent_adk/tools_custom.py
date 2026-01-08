from __future__ import annotations

import logging
import typing
from typing import TYPE_CHECKING

from google.adk.tools import BaseTool, ToolContext
from google.genai import types

if TYPE_CHECKING:
    from google.adk.models import LlmRequest

logger = logging.getLogger(__name__)


class FileSearchTool(BaseTool):
    """
    A custom ADK tool that enables the Gemini File Search (retrieval) capability.
    This attaches the native 'file_search' tool configuration to the model request.
    """

    def __init__(self, file_search_store_names: list[str]):
        """
        Initialize the FileSearchTool.

        Args:
            file_search_store_names: The resource name of the File Search Store.
                    e.g. ["fileSearchStores/mystore-abcdef0pqrst", ...]
        """
        # Note: We don't define 'functions' or 'code' here because it's a server-side tool
        super().__init__(name="file_search", description="Retrieval from file search store")
        self.file_search_store_names = file_search_store_names

    async def process_llm_request(
        self,
        *,
        tool_context: ToolContext,
        llm_request: LlmRequest,
    ) -> None:
        """
        Updates the model request configuration to include the File Search tool.
        """
        logger.debug(f"Attaching File Search Store: {self.file_search_store_names}")

        llm_request.config = llm_request.config or types.GenerateContentConfig()
        llm_request.config = llm_request.config or types.GenerateContentConfig()

        # Ensure tools is a list we can append to
        current_tools = llm_request.config.tools or []
        # cast to list to satisfy mypy, assuming it's mutable
        target_tools = typing.cast(list[types.Tool], current_tools)

        # Append the native tool configuration for File Search
        target_tools.append(
            types.Tool(file_search=types.FileSearch(file_search_store_names=self.file_search_store_names))
        )
        # Cast back to Any to satisfy the complex Union expected by GenerateContentConfig.tools
        llm_request.config.tools = typing.cast(list[typing.Any], target_tools)
