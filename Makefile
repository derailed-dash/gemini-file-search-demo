# Install dependencies using uv package manager
install:
	@command -v uv >/dev/null 2>&1 || { echo "uv is not installed. Installing uv..."; curl -LsSf https://astral.sh/uv/0.8.13/install.sh | sh; source $HOME/.local/bin/env; }
	uv sync --extra jupyter

# Launch local dev playground
adk-playground:
	@echo "==============================================================================="
	@echo "| 🚀 Starting your agent playground...                                        |"
	@echo "==============================================================================="
	uv run adk web app --port 8501 --reload_agents

# Run the SDK agent (Google Search only)
sdk-agent:
	uv run python app/sdk_agent.py

# Run the SDK RAG agent (File Search + Google Search)
sdk-rag-agent:
	uv run python app/sdk_rag_agent.py

# Run code quality checks (codespell, ruff, mypy)
lint:
	uv sync --dev --extra lint
	uv run codespell
	uv run ruff check . --diff
	uv run ruff format . --check --diff
	uv run mypy .