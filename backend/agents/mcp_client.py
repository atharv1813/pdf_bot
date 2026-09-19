import sys
from datetime import timedelta
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

BACKEND_DIR = Path(__file__).resolve().parent.parent

# Self-RAG tool calls are several sequential LLM round-trips (~40-60s) — well
# past langchain-mcp-adapters' default read timeout, so every server here
# gets the same generous ceiling we already proved out in debug_client.py.
CALL_TIMEOUT = timedelta(seconds=180)

MCP_SERVERS = {
    "chat_pdf_rag": {
        "transport": "stdio",
        # sys.executable, not a bare "python" string — guarantees the
        # subprocess uses this same interpreter/venv (bit us once already
        # in debug_client.py).
        "command": sys.executable,
        "args": ["-m", "mcp_server.server"],
        "cwd": str(BACKEND_DIR),
        "session_kwargs": {"read_timeout_seconds": CALL_TIMEOUT},
    },
    "duckduckgo": {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", "duckduckgo_mcp_server.server"],
        "session_kwargs": {"read_timeout_seconds": CALL_TIMEOUT},
    },
}


async def get_mcp_tools() -> list[BaseTool]:
    """Connect to every configured MCP server and return their tools as a
    flat list of LangChain Tool objects, ready for llm.bind_tools()."""
    client = MultiServerMCPClient(MCP_SERVERS)
    return await client.get_tools()


if __name__ == "__main__":
    import asyncio

    async def _list():
        tools = await get_mcp_tools()
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")

    asyncio.run(_list())
