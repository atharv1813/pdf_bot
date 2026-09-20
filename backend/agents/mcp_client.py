import sys
from datetime import timedelta

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

# Self-RAG tool calls are several sequential LLM round-trips (~40-60s) — well
# past langchain-mcp-adapters' default read timeout, so every server here
# gets the same generous ceiling we already proved out in debug_client.py.
CALL_TIMEOUT = timedelta(seconds=180)

MCP_SERVERS = {
    "chat_pdf_rag": {
        # Persistent server, run separately (`python -m mcp_server.server`
        # in its own terminal) — not spawned per call. That means it only
        # ingests and loads its models once, ever, instead of on every
        # single tool call.
        "transport": "streamable_http",
        "url": "http://127.0.0.1:8000/mcp",
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
