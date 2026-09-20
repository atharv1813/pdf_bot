import asyncio
import sys

from langchain_core.tools import BaseTool, tool
from langchain_community.tools import DuckDuckGoSearchRun


def _format_tool_result(result) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts = []
        for block in result:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "\n".join(parts) if parts else str(result)
    return str(result)


async def _load_mcp_search_tool() -> BaseTool | None:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "duckduckgo": {
                "command": sys.executable,
                "args": ["-m", "duckduckgo_mcp_server.server"],
                "transport": "stdio",
            }
        }
    )
    mcp_tools = await client.get_tools()
    for mcp_tool in mcp_tools:
        if mcp_tool.name == "search":
            return mcp_tool
    return None


def get_web_search_tool() -> BaseTool:
    """Return web_search tool backed by MCP, with DuckDuckGo fallback."""
    mcp_search = None
    try:
        mcp_search = asyncio.run(_load_mcp_search_tool())
        if mcp_search is not None:
            print("[mcp] Using DuckDuckGo MCP for web_search")
    except Exception as exc:
        print(f"[mcp] DuckDuckGo MCP unavailable, using fallback: {exc}")

    if mcp_search is not None:

        @tool
        def web_search(query: str) -> str:
            """Search the web for current information via DuckDuckGo MCP."""
            result = asyncio.run(mcp_search.ainvoke({"query": query}))
            return _format_tool_result(result)

        return web_search

    print("[mcp] Using DuckDuckGoSearchRun fallback for web_search")
    fallback = DuckDuckGoSearchRun()

    @tool
    def web_search(query: str) -> str:
        """Search the web for current information."""
        return fallback.run(query)

    return web_search
