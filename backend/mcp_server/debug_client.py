import asyncio
import json
import time
from datetime import timedelta

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# Connects to an already-running server (`python -m mcp_server.server` in
# its own terminal), instead of spawning one — same persistent-server model
# the agent now uses, so you're debugging the actual thing you'll talk to,
# not a fresh one-off subprocess.
SERVER_URL = "http://127.0.0.1:8000/mcp"

CALL_TIMEOUT = timedelta(seconds=180)


async def call_tool(session: ClientSession, name: str, arguments: dict | None = None):
    t0 = time.time()
    result = await session.call_tool(name, arguments or {}, read_timeout_seconds=CALL_TIMEOUT)
    elapsed = time.time() - t0
    print(f"\n=== {name}({arguments or {}}) — {elapsed:.1f}s ===")
    for block in result.content:
        text = getattr(block, "text", None)
        if text is None:
            print(block)
            continue
        try:
            print(json.dumps(json.loads(text), indent=2))
        except json.JSONDecodeError:
            print(text)


async def main():
    print(f"[debug_client] connecting to {SERVER_URL}")
    print("[debug_client] make sure `python -m mcp_server.server` is already running in another terminal")

    async with streamablehttp_client(SERVER_URL) as (read, write, _):
        async with ClientSession(read, write, read_timeout_seconds=CALL_TIMEOUT) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Available tools:", [t.name for t in tools.tools])

            await call_tool(session, "list_documents")
            await call_tool(session, "ask_documents", {"query": "What is human-in-the-loop in LangGraph?"})


if __name__ == "__main__":
    asyncio.run(main())
