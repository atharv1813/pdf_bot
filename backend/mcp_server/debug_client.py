import asyncio
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

BACKEND_DIR = Path(__file__).resolve().parent.parent

# Spawns server.py itself as a subprocess over stdio — same transport a real
# agent will use later. No inspector UI, no fixed 60s cutoff: the timeout
# below is explicit and generous, so a genuinely slow (not hung) tool call
# still completes instead of getting cut off mid-run.
# sys.executable (not the bare string "python") guarantees the subprocess
# uses the exact same interpreter/venv this script is running under.
SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-m", "mcp_server.server"],
    cwd=str(BACKEND_DIR),
)

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
    print(f"[debug_client] spawning: {SERVER_PARAMS.command} {' '.join(SERVER_PARAMS.args)} (cwd={SERVER_PARAMS.cwd})")
    print("[debug_client] server does ingestion + graph compile before it's ready — this may take a few seconds")

    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=CALL_TIMEOUT) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Available tools:", [t.name for t in tools.tools])

            await call_tool(session, "list_documents")
            await call_tool(session, "ask_documents", {"query": "What is human-in-the-loop in LangGraph?"})


if __name__ == "__main__":
    asyncio.run(main())
