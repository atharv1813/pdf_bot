import asyncio
import json
import re

from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from agents.mcp_client import get_mcp_tools
from agents.prompts import SYSTEM_PROMPT
from agents.state import AgentState
from config import model

URL_PATTERN = re.compile(r"https?://\S+")


async def build_agent():
    # MCP tools are async-only (they spawn a subprocess and speak stdio),
    # so both fetching them and running the compiled graph have to happen
    # on the async path — no sync .invoke() anywhere in this file.
    tools = await get_mcp_tools()
    llm_with_tools = model.bind_tools(tools)

    async def call_model(state: AgentState) -> dict:
        messages = [SYSTEM_PROMPT] + state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()


def _tool_text(message: ToolMessage) -> str:
    # MCP tool results arrive as a list of content blocks
    # ([{"type": "text", "text": "..."}]), not a plain string.
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def format_sources(new_messages: list) -> str:
    """Pull real citations out of this turn's tool results — never the LLM's
    own prose — so sources show up whether or not the model bothered to
    mention them, and only once per turn (not re-shown on later turns)."""
    doc_sources: list[dict] = []
    urls: list[str] = []

    for m in new_messages:
        if not isinstance(m, ToolMessage):
            continue
        text = _tool_text(m)
        if m.name == "ask_documents":
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            doc_sources.extend(payload.get("sources", []))
        elif m.name in ("search", "fetch_content"):
            urls.extend(URL_PATTERN.findall(text))

    seen = set()
    unique_docs = []
    for s in doc_sources:
        key = (s.get("title"), s.get("source"))
        if key not in seen:
            seen.add(key)
            unique_docs.append(s)
    unique_urls = list(dict.fromkeys(urls))

    if not unique_docs and not unique_urls:
        return ""

    lines = ["sources:"]
    if unique_docs:
        lines.append("  internal docs:")
        for s in unique_docs:
            title = s.get("title", "unknown")
            source = s.get("source", "")
            lines.append(f"    - {title}" + (f" ({source})" if source else ""))
    if unique_urls:
        lines.append("  internet:")
        for u in unique_urls:
            lines.append(f"    - {u}")
    return "\n".join(lines)


async def main():
    print("Building agent (connecting to MCP servers)...")
    compiled_agent = await build_agent()

    messages = []
    print("Ask a question (Ctrl+C to quit).")
    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        turn_start = len(messages) - 1  # index of the human message just added
        result = await compiled_agent.ainvoke({"messages": messages})
        messages = result["messages"]
        new_messages = messages[turn_start:]

        final_message = new_messages[-1]
        print(f"\nagent> {final_message.content}")

        sources = format_sources(new_messages)
        if sources:
            print(f"\n{sources}")


if __name__ == "__main__":
    asyncio.run(main())
