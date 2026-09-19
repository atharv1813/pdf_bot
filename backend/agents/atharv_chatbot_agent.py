import asyncio

from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from agents.mcp_client import get_mcp_tools
from agents.prompts import SYSTEM_PROMPT
from agents.state import AgentState
from config import model


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
        result = await compiled_agent.ainvoke({"messages": messages})
        messages = result["messages"]

        final_message = messages[-1]
        print(f"\nagent> {final_message.content}")


if __name__ == "__main__":
    asyncio.run(main())
