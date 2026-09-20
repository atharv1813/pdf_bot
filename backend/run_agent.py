from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import build_agent_graph

DEMO_QUERIES = [
    "What documents do you have?",
    "What do keybullet kin drop?",
    "What is the weather in Mumbai today?",
    "Tell me a joke",
]


def print_trace(messages: list) -> None:
    step = 0
    print("--- Agent Trace ---")

    for msg in messages:
        if isinstance(msg, HumanMessage):
            continue

        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                for call in msg.tool_calls:
                    step += 1
                    args = call.get("args", {})
                    print(f"[{step}] TOOL CALL  -> {call['name']}({args})")
            elif msg.content:
                step += 1
                print(f"[{step}] FINAL      -> {msg.content}")

        elif isinstance(msg, ToolMessage):
            step += 1
            preview = str(msg.content)
            if len(preview) > 200:
                preview = preview[:200] + "..."
            print(f"[{step}] TOOL RESULT -> {preview}")


def run_query(agent_graph, query: str) -> None:
    print(f"\n{'=' * 50}")
    print(f"Query: {query}")

    result = agent_graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"recursion_limit": 12},
    )

    print_trace(result["messages"])


if __name__ == "__main__":
    agent_graph = build_agent_graph()

    for query in DEMO_QUERIES:
        run_query(agent_graph, query)
