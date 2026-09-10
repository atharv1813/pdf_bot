import json
from pathlib import Path

from langchain_core.tools import tool

from agent.mcp_client import get_web_search_tool

DOCUMENTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "documents"


@tool
def retrieve_documents(query: str) -> str:
    """Search the document knowledge base and return a grounded answer."""
    from main import build_graph
    from utils.state import RAGState

    result = build_graph().compile().invoke(RAGState(query=query))
    final_answer = result.get("final_answer")

    if not final_answer:
        return json.dumps({"status": "no_answer", "message": "No answer found in documents."})

    payload = {
        "status": "ok",
        "answer": final_answer.answer,
        "source_doc_ids": final_answer.source_doc_ids,
    }
    if final_answer.support_state:
        payload["support"] = final_answer.support_state.is_supported
    if final_answer.usefulness_state:
        payload["usefulness"] = final_answer.usefulness_state.is_useful

    return json.dumps(payload)


@tool
def list_documents() -> str:
    """List all documents available in the knowledge base."""
    if not DOCUMENTS_DIR.exists():
        return json.dumps({"documents": [], "count": 0})

    files = sorted(
        (p.name for p in DOCUMENTS_DIR.glob("*.txt")),
        key=lambda name: int(name.removesuffix(".txt")),
    )
    return json.dumps({"documents": files, "count": len(files)})


def get_all_tools():
    """Return local tools plus MCP-backed web search."""
    web_search = get_web_search_tool()
    return [retrieve_documents, list_documents, web_search]
