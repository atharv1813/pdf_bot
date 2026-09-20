import json
import sys
from pathlib import Path

# This file gets loaded different ways (direct run, `mcp dev`'s own file
# loader, later an agent spawning it as a subprocess) and not all of them
# put backend/ on sys.path automatically — `mcp dev` in particular imports
# this file directly and never adds backend/, so rag/config bare imports
# fail unless we add it ourselves here.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP

from rag.ingest import run_ingestion, CORPORA
from rag.graph import build_graph
from rag.state import RAGState

mcp = FastMCP("chat_pdf_rag_server")

# Ingest once at startup, not lazily inside a tool call, so the vector DB is
# populated before the first real request instead of answering into an
# empty collection on a cold start.
print("[mcp_server] running ingestion...", file=sys.stderr)
run_ingestion()

print("[mcp_server] compiling Self-RAG graph...", file=sys.stderr)
_compiled_graph = build_graph().compile()


@mcp.tool()
def ask_documents(query: str) -> str:
    """Answer a question using the local LangChain/LangGraph documentation corpus."""
    try:
        result = _compiled_graph.invoke(RAGState(query=query))
    except Exception as exc:
        return json.dumps({"status": "error", "message": str(exc)})

    final_answer = result.get("final_answer")
    if not final_answer:
        return json.dumps({"status": "no_answer", "message": "No answer found in documents."})

    sources = [
        {
            "title": doc.metadata.get("title"),
            "source": doc.metadata.get("source"),
            "corpus": doc.metadata.get("corpus"),
        }
        for doc in result.get("relevant_context", [])
    ]

    payload = {
        "status": "ok",
        "answer": final_answer.answer,
        "sources": sources,
    }
    if final_answer.support_state:
        payload["support"] = final_answer.support_state.is_supported
    if final_answer.usefulness_state:
        payload["usefulness"] = final_answer.usefulness_state.is_useful

    return json.dumps(payload)


@mcp.tool()
def list_documents() -> str:
    """List every document available in the knowledge base, grouped by corpus."""
    grouped: dict[str, list[dict]] = {}
    for corpus_name, corpus_dir in CORPORA.items():
        manifest_path = corpus_dir / "manifest.json"
        if not manifest_path.exists():
            grouped[corpus_name] = []
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        grouped[corpus_name] = [
            {
                "title": entry.get("title", entry.get("source_path", entry["file"])),
                "source": entry.get("source", entry.get("source_path", "")),
            }
            for entry in manifest
        ]

    total = sum(len(docs) for docs in grouped.values())
    return json.dumps({"count": total, "documents": grouped})


if __name__ == "__main__":
    # streamable-http, not stdio: runs as a standalone, long-lived process
    # (its own terminal) that the agent connects to over HTTP, instead of
    # being spawned fresh — and re-ingesting/reloading models — on every
    # single tool call.
    mcp.run(transport="streamable-http")
