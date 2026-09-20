# chat_pdf

A hands-on learning project for **LangGraph, MCP, and agentic RAG** — built
incrementally, version by version, as a way to actually understand these
concepts by implementing them rather than just reading about them. What
started as a basic retrieval pipeline is now a small multi-piece system: a
Self-RAG graph, served over its own MCP server, called by a tool-using
LangGraph agent that also has web search available via a second MCP server.

## Architecture

```
you
 │
 ▼
agents/atharv_chatbot_agent.py   (LangGraph ReAct agent)
 │
 ├── MCP ──► mcp_server/server.py  ──► rag/graph.py  (Self-RAG over local docs)
 │                                       │
 │                                       ▼
 │                                 Chroma vector store
 │                                 (langchain_dataset_starter/ or langchain_dataset/)
 │
 └── MCP ──► duckduckgo_mcp_server  (web search + page fetch)
```

The agent decides per-question whether to answer from the local documentation
corpus or fall back to the web, and both paths go through MCP rather than
being called as plain local functions — the point of this version was
specifically to learn MCP as a real client/server boundary, not just as a
buzzword.

### `rag/` — the Self-RAG pipeline
A LangGraph graph that doesn't just retrieve-and-generate: it decides *whether*
retrieval is even needed, grades retrieved chunks for relevance (one batched
LLM call, not one per chunk), checks whether the generated answer is actually
grounded in the retrieved context, revises itself if not (up to 5 times), and
finally scores its own answer's usefulness before returning it. `ingest.py`
loads the corpus via each folder's `manifest.json` (title/source/corpus per
chunk), so retrieved context always carries real citations, not just a
filename.

### `mcp_server/` — the RAG pipeline as an MCP server
Wraps the graph above behind two MCP tools:
| Tool | Description |
|------|-------------|
| `ask_documents` | Runs the full Self-RAG graph, returns a grounded answer + real source titles/URLs |
| `list_documents` | Lists everything in the corpus, grouped by `langgraph`/`langchain` |

`debug_client.py` is a small standalone MCP client (raw SDK, no inspector UI)
for calling these tools directly and seeing real output/timing while
debugging — useful since Self-RAG answers take tens of seconds and generic
MCP inspector UIs tend to have short default timeouts.

### `agents/` — the actual chatbot
`atharv_chatbot_agent.py` is a standard LangGraph ReAct agent (`bind_tools` →
`ToolNode` → `tools_condition`) whose tools come entirely from MCP: the two
tools above, plus `search`/`fetch_content` from an external DuckDuckGo MCP
server. Sources are extracted **programmatically** from tool results after
each turn (not left to the model to remember to cite) and printed as a
separate, deduped footer split into "internal docs" vs "internet" — so
citations are always accurate regardless of what the model's prose says.

## Data

- **`langchain_dataset/`** — the full LangChain + LangGraph documentation
  (~627 pages), exported from each project's `llms-full.txt`.
- **`langchain_dataset_starter/`** — a curated ~28-page subset covering just
  the concepts this project itself uses (agents, tools, memory, MCP,
  human-in-the-loop, persistence, streaming) — small enough to ingest and
  iterate on quickly while developing.

Both are split per-page with a `manifest.json` mapping each `.txt` file back
to its original title/source.

## Running it

```bash
cd backend
pip install mcp langchain-mcp-adapters langchain-aws langchain-chroma \
            langchain-huggingface langgraph duckduckgo-mcp-server python-dotenv
```

Needs a `.env` with AWS credentials for Bedrock (`ChatBedrockConverse`,
currently `us.anthropic.claude-sonnet-4-6`) — see `config.py`.

**Talk to the full agent:**
```bash
python -m agents.atharv_chatbot_agent
```

**Test the RAG-over-MCP server on its own**, without the agent:
```bash
python -m mcp_server.debug_client
```

**Ingest the corpus by itself** (also runs automatically on server startup):
```bash
python -c "from rag.ingest import run_ingestion; run_ingestion()"
```

## Known limitations

This is a learning project, not a production system — a few known rough
edges, left as-is deliberately or as next steps:
- No `requirements.txt`/`pyproject.toml` yet — dependencies were added ad hoc
  while building.
- Each MCP tool call currently spawns the RAG server fresh rather than
  reusing one long-lived connection, so there's a few seconds of startup
  overhead (embeddings model load, ingestion idempotency check) per call.
- No persistence (conversations aren't remembered across runs) or
  human-in-the-loop approval yet — both are planned next steps once this
  base is solid.
- Retrieval sometimes cites a broadly-relevant overview page instead of the
  most specific one available — an embedding-quality tuning question, not a
  correctness bug (answers are still grounded and checked before being
  returned to the user).
