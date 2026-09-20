# chat_pdf

A hands-on learning project for **LangGraph, MCP, and agentic RAG** — built
incrementally, version by version (v0 through v8, see git history), as a way
to actually understand these concepts by implementing them rather than just
reading about them. What started as a basic PDF retrieval script is now a
small multi-piece system: a **Self-RAG graph** with **hybrid retrieval and
cross-encoder reranking**, served over its own **MCP server**, called by a
**tool-using LangGraph agent** that also has web search available via a
second MCP server — with logging threaded through every stage so the whole
pipeline is actually observable instead of a black box.

## Architecture

```
you
 │
 ▼
agents/atharv_chatbot_agent.py        (LangGraph ReAct agent)
 │
 ├── MCP (HTTP) ──► mcp_server/server.py   (persistent process, own terminal)
 │                       │
 │                       ▼
 │                  rag/graph.py       (Self-RAG: hybrid retrieval → rerank →
 │                       │              relevance grading → generate →
 │                       │              groundedness check → revise loop →
 │                       ▼              usefulness score)
 │                  Chroma vector store + in-memory BM25 index
 │                  (langchain_dataset_starter/ or langchain_dataset/)
 │
 └── MCP (stdio) ──► duckduckgo_mcp_server   (web search + page fetch)
```

The agent decides per-question whether to answer from the local
documentation corpus or fall back to the web, and both paths go through MCP
rather than being called as plain local functions — the point of this
project was specifically to learn MCP as a real client/server boundary, not
just a buzzword. The RAG server runs as a **standalone, long-lived process**
(`streamable-http` transport) rather than being spawned fresh per call, so
it ingests and loads its models exactly once, no matter how many questions
you ask.

## `rag/` — the Self-RAG pipeline

A LangGraph graph that doesn't just retrieve-and-generate. Per query, in
order:

1. **`decide_retrieval_node`** — decides whether the query needs external
   documents at all, or the LLM can answer directly.
2. **`retrieve_node`** — **hybrid retrieval**: a BM25 (lexical/keyword)
   retriever and a Chroma vector (semantic) retriever each run independently
   over the corpus, then their rankings are fused via
   `EnsembleRetriever.weighted_reciprocal_rank` (Reciprocal Rank Fusion,
   weighted 0.4 BM25 / 0.6 vector). BM25 is rebuilt per query straight from
   whatever's currently in Chroma rather than persisted separately.
3. **`rerank_node`** — a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
   scores every `(query, chunk)` pair jointly, trimming the 15 hybrid
   candidates down to the top 5 — catches cases where BM25 and vector search
   each individually like a candidate for the wrong reason.
4. **`is_relevant_node`** — one batched LLM call grades all 5 survivors for
   relevance at once (not 5 separate round-trips).
5. **`generate_from_context_node`** — drafts an answer from the relevant
   context.
6. **`is_supported_node`** — checks whether the draft is actually grounded
   in the retrieved text. If not, **`revise_answer_node`** regenerates and
   loops back, up to 5 times, before being forced through.
7. **`is_useful_node`** — scores the final answer's usefulness 1–5.

`ingest.py` loads the corpus via each folder's `manifest.json`
(title/source/corpus per chunk), so retrieved context always carries real
citations, not just a filename — and every one of the seven steps above
prints a tagged line to stderr (`[DECIDE]`, `[RETRIEVE]`, `[RERANK]`,
`[SUPPORT]`, `[REVISE]`) so you can watch BM25 vs. vector rankings diverge,
see exact cross-encoder scores, and watch the self-correction loop fire in
real time. See [`docs/demos/full_walkthrough.md`](docs/demos/full_walkthrough.md)
for a full captured example of this, including a real 3-retry revision loop.

## `mcp_server/` — the RAG pipeline as a persistent MCP server

Wraps the graph above behind two MCP tools, served over `streamable-http` as
a standalone process:

| Tool | Description |
|------|-------------|
| `ask_documents` | Runs the full Self-RAG graph, returns a grounded answer + real source titles/URLs |
| `list_documents` | Lists everything in the corpus, grouped by `langgraph`/`langchain` |

Both log `[MCP-SERVER]` lines on entry and return (to stderr — this file
never prints to stdout, since stdout is the actual JSON-RPC channel to the
client; a stray print there will corrupt the protocol mid-stream).

`debug_client.py` is a small standalone MCP client (raw SDK, no inspector
UI) for calling these tools directly against an already-running server and
seeing real output/timing while debugging — useful since Self-RAG answers
take tens of seconds and generic MCP inspector UIs tend to have short
default timeouts.

## `agents/` — the actual chatbot

`atharv_chatbot_agent.py` is a standard LangGraph ReAct agent (`bind_tools`
→ `ToolNode` → `tools_condition`) whose tools come entirely from MCP: the
two tools above, plus `search`/`fetch_content` from an external DuckDuckGo
MCP server (`mcp_client.py`, via `langchain-mcp-adapters`' `MultiServerMCPClient`).
`call_model` logs `[AGENT] calling tool: ...` for every tool call the model
emits, so you can see exactly which tool it reached for and why.

The system prompt (`prompts.py`) explicitly tells the agent to prefer
`ask_documents` over web search for LangChain/LangGraph questions, and to
call `ask_documents` **at most once per question** unless the first call
came back empty — each call is a genuinely expensive multi-step pipeline
(tens of seconds), and without this constraint the agent would sometimes
call it 2–3 times per turn "for thoroughness."

Sources are extracted **programmatically** from tool results after each
turn (`format_sources`) — never left to the model's memory — and printed as
a separate, deduped footer split into "internal docs" vs. "internet," so
citations are always accurate regardless of what the model's prose says.

## Data

- **`langchain_dataset/`** — the full LangChain + LangGraph documentation
  (~627 pages), exported from each project's `llms-full.txt`.
- **`langchain_dataset_starter/`** — a curated ~28-page subset covering just
  the concepts this project itself uses (agents, tools, memory, MCP,
  human-in-the-loop, persistence, streaming) — small enough to ingest and
  iterate on quickly while developing. This is what the pipeline actually
  runs against today.

Both are split per-page with a `manifest.json` mapping each `.txt` file back
to its original title/source.

## See it working

[`docs/demos/full_walkthrough.md`](docs/demos/full_walkthrough.md) is a
captured, real run of three deliberately-chosen queries against the live
system, with every `[TAG]`ged log line kept intact:

1. A well-covered question that still triggers 2 rounds of self-correction
   before the groundedness checker is satisfied.
2. A question half-covered by the local corpus (`AgentExecutor` never
   appears in it) — the agent tries 3 local retrieval phrasings, then falls
   back to web search for the uncovered half, and synthesizes both sources
   into one answer.
3. A "check the web" question the agent correctly routes to `search`/
   `fetch_content` without ever touching `ask_documents` — proof the routing
   decision is real, not just a static preference.

## Running it

```bash
cd backend
pip install mcp langchain-mcp-adapters langchain-aws langchain-chroma \
            langchain-huggingface langgraph langchain-classic \
            langchain-community sentence-transformers rank_bm25 \
            duckduckgo-mcp-server python-dotenv
```

Needs a `.env` with AWS credentials for Bedrock (`ChatBedrockConverse`,
currently `us.anthropic.claude-sonnet-4-6`) — see `config.py`.

**Two terminals** — the RAG server runs standalone and the agent connects to
it over HTTP, rather than being spawned per call:

```bash
# terminal 1 — leave running, watch its logs live
cd backend
python -m mcp_server.server
```
```bash
# terminal 2
cd backend
python -m agents.atharv_chatbot_agent
```

**Test the RAG-over-MCP server on its own** (terminal 1 must already be
running), without the agent:
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
- No persistence (conversations aren't remembered across process restarts)
  or human-in-the-loop approval yet — both are planned next steps.
- Retrieval sometimes cites a broadly-relevant overview page instead of the
  most specific one available — hybrid retrieval + reranking narrowed this
  considerably, but it's still an embedding/ranking-quality question, not a
  correctness bug (answers are still grounded and checked before being
  returned to the user).
- `hi.py` in `backend/` is a leftover one-off credentials sanity check, not
  part of the actual pipeline.
