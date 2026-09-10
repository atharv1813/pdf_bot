# chat_pdf — Agentic RAG Showcase (version7)

LangGraph RAG pipeline with a simple agent layer: tool-calling, MCP web search, and CLI action trace.

## Run agent demo

```bash
cd backend
python run_agent.py
```

## Tools

| Tool | Description |
|------|-------------|
| `retrieve_documents` | Runs the RAG graph on the local `.txt` corpus |
| `list_documents` | Lists files in `data/documents/` |
| `web_search` | DuckDuckGo via MCP (falls back to `DuckDuckGoSearchRun`) |

## Run RAG pipeline only

```bash
cd backend
python main.py
```
