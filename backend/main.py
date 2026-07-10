from dotenv import load_dotenv
import os
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from utils.state import RAGState
from sentence_transformers import CrossEncoder

import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# ===================== SETUP =====================
load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# ===================== VECTOR DB =====================
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "pdf_chunks"
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
VECTOR_DB = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR
)

# ===================== CROSS-ENCODER (for reranking) =====================
RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ===================== PDF Identity maker =====================
import hashlib
def get_pdf_hash(pdf_path: str) -> str:
    """Stable ID based on file content — same file = same hash, different file = different hash."""
    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:12]


# ===================== NODES =====================
def rag_node_hybrid(state: RAGState):
    """Retrieval stage: loads/chunks PDF, ingests if new, then runs TRUE hybrid
    search — BM25 (lexical/keyword) + vector (semantic) in parallel, fused via
    EnsembleRetriever (Reciprocal Rank Fusion). Over-fetches so the reranker
    downstream has a wide, diverse candidate pool to work with."""

    print("\n" + "="*60)
    print("🔵 [RAG NODE] Starting...")
    print(f"   Query    : {state.query}")
    print(f"   PDF Path : {state.pdf_path}")

    # 1. Load + chunk
    loader = PyPDFLoader(state.pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"\n📄 {len(docs)} pages → {len(chunks)} chunks")

    # 2. Hash-based dedup ingestion
    pdf_hash = get_pdf_hash(state.pdf_path)
    print(f"🔑 PDF hash: {pdf_hash}")

    existing = VECTOR_DB.get(where={"pdf_id": pdf_hash})
    if existing["ids"]:
        print(f"💾 Already ingested ({len(existing['ids'])} chunks) — skipping")
    else:
        for chunk in chunks:
            chunk.metadata["pdf_id"] = pdf_hash
        VECTOR_DB.add_documents(chunks)
        print(f"💾 Ingested {len(chunks)} chunks with pdf_id={pdf_hash}")

    # 3. Resolve active pdf_ids
    active_pdf_ids = state.pdf_ids if state.pdf_ids else [pdf_hash]
    print(f"\n🗂️  Searching across pdf_ids: {active_pdf_ids}")

    RETRIEVAL_K = 15   # wider net; reranker will trim this down later

    # 4a. Vector retriever — semantic similarity (Chroma)
    vector_retriever = VECTOR_DB.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": RETRIEVAL_K,
            "filter": {"pdf_id": {"$in": active_pdf_ids}}
        }
    )

    # 4b. BM25 retriever — lexical/keyword search
    # BM25Retriever works in-memory over a fixed doc list, so pull ALL chunks
    # belonging to the active pdf_ids straight out of Chroma to build it from.
    raw = VECTOR_DB.get(
        where={"pdf_id": {"$in": active_pdf_ids}},
        include=["documents", "metadatas"]
    )
    bm25_corpus = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]
    print(f"\n📖 BM25 corpus built from {len(bm25_corpus)} chunks (pdf_ids={active_pdf_ids})")

    bm25_retriever = BM25Retriever.from_documents(bm25_corpus)
    bm25_retriever.k = RETRIEVAL_K

    # 4c. Fuse both via EnsembleRetriever (Reciprocal Rank Fusion)
    # weights: [bm25, vector] — tune these based on how much you trust
    # keyword matches vs semantic matches for your use case.
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )

    retrieved_docs = hybrid_retriever.invoke(state.query)
    print(f"\n📚 {len(retrieved_docs)} candidate docs retrieved via hybrid (BM25 + vector)")

    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\n   [Doc {i}] page={doc.metadata.get('page','?')} pdf_id={doc.metadata.get('pdf_id','?')}")
        print(f"            {doc.page_content[:120].strip()}...")

    context = [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in retrieved_docs
    ]

    print(f"\n✅ [RAG NODE] Done.")
    print("="*60)

    return {
        "context": context,
        "pdf_ids": active_pdf_ids
    }


def rerank_node(state: RAGState):
    """Reranking stage: cross-encoder scores each (query, chunk) pair jointly —
    especially useful now, since BM25 + vector fusion can surface candidates
    that are individually strong on one axis (keyword or semantic) but not
    jointly relevant. The cross-encoder re-evaluates true relevance directly."""

    print("\n" + "="*60)
    print("🟠 [RERANK NODE] Starting...")
    print(f"   Query           : {state.query}")
    print(f"   Candidates in   : {len(state.context)}")

    TOP_N = 5

    if not state.context:
        print("⚠️  No candidates to rerank — skipping")
        return {"context": []}

    pairs = [(state.query, c["content"]) for c in state.context]
    scores = RERANKER.predict(pairs)

    scored_context = [
        {**c, "rerank_score": float(score)}
        for c, score in zip(state.context, scores)
    ]
    scored_context.sort(key=lambda x: x["rerank_score"], reverse=True)
    reranked = scored_context[:TOP_N]

    print(f"\n📊 Reranked scores (all candidates):")
    for i, c in enumerate(scored_context, 1):
        marker = "✅" if i <= TOP_N else "  "
        page = c["metadata"].get("page", "?")
        print(f"   {marker} [{i:2}] score={c['rerank_score']:.4f}  page={page}  {c['content'][:80].strip()}...")

    print(f"\n✅ [RERANK NODE] Kept top {len(reranked)} of {len(scored_context)}")
    print("="*60)

    return {"context": reranked}


def generate_node(state: RAGState):

    print("\n" + "="*60)
    print("🟢 [GENERATE NODE] Starting...")
    print(f"   Query          : {state.query}")
    print(f"   Context chunks : {len(state.context)}")

    context_text = "\n\n".join(c["content"] for c in state.context)

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the context does not contain enough information, say so honestly.

Context:
{context_text}

Question: {state.query}

Answer:"""

    print(f"\n🤖 Sending to Gemini ({len(prompt)} chars)...")
    response = llm.invoke(prompt)
    answer = response.content.strip()

    print(f"\n💬 Answer ({len(answer)} chars):\n   {answer[:300]}...")
    print("="*60)

    return {"answer": answer}


# ===================== GRAPH =====================

def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("rag", rag_node_hybrid)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "rag")
    graph.add_edge("rag", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


langgraph_app = build_graph()

initial_state = {
    "query": "what happened to Germany after WW1",
    "expanded_query": "",
    "answer": "",
    "pdf_ids": None,
    "pdf_path": "data/ww2.pdf",
    "context": []
}

result = langgraph_app.invoke(initial_state)

print("=" * 60)
print("QUERY:", result["query"])
print("=" * 60)
print("\nANSWER:\n", result["answer"])
print("\nCONTEXT CHUNKS USED HERE:")
for i, chunk in enumerate(result["context"], 1):
    print(f"\n[Chunk {i}]")
    print("Content:", chunk["content"][:300], "...")
    print("Metadata:", chunk["metadata"])
    print("Rerank score:", chunk.get("rerank_score"))