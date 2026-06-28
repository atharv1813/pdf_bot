from dotenv import load_dotenv
import os
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from langgraph.graph import StateGraph, START, END
from utils.state import RAGState 

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


import hashlib
def get_pdf_hash(pdf_path: str) -> str:
    """Stable ID based on file content — same file = same hash, different file = different hash."""
    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:12]
    
def rag_node(state: RAGState):

    print("\n" + "="*60)
    print("🔵 [RAG NODE] Starting...")
    print(f"   Query    : {state.query}")
    print(f"   PDF Path : {state.pdf_path}")

    # 1. Load + chunk

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

    # 4. MMR retriever — relevance + diversity in one shot, no query expansion needed
    mmr_retriever = VECTOR_DB.as_retriever(
        search_type="mmr",                        # Maximum Marginal Relevance
        search_kwargs={
            "k": 5,                               # final docs to return
            "fetch_k": 20,                        # candidate pool to pick from
            "lambda_mult": 0.7,                   # 1.0 = pure relevance, 0.0 = pure diversity
            "filter": {"pdf_id": {"$in": active_pdf_ids}}
        }
    )

    retrieved_docs = mmr_retriever.invoke(state.query)
    print(f"\n📚 {len(retrieved_docs)} diverse docs retrieved via MMR")

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
    graph = StateGraph(RAGState)          # schema passed here

    graph.add_node("rag", rag_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "rag")
    graph.add_edge("rag", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


langgraph_app = build_graph()

initial_state = {
    "query": "what happened to Germany after WW1",  
    "expanded_query": "",
    "answer": "",
    "pdf_ids": None,
    "pdf_path": "data/ww2.pdf",   # <-- your PDF
    "context": []
}

result = langgraph_app.invoke(initial_state)

print("=" * 60)
print("QUERY:", result["query"])
print("=" * 60)
print("\nANSWER:\n", result["answer"])
print("\nCONTEXT CHUNKS USED:")
for i, chunk in enumerate(result["context"], 1):
    print(f"\n[Chunk {i}]")
    print("Content:", chunk["content"][:300], "...")
    print("Metadata:", chunk["metadata"])

