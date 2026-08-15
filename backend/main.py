from dotenv import load_dotenv
import os
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from utils.state import RAGState
from sentence_transformers import CrossEncoder


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


# ===================== PDF Identity maker =====================
import hashlib
def get_pdf_hash(pdf_path: str) -> str:
    """Stable ID based on file content — same file = same hash, different file = different hash."""
    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:12]


if __name__ == "__main__":
    # Example usage
    pdf_path = "example.pdf"
    pdf_hash = get_pdf_hash(pdf_path)
    print(f"PDF Hash: {pdf_hash}")
    