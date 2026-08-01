
# from unittest import loader

# import pytest
# from langchain_core.documents import Document

# from langchain_pymupdf4llm import pymupdf4llm_loader as loader_module
# from langchain_pymupdf4llm.pymupdf4llm_loader import BasePDFLoader, PyMuPDF4LLMLoader
from langchain_pymupdf4llm import PyMuPDF4LLMLoader

loader = PyMuPDF4LLMLoader(
    file_path="/home/atharv/GenAI/study_vedya/langgraph/projects/chat_pdf/backend/data/attention_paper.pdf"
)

docs = loader.load()
print(type(docs))
print(f"Total chunks: {len(docs)}")

for doc in docs:
    print("\n" + "="*60)
    print(f"Metadata: {doc.metadata}")
    print(f"Content: {doc.page_content}...")
    
    
    
# attention_paper.pdf
#         │
#         ▼
# PyMuPDF4LLMLoader
#         │
#         ▼
# Open PDF
#         │
#         ▼
# Read Metadata
#         │
#         ▼
# Loop Through Every Page
#         │
#         ├── Extract Text
#         ├── OCR (if needed)
#         ├── Extract Tables
#         ├── Extract Images
#         └── Convert to Markdown
#         │
#         ▼
# Create LangChain Document
#         │
#         ▼
# [
#  Document(Page 1),
#  Document(Page 2),
#  ...
#  Document(Page 15)
# ]
#         │
#         ▼
# RecursiveCharacterTextSplitter
#         │
#         ▼
# Smaller Chunks
#         │
#         ▼
# Embeddings
#         │
#         ▼
# Vector Database
#         │
#         ▼
# Retriever
#         │
#         ▼
# LLM