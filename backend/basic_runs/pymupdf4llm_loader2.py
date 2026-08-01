from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_pymupdf4llm import PyMuPDF4LLMParser

loader = GenericLoader(
    blob_loader=FileSystemBlobLoader(path="/home/atharv/GenAI/study_vedya/langgraph/projects/chat_pdf/backend/data/attention_paper.pdf", glob="*.pdf"),
    blob_parser=PyMuPDF4LLMParser(),
)



docs = loader.load()
print(type(docs))
print(f"Total chunks: {len(docs)}")

for doc in docs:
    print("\n" + "="*60)
    print(f"Metadata: {doc.metadata}")
    print(f"Content: {doc.page_content}...")