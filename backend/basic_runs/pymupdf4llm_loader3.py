from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.document_loaders.parsers import LLMImageBlobParser

load_dotenv()

llm = ChatOpenAI(
    model="qwen/qwen3-vl-8b-instruct",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

loader = PyMuPDF4LLMLoader(
    "/home/atharv/GenAI/study_vedya/langgraph/projects/chat_pdf/backend/data/attention_paper.pdf",
    mode="page",
    extract_images=True,
    images_parser=LLMImageBlobParser(
        model=llm,
        prompt="Describe every figure in detail."
    ),
)



docs = loader.load()
print(type(docs))
print(f"Total chunks: {len(docs)}")

for doc in docs:
    print("\n" + "="*60)
    print(f"Metadata: {doc.metadata}")
    print(f"Content: {doc.page_content}...")