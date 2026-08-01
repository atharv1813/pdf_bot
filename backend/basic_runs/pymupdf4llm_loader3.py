from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os

loader = PyMuPDF4LLMLoader(
    "/home/atharv/GenAI/study_vedya/langgraph/projects/chat_pdf/backend/data/attention_paper.pdf",
    mode="page",
    extract_images=True,
    images_parser=LLMImageBlobParser(
        model=ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY")
        ),
        prompt="Describe the content of each image in a few sentences."
    ),
)



docs = loader.load()
print(type(docs))
print(f"Total chunks: {len(docs)}")

for doc in docs:
    print("\n" + "="*60)
    print(f"Metadata: {doc.metadata}")
    print(f"Content: {doc.page_content}...")