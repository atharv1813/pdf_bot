from langchain_aws import ChatBedrockConverse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

#=========== LLM MODEL SETUP =====================
model = ChatBedrockConverse(
    model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region_name="us-east-1"
)


# ===================== VECTOR DB =====================
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "file_chunked_docs"
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

VECTOR_DB = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR
)

