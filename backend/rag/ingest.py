import hashlib
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents.base import Document
from pathlib import Path
from config import VECTOR_DB

BASE_DIR = Path(__file__).resolve().parent.parent.parent / "langchain_dataset_starter"
CORPORA = {
    "langgraph": BASE_DIR / "langgraph",
    "langchain": BASE_DIR / "langchain",
}


# ===================== PER-FILE IDENTITY =====================
def get_file_hash(content: str) -> str:
    """Stable ID based on file content — same content = same hash."""
    return hashlib.md5(content.encode("utf-8")).hexdigest()[:12]


# ===================== MANIFEST-AWARE LOADER =====================
# Each corpus folder carries a manifest.json (file -> title/source) generated
# when the starter set was split out of the full llms-full.txt exports.
# Reading it directly (instead of a blind DirectoryLoader glob) means every
# chunk keeps its real page title/source/corpus instead of just a filename.
def load_corpus(corpus_dir: Path, corpus_name: str) -> List[Document]:
    manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    docs: List[Document] = []
    for entry in manifest:
        text = (corpus_dir / entry["file"]).read_text(encoding="utf-8")
        if not text.strip():
            continue  # a few starter pages came out empty from the original split
        docs.append(Document(
            page_content=text,
            metadata={
                "file_id": get_file_hash(text),
                "corpus": corpus_name,
                "title": entry.get("title", entry.get("source_path", entry["file"])),
                "source": entry.get("source", entry.get("source_path", "")),
            },
        ))
    return docs


def run_ingestion():
    raw_docs: List[Document] = []
    for corpus_name, corpus_dir in CORPORA.items():
        raw_docs.extend(load_corpus(corpus_dir, corpus_name))

    # ===================== TEXT SPLITTER =====================
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_docs: List[Document] = text_splitter.split_documents(raw_docs)

    existing_ids: set = set()

    try:
        existing = VECTOR_DB.get()
        if existing and existing.get('metadatas'):
            existing_ids = {
                m["file_id"]
                for m in existing["metadatas"]
                if m.get("file_id") # skip any chunk missing a file ids just in case
            }
    except Exception:
        
        pass

    new_chunks: List[Document] = [
        c for c in chunked_docs
        if c.metadata["file_id"] not in existing_ids
    ]

    if new_chunks:
        VECTOR_DB.add_documents(new_chunks)
        new_file_count = len(set(c.metadata["file_id"] for c in new_chunks))
        print(f"💾 Ingested {len(new_chunks)} new chunks across {new_file_count} file(s)")
    else:
        print("💾 All files already ingested — skipping")