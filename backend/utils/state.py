# utils/state.py
from typing import List, Optional
from pydantic import BaseModel, Field

class RAGState(BaseModel):
    query: str = Field(description="User query")
    expanded_query: str = Field(default="", description="Final query given to LLM")
    answer: str = Field(default="", description="Answer given by LLM")
    pdf_path: Optional[str] = Field(default=None, description="Path to PDF file on disk")
    pdf_ids: Optional[List[str]] = Field(
        default=None,
        description="List of active PDF IDs used for retrieval"
    )
    context: List[dict] = Field(
        default_factory=list,
        description="Retrieved (and compressed) context from vector database"
    )

