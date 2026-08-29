from typing import List

from langchain_core.documents.base import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from utils.state import Candidate_answer, RAGState, Support_state, Usefulness_state, Retrieved_docs

from dotenv import load_dotenv
load_dotenv()

from langchain_aws import ChatBedrockConverse
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# ===========P LLM MODEL SETUP =====================
llm = ChatBedrockConverse(
    model="amazon.nova-lite-v1:0",
    region_name="us-east-1"
)

import hashlib 
# ===================== PER-FILE IDENTITY =====================
def get_file_hash(content: str) -> str:
    """Stable ID based on file content — same content = same hash."""
    return hashlib.md5(content.encode("utf-8")).hexdigest()[:12]


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
        
# ===================== Docuemnt loader =====================
loader = DirectoryLoader(
    "../data/documents",
    glob="*.txt",
    loader_cls=TextLoader
)
raw_docs: List[Document] = loader.load()

for doc in raw_docs:
    doc.metadata["file_id"] = get_file_hash(doc.page_content)
    

# ===================== TEXT SPLITTER =====================
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunked_docs: List[Document] = text_splitter.split_documents(raw_docs)
    
# ===================== ONE-TIME INGESTION =====================
# Goal: only add chunks to the vector DB for files that AREN'T already
# stored there — so re-running this script never duplicates data.

# Will hold the pdf_id of every FILE (not chunk) already present in the DB,
# collected from a previous run. Starts empty.
existing_ids: set = set()

try:
    # Ask Chroma for everything currently in the collection.
    # On the very first run (empty DB), this may behave oddly depending on
    # the Chroma version — hence wrapping it in try/except below.
    existing = VECTOR_DB.get()
    if existing and existing.get('metadatas'):
        # existing["metadatas"] is a list with ONE dict per chunk already
        # stored, e.g. [{"pdf_id": "abc123", "source": "0.txt"}, {"pdf_id": "abc123", ...}, ...]
        #
        # Many chunks share the same pdf_id (all chunks from the same file).
        # Using a set comprehension automatically de-duplicates those down
        # to one entry per FILE, not per chunk.
        existing_ids = {
            m["file_id"]
            for m in existing["metadatas"]
            if m.get("file_id") # skip any chunk missing a file ids just in case
        }
except Exception:
    # If VECTOR_DB.get() fails (e.g. brand-new empty DB), fall back to
    # treating everything as new — existing_ids just stays empty.
    pass

# chunked_docs = ALL chunks from ALL files loaded this run (every chunk
# carries its parent file's pdf_id in its metadata, from earlier tagging).
#
# Keep only the chunks whose pdf_id is NOT already in existing_ids —
# i.e., chunks belonging to files we haven't ingested before.
# - First ever run: existing_ids is empty -> every chunk is "new".
# - Re-run with no new files: every pdf_id is already known -> new_chunks = [].
# - Re-run after adding file #21: only that file's chunks pass the filter.
new_chunks: List[Document] = [
    c for c in chunked_docs
    if c.metadata["file_id"] not in existing_ids
]

if new_chunks:
    # Only touch the DB if there's actually something new to embed/add —
    # avoids a pointless call when nothing changed.
    VECTOR_DB.add_documents(new_chunks)
    
    # Just for a friendly log message: how many DISTINCT files did these
    # new chunks come from (de-duplicated via set(), same trick as above).
    new_file_count = len(set(c.metadata["file_id"] for c in new_chunks))
    print(f"💾 Ingested {len(new_chunks)} new chunks across {new_file_count} file(s)")
else:
    # Every file's pdf_id was already found in the DB -> nothing to add.
    print("💾 All files already ingested — skipping")
    
    
    
    
# ===================== GRAPH NODES and ROUTING FUNCTIONS =====================
from pydantic import BaseModel, Field
from typing import Literal, List

class Decision_retrieval(BaseModel):
    decision: bool = Field(description = "decide whether if query needs retriever or not? ")
    

def decide_retrieval_node(state: RAGState) -> dict:
    query = state.query
    prompt = f"""
        Assume you are a professinal LLM redirecter and for given query you can decide whether this query needs external data before generating an answer
        query: {query}
    """
    decider_llm = llm.with_structured_output(Decision_retrieval)
    
    decision_res = decider_llm.invoke(prompt)
    
    return {"query_retrieval_decision" : decision_res.decision}

def route_retrieval_decision(state:RAGState) -> Literal['retrieve', 'dont_retrieve']:
    decision = state.query_retrieval_decision
    if (decision == True):
        return 'retrieve'
    else:
        return 'dont_retrieve'
    
def retrieve_node(state: RAGState) -> dict:
    # file_hash = get_file_hash(state.file_path)
    # existing = VECTOR_DB.get(where={"file_id": file_hash})
    # if existing["ids"]:
    #     print(f"💾 Already ingested ({len(existing['ids'])} chunked_docs) — skipping")
    # else:
    #     for chunk in chunked_docs:
    #         chunk.metadata["file_id"] = file_hash
    #     VECTOR_DB.add_documents(chunked_docs)
    #     print(f"💾 Ingested {len(chunked_docs)} chunked_docs with file_id={file_hash}")

    # 3. Resolve active file_ids
    active_file_ids = state.file_ids if state.file_ids else None
    
    RETRIEVAL_K = 15   # wider net; reranker will trim this down later
    
    search_kwargs: dict[str, int | str] = {"k": RETRIEVAL_K}
    if active_file_ids:
        search_kwargs["filter"] = {"file_id": {"$in": active_file_ids}}
        
    

    # 4a. Vector retriever — semantic similarity (Chroma)
    vector_retriever = VECTOR_DB.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )
    retrieved = vector_retriever.invoke(state.query)
    
    converted: List[Retrieved_docs] = [
    Retrieved_docs(
            doc_id=d.metadata.get("file_id", ""),   # pull file_id out of LangChain's metadata dict
            content=d.page_content,                 # map page_content -> your `content` field
            metadata=d.metadata                     # carry the whole metadata dict along too
        )
        for d in retrieved   # loop over each LangChain Document returned by the retriever
    ]
    
    return {"context": converted}
    
    

def generate_directly_node(state: RAGState) -> dict:
    query = state.query
    
    prompt = f"""
        Given the query: {query}. Answer the following query.  
    """
    
    result = llm.with_structured_output(Candidate_answer).invoke(prompt)
    
    return {"final_answer": result}

class Is_relevant(BaseModel):
    is_relevant: bool = Field(description="is the docuemnt relevant to the query")
    
def is_relevant_node(state: RAGState) -> dict:
    relevancy_llm = llm.with_structured_output(Is_relevant)
    relevant = []
    for doc in state.context:
        ans = relevancy_llm.invoke(f"Query: {state.query}\nDoc: {doc.content}\nRelevant?")
        if ans.is_relevant:
            relevant.append(doc)
            
    return {"relevant_context": relevant}
        
def is_relevant_decision(state: RAGState) -> Literal['yes_rel', 'not_rel']:
    return 'yes_rel' if state.relevant_context else 'not_rel'
    


def generate_from_context_node(state: RAGState) -> dict:
    query = state.query
    relevant_context: List[Retrieved_docs] = state.relevant_context
    relevant_context_docs: str = "\n\n".join(d.content for d in relevant_context)
    prompt = f"""
        Given the user query: {query}.
        and the retrieved context: {relevant_context_docs}
        I want you to generate answer from the given query and context. Make sure to 
        stick to the context to gather or interpret facts and write answer 
    """
    
    result = llm.with_structured_output(Candidate_answer).invoke(prompt)
    
    return {"candidate_answer": result}

def is_supported_node(state: RAGState) -> dict:
    query = state.query
    candidate_ans = state.candidate_answer
    relevant_docs = state.relevant_context

    support_llm = llm.with_structured_output(Support_state)

    prompt = f"""
        You have given a query: {query}
        and a candidate answer for the query: {candidate_ans.answer}
        Now you have to determine whether this answer is supported by the retrieved
        documents or not. Documents used for answer: {[d.content for d in relevant_docs]}
    """

    result: Support_state = support_llm.invoke(prompt)   # ONE call, ONE object back

    updated_answer:Candidate_answer = candidate_ans.model_copy(update={"support_state": result}) 
    # because llm only return parts of level0 state key not entire so first update 
    # that key then return that key in a dict 

    return {"candidate_answer": updated_answer}   # return a dict, key = state field name

def is_supported_decision(state: RAGState) -> Literal['fully', 'not_fully']:
    decision: Literal['fully'] | Literal['partially'] | Literal['no'] = state.candidate_answer.support_state.is_supported
    if decision == 'fully':
        return 'fully'
    if state.retries >= 5:
        return 'fully'   # give up retrying, proceed with best-effort answer
    return 'not_fully' # can be no or partially supported in both go to revise answer

def revise_answer_node(state: RAGState) -> dict:
    query: str = state.query
    candidate_answer: Candidate_answer | None = state.candidate_answer
    feedback = state.candidate_answer.support_state.is_supported_feedback
    docs: List[Retrieved_docs] = state.relevant_context
    
    prompt: str = f""""
        You are a answer reveiwer based on the feedback of the previous answer.
        query was: {query}.
        previous answer was: {candidate_answer.answer}.
        Now after judgement for above answer it wasnt matching with retrieved docs so the
        feedback for that response is: {feedback}.
        for reference docs cotext is: {docs}
    """
    
    retries: int = state.retries + 1
    
    result: Candidate_answer = llm.with_structured_output(Candidate_answer).invoke(prompt)
    
    return {"candidate_answer" : result, "retries": retries}
  
    
def is_useful_node(state: RAGState) -> dict:
    query: str = state.query
    final_ans: str = state.candidate_answer.answer
    
    prompt: str = f"""
        You are LLM answer reviewer. You will be given query given to them LLM 
        and its answer given by the LLM. You have provide score between 1 to 5 
        for that answer and feedback for that answer.
        query : {query}
        final_ans: {final_ans}
        
        make sure to give both the score and the feedback
    """
    
    usefulness_llm = llm.with_structured_output(Usefulness_state)
    
    result: Usefulness_state = usefulness_llm.invoke(prompt)
    
    updated_ans: Candidate_answer = state.candidate_answer.model_copy(update={"usefulness_state": result})
    
    
    return {"candidate_answer" : updated_ans, "final_answer": updated_ans}
    
def is_useful_decision(state: RAGState) -> Literal['yes_useful', 'not_useful']:
    decision: Literal['1'] | Literal['2'] | Literal['3'] | Literal['4'] | Literal['5'] = state.candidate_answer.usefulness_state.is_useful
    if(int(decision) >= 4):
        return 'yes_useful'
    else:
        return 'not_useful'

def no_doc_useful_node(state: RAGState) -> dict:
    query: str = state.query
    prompt: str = f"""
        You have to tell the user that for your query:{query} and retriever DB.
        we couldnt find any relevant context so can you reqrite the query or do you 
        want to call web search to get the results
    """
    result:Candidate_answer = llm.with_structured_output(Candidate_answer).invoke(prompt)
    
    return {"final_answer": result}



# ===================== FUNCTION BUILDING GRAPH AND ITS ROUTING PATTERNS =====================
    
def build_graph():
    graph = StateGraph(RAGState)
    
    graph.add_node("decide_retrieval", decide_retrieval_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate_directly", generate_directly_node)
    graph.add_node("is_relevant", is_relevant_node)
    graph.add_node("generate_from_context", generate_from_context_node)
    graph.add_node("is_supported", is_supported_node)
    graph.add_node("revise_answer", revise_answer_node)
    graph.add_node("is_useful", is_useful_node)
    graph.add_node("no_doc_useful", no_doc_useful_node)
    
    graph.add_edge(START, "decide_retrieval")
    graph.add_conditional_edges(
        "decide_retrieval", 
        route_retrieval_decision,
        {
            "retrieve": "retrieve",
            "dont_retrieve": "generate_directly"
        }
        )
    graph.add_edge("generate_directly", END)
    graph.add_edge("retrieve", "is_relevant")
    graph.add_conditional_edges(
        "is_relevant",
        is_relevant_decision,
        {
            "yes_rel": "generate_from_context",
            "not_rel": "no_doc_useful"
        }
    )
    graph.add_edge("generate_from_context", "is_supported")
    graph.add_conditional_edges(
        "is_supported",
        is_supported_decision,
        {
            "fully": "is_useful",
            "not_fully": "revise_answer"
        }
        )
    graph.add_edge("revise_answer", "is_supported")
    graph.add_conditional_edges(
        "is_useful",
        is_useful_decision,
        {
            "yes_useful": END,
            "not_useful": "no_doc_useful"
        }
    )
    graph.add_edge("no_doc_useful", END)
    
    return graph
    
  
    
if __name__ == "__main__":
    
    graph  = build_graph()
    
    compiled_graph = graph.compile()
    
    png_bytes = compiled_graph.get_graph().draw_mermaid_png()

    with open("graph.png", "wb") as f:
        f.write(png_bytes)
        
    test_queries = [
        "What is relativity?",
        "What do keybullet kin drop?"
    ]

    for q in test_queries:
        print(f"\n--- Query: {q} ---")
        initial_state = RAGState(query=q)
        result = compiled_graph.invoke(initial_state)

        print(f"Used retrieval: {result['query_retrieval_decision']}")

        final_answer = result.get("final_answer")
        if final_answer:
            print(f"Answer: {final_answer.answer}")
            if final_answer.support_state:
                print(f"Support: {final_answer.support_state.is_supported}")
            if final_answer.usefulness_state:
                print(f"Usefulness: {final_answer.usefulness_state.is_useful}/5")
        else:
            print("No final answer was produced.")
    
    
    