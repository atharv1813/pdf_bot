from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from utils.state import RAGState, Support_state, Usefulness_state

from dotenv import load_dotenv
load_dotenv()

from langchain_aws import ChatBedrockConverse
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# ===========P LLM MODEL SETUP =====================
llm = ChatBedrockConverse(
    model="amazon.nova-lite-v1:0",
    region_name="us-east-1"
)


# ===================== PDF Identity maker =====================
import hashlib
def get_pdf_hash(pdf_path: str) -> str:
    """Stable ID based on file content — same file = same hash, different file = different hash."""
    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:12]


# ===================== VECTOR DB =====================
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "pdf_chunked_docs"
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
raw_docs = loader.load()


# ===================== TEXT SPLITTER =====================
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunked_docs = text_splitter.split_documents(raw_docs)
    
    


from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class Decision_retrieval(BaseModel):
    decision: bool = Field(description = "decide whether if query needs retriever or not? ")
    

def decide_retrieval_node(state: RAGState) -> RAGState:
    query = state.query
    prompt = f"""
        Assume you are a professinal LLM redirecter and for given query you can decide whether this query needs external data before genrating an answer
        query: {query}
    """
    decider_llm = llm.with_structured_output(Decision_retrieval)
    
    state["query_retrieval_decision"] = decider_llm.invoke(prompt)
    
    return state["query_retrieval_decision"]

    
    
def retrieve_node(state: RAGState) -> RAGState:
    pdf_hash = get_pdf_hash(state.pdf_path)
    existing = VECTOR_DB.get(where={"pdf_id": pdf_hash})
    if existing["ids"]:
        print(f"💾 Already ingested ({len(existing['ids'])} chunked_docs) — skipping")
    else:
        for chunk in chunked_docs:
            chunk.metadata["pdf_id"] = pdf_hash
        VECTOR_DB.add_documents(chunked_docs)
        print(f"💾 Ingested {len(chunked_docs)} chunked_docs with pdf_id={pdf_hash}")

    # 3. Resolve active pdf_ids
    active_pdf_ids = state.pdf_ids if state.pdf_ids else [pdf_hash]
    
    RETRIEVAL_K = 15   # wider net; reranker will trim this down later

    # 4a. Vector retriever — semantic similarity (Chroma)
    vector_retriever = VECTOR_DB.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": RETRIEVAL_K,
            "filter": {"pdf_id": {"$in": active_pdf_ids}}
        }
    )
    state.context = vector_retriever.invoke(state.query)
    
    return state 
    
    
    
    

def generate_directly_node(state: RAGState) -> RAGState:
    query = state.query
    
    prompt = f"""
        Given the query: {query}. Answer the following query.  
    """
    
    state["final_answer"] = llm.invoke(prompt)
    
    return state["final_answer"]

class Is_relevant(BaseModel):
    is_relevant: bool = Field(description="is the docuemnt relevant to the query")
    
def is_relevant_node(state: RAGState) -> RAGState:
    query = state.query
    context = state["context"]
    relevancy_llm = llm.with_structured_output(Is_relevant)
    
    
    for i in range(len(context)):
        prompt = f"""
                    Given the query: {query} tell whether if given context {context[i]} is
                    relevant to the query or not 
                """
        ans = relevancy_llm.invoke(prompt)
        if ans:
            state.relevant_context.append(context[i])
            
    return state.relevant_context
        
    

def generate_from_context_node(state: RAGState) -> RAGState:
    query = state.query
    relevant_context = state.relevant_context
    prompt = """
        Given the user query: {query}.
        and the retrieved context: {relevant_context}
        I want you to generate answer from the given query and context. Make sure to 
        stick to the context to gather or interpret facts and write answer 
    """
    
    state.candidate_answer = llm.invoke(prompt)
    
    return state.candidate_answer


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

    updated_answer = candidate_ans.model_copy(update={"support_state": result}) 
    # because llm only return parts of level0 state key not entire so first update 
    # that key then return that key in a dict 

    return {"candidate_answer": updated_answer}   # return a dict, key = state field name

def revise_answer_node(state: RAGState) -> RAGState:
    query = state.query
    candidate_answer = state.candidate_answer
    feedback = state.is_supported_feedback
    docs = state.relevant_context
    
    prompt = f""""
        You are a answer reveiwer based on the feedback of the previous answer.
        query was: {query}.
        previous answer was: {candidate_answer}.
        Now after judgement for above answer it wasnt matching with retrieved docs so the
        feedback for that response is: {feedback}.
        for reference docs cotext is: {docs}
    """
    
    state.candidate_answer = llm.invoke(prompt)
    state.is_supported_feedback = None
    state.is_supported = None
    
    return {state.candidate_answer, state.is_supported_feedback, state.is_supported}
  
    
def is_useful_node(state: RAGState) -> RAGState:
    query = state.query
    final_ans = state["final_answer"]
    
    prompt = f"""
        You are LLM answer reviewer. You will be given query given to them LLM 
        and its answer given by the LLM. You have provide score between 1 to 5 
        for that answer and feedback for that answer.
        query : {query}
        final_ans: {final_ans}
        
        make sure to give both the score and the feedback
    """
    
    usefulness_llm = llm.with_structured_output(Usefulness_state)
    
    state["is_useful"], state["is_useful_feedback"] = usefulness_llm.invoke(prompt)
    
    return {state["is_useful"], state["is_useful_feedback"]}
    

def no_doc_useful_node(state: RAGState) -> RAGState:
    query = state.query
    prompt = f"""
        You have to tell the user that for your query:{query} and retriever DB.
        we couldnt find any relevant context so can you reqrite the query or do you 
        want to call web search to get the results
    """
    state["final_answer"] = llm.invoke(prompt)
    
    return state["final_answer"] 


    
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
    
    graph.add_edge(START, decide_retrieval_node)
    graph.add_edge(is_useful_node, END)
    
    return graph
    
  
    
# if __name__ == "__main__":
#     pdf_path = "example.pdf"
#     pdf_hash = get_pdf_hash(pdf_path)
#     print(f"PDF Hash: {pdf_hash}")
    
#     graph  = build_graph()
    
#     graph = graph.compile()
    
#     initial_state = {
#         "query": "what is relativity? ",
#         ""
#     }
    
#     final_state = graph.invoke(initial_state)
    
    
    