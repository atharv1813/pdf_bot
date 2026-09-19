from typing import List
from langgraph.graph import StateGraph, START, END
from rag.state import Candidate_answer, RAGState, Support_state, Usefulness_state, Retrieved_docs
from config import model, VECTOR_DB
from dotenv import load_dotenv
load_dotenv()


    
# ===================== GRAPH NODES and ROUTING FUNCTIONS =====================
from pydantic import BaseModel, Field
from typing import Literal, List

class Decision_retrieval(BaseModel):
    decision: bool = Field(description = "decide whether if query needs retriever or not? ")
    

def decide_retrieval_node(state: RAGState) -> dict:
    query = state.query
    prompt = f"""
        Assume you are a professinal model redirecter and for given query you can decide whether this query needs external data before generating an answer
        query: {query}
    """
    decider_model = model.with_structured_output(Decision_retrieval)
    
    decision_res = decider_model.invoke(prompt)
    
    return {"query_retrieval_decision" : decision_res.decision}

def route_retrieval_decision(state:RAGState) -> Literal['retrieve', 'dont_retrieve']:
    decision = state.query_retrieval_decision
    if (decision == True):
        return 'retrieve'
    else:
        return 'dont_retrieve'
    
def retrieve_node(state: RAGState) -> dict:

    # 3. Resolve active file_ids
    active_file_ids = state.file_ids if state.file_ids else None
    
    RETRIEVAL_K = 15   # wider net; relevance grading is now one batched call, not one per doc
    
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
    
    result = model.with_structured_output(Candidate_answer).invoke(prompt)
    
    return {"final_answer": result}

class Is_relevant_batch(BaseModel):
    relevant_indices: List[int] = Field(
        description="Indices (0-based, from the numbered document list) of every document that is relevant to the query. Empty list if none are relevant."
    )

def is_relevant_node(state: RAGState) -> dict:
    # One structured-output call graded over every retrieved doc at once,
    # instead of one Bedrock round-trip per doc — the same k=15 docs that
    # used to cost 15 sequential calls now cost exactly 1.
    if not state.context:
        return {"relevant_context": []}

    relevancy_model = model.with_structured_output(Is_relevant_batch)
    numbered_docs = "\n\n".join(
        f"Document {i}:\n{doc.content}" for i, doc in enumerate(state.context)
    )
    ans = relevancy_model.invoke(
        f"""
            Given the query below and a numbered list of candidate documents, determine
            which documents (if any) are relevant to answering the query.

            Query:
            {state.query}

            {numbered_docs}

            Return the 0-based indices of every relevant document. Return an empty
            list if none of the documents are relevant.
            """
    )

    relevant = [
        state.context[i]
        for i in ans.relevant_indices
        if 0 <= i < len(state.context)
    ]
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
    
    result = model.with_structured_output(Candidate_answer).invoke(prompt)
    
    return {"candidate_answer": result}

def is_supported_node(state: RAGState) -> dict:
    query = state.query
    candidate_ans = state.candidate_answer
    relevant_docs = state.relevant_context

    support_model = model.with_structured_output(Support_state)

    prompt = f"""
        You have given a query: {query}
        and a candidate answer for the query: {candidate_ans.answer}
        Now you have to determine whether this answer is supported by the retrieved
        documents or not. Documents used for answer: {[d.content for d in relevant_docs]}
    """

    result: Support_state = support_model.invoke(prompt)   # ONE call, ONE object back

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
    
    result: Candidate_answer = model.with_structured_output(Candidate_answer).invoke(prompt)
    
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
    
    usefulness_model = model.with_structured_output(Usefulness_state)
    
    result: Usefulness_state = usefulness_model.invoke(prompt)
    
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
    result:Candidate_answer = model.with_structured_output(Candidate_answer).invoke(prompt)
    
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
    
  