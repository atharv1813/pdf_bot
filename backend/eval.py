import json

from main import build_graph
from utils.state import RAGState

from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)


# ===================== SETUP =====================

graph = build_graph()

compiled_graph = graph.compile()

QUESTIONS_DIR = "../data/questions"

def load_json(path: str):
    with open(path) as f:
        return json.load(f)
    
from langchain_aws import ChatBedrockConverse 
from ragas.llms import LangchainLLMWrapper
# ===================== RAGAS EVALUATOR LLM ===================== 
# # Claude Sonnet 4.6 is used as the RAGAS judge/evaluator. 
# # This is separate from the LLM used inside the RAG pipeline, 
# # even though both use the same model.
ragas_evaluator_llm = ChatBedrockConverse(
    model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region_name="us-east-1"
)
from ragas.embeddings import LangchainEmbeddingsWrapper
from main import embeddings as hf_embeddings   # reuse the same HuggingFace embeddings from main.py

ragas_evaluator_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

# ===================== RUN ONE QUESTION THROUGH THE GRAPH =====================
def run_query_for_eval(question: str) -> dict:
    """
    Runs a single question through the compiled graph exactly as a real user
    would — no document_index or file_ids are passed, so retrieval must find
    the right context on its own.
    """
    try:
        initial_state = RAGState(query = question)
        final_state  = compiled_graph.invoke(initial_state)
        
        final_answer = final_state.get("final_answer")
        answer_text = final_answer.answer if final_answer else ""
        contexts = final_state.get("relevant_context") or final_state.get("context") or []
        context_texts = [c.content for c in contexts]

        return {
            "user_input": question,
            "response": answer_text,
            "retrieved_contexts": context_texts,
            "_had_relevant_context": bool(final_state.get("relevant_context")),  # extra field for refusal check
            "_error": None,
        }
    except Exception as e:
        print(f"Failed on question: {question} - {type(e).__name__}: {e}")
        return {
                "user_input": question,
                "response": "",
                "retrieved_contexts": [],
                "_had_relevant_context": False,  # extra field for refusal check
                "_error": str(e),
        }
        
    

# ===================== BUILD EVAL SAMPLES =====================
def build_eval_samples(items: list, has_reference: bool) -> list:
    samples = []
    for item in items:
        sample = run_query_for_eval(item["question"])
        if has_reference:
            sample["reference"] = item["answer"]
        sample["_document_index"] = item.get("document_index")
        samples.append(sample)
        
    return samples

# ===================== FILTER OUT FAILED RUNS BEFORE SCORING =====================
def filter_failed(samples: list)-> list:
    ok = [s for s in samples if s.get("_error") is None]
    failed_count = len(samples) - len(ok)
    if failed_count:
        print(f" {failed_count}/{len(samples)} question failed ans were excluded from scoring")
    return ok

# ===================== CUSTOM REFUSAL CHECK (for no_answer.json) =====================
def check_refusal_rate(samples: list) -> tuple[int, list]:
    refused = 0
    hallucinated = []
    
    for s in samples:
        if not s["_had_relevant_context"]:
            refused += 1
        else:
            hallucinated.append({"question": s["user_input"], "answer": s["response"]})

    print(f"Correctly found 'no relevant context': {refused}/{len(samples)}")
    if hallucinated:
        print("Questions where SOME context was found (check manually for hallucination):")
        for h in hallucinated:
            print(f"  Q: {h['question']}\n  A: {h['answer']}\n")

    return refused, hallucinated



def strip_internal(samples: list) -> list:
    """
    Removes any key starting with '_' from each sample dict before handing
    it to RAGAS — those are our own bookkeeping fields (e.g. _had_relevant_context),
    not part of RAGAS's expected schema (user_input, response, retrieved_contexts, reference).
    """
    cleaned = []
    for sample in samples:
        cleaned_sample = {}
        for key, value in sample.items():
            if not key.startswith("_"):
                cleaned_sample[key] = value
        cleaned.append(cleaned_sample)
    return cleaned


# ===================== MAIN =====================
if __name__ == "__main__":
    single = load_json(f"{QUESTIONS_DIR}/single.json")
    multi = load_json(f"{QUESTIONS_DIR}/multi.json")
    no_ans = load_json(f"{QUESTIONS_DIR}/no_answer.json")
    
    metrics = [ 
               Faithfulness(llm=ragas_evaluator_llm), 
               AnswerRelevancy(llm=ragas_evaluator_llm, embeddings=ragas_evaluator_embeddings), 
               ContextPrecision(llm=ragas_evaluator_llm), 
               ContextRecall(llm=ragas_evaluator_llm) 
            ]
    
    
    # ---- single.json ----
    print("\n=== Running single.json ===")
    single_samples = build_eval_samples(single, has_reference=True)
    single_samples = filter_failed(single_samples)
    single_dataset = EvaluationDataset.from_list(strip_internal(single_samples))
    single_scores = evaluate(dataset = single_dataset, metrics=metrics, llm=ragas_evaluator_llm, embeddings=ragas_evaluator_embeddings)
    print(single_scores)
    single_df = single_scores.to_pandas()
    single_df["document_index"] = [s["_document_index"] for s in single_samples]  # attached AFTER, for analysis only
    print(single_df[["user_input", "faithfulness", "answer_relevancy", "context_precision", "context_recall", "document_index"]])
    
    
    
    # ---- multi.json ----
    print("\n=== Running multi.json ===")
    multi_samples = build_eval_samples(multi, has_reference=True)
    multi_samples = filter_failed(multi_samples)
    multi_dataset = EvaluationDataset.from_list(strip_internal(multi_samples))
    multi_scores = evaluate(dataset = multi_dataset, metrics=metrics, llm=ragas_evaluator_llm, embeddings=ragas_evaluator_embeddings)
    print(multi_scores)
    multi_df = multi_scores.to_pandas()
    multi_df["document_index"] = [s["_document_index"] for s in multi_samples]
    print(multi_df[["user_input", "faithfulness", "answer_relevancy", "context_precision", "context_recall", "document_index"]])



    # ---- no_answer.json ----
    print("\n=== Running no_answer.json ===")
    no_answer_samples = build_eval_samples(no_ans, has_reference=False)
    no_answer_samples = filter_failed(no_answer_samples)
    no_answer_dataset = EvaluationDataset.from_list(strip_internal(no_answer_samples))
    no_answer_scores = evaluate(dataset=no_answer_dataset, metrics=[Faithfulness(llm=ragas_evaluator_llm)], llm=ragas_evaluator_llm, embeddings=ragas_evaluator_embeddings)
    print(no_answer_scores)

    print("\n=== no_answer.json refusal check ===")
    check_refusal_rate(no_answer_samples)
    

