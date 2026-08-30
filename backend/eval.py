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
    
    


# ===================== RUN ONE QUESTION THROUGH THE GRAPH =====================
def run_query_for_eval(question: str) -> dict:
    """
    Runs a single question through the compiled graph exactly as a real user
    would — no document_index or file_ids are passed, so retrieval must find
    the right context on its own.
    """
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
    }
    

# ===================== BUILD EVAL SAMPLES =====================
def build_eval_samples(items: list, has_reference: bool) -> list:
    samples = []
    for item in items:
        sample = run_query_for_eval(item["question"])
        if has_reference:
            sample["reference"] = item["answer"]
        samples.append(sample)
        
    return samples

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
    
    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]
    
    
    # ---- single.json ----
    print("\n=== Running single.json ===")
    single_samples = build_eval_samples(single, has_reference=True)
    single_dataset = EvaluationDataset.from_list(strip_internal(single_samples))
    single_scores = evaluate(dataset = single_dataset, metrics=metrics)
    print(single_scores)
    single_df = single_scores.to_pandas()
    single_df["document_index"] = [item["document_index"] for item in single]  # attached AFTER, for analysis only
    print(single_df[["user_input", "faithfulness", "answer_relevancy", "context_precision", "context_recall", "document_index"]])
    
    
    
    # ---- multi.json ----
    print("\n=== Running multi.json ===")
    multi_samples = build_eval_samples(multi, has_reference=True)
    multi_dataset = EvaluationDataset.from_list(strip_internal(multi_samples))
    multi_scores = evaluate(dataset = multi_dataset, metrics=metrics)
    print(multi_scores)
    multi_df = multi_scores.to_pandas()
    multi_df["document_index"] = [item["document_index"] for item in multi]
    print(multi_df[["user_input", "faithfulness", "answer_relevancy", "context_precision", "context_recall", "document_index"]])



    # ---- no_answer.json ----
    print("\n=== Running no_answer.json ===")
    no_answer_samples = build_eval_samples(no_ans, has_reference=False)
    no_answer_dataset = EvaluationDataset.from_list(strip_internal(no_answer_samples))
    no_answer_scores = evaluate(dataset=no_answer_dataset, metrics=[Faithfulness()])
    print(no_answer_scores)

    print("\n=== no_answer.json refusal check ===")
    check_refusal_rate(no_answer_samples)
    

