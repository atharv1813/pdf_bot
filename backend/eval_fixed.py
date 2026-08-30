import json

from main import build_graph
from main import llm as pipeline_llm
from main import embeddings as hf_embeddings
from utils.state import RAGState

from pydantic import BaseModel, Field

from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_aws import ChatBedrockConverse


# ===================== SETUP =====================

graph = build_graph()
compiled_graph = graph.compile()

QUESTIONS_DIR = "../data/questions"


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


# ===================== RAGAS EVALUATOR LLM + EMBEDDINGS =====================
# Claude Sonnet 4.5 is used as the RAGAS judge/evaluator.
# This is separate from the LLM used inside the RAG pipeline (main.py's `llm`),
# even though both currently point at the same model.
ragas_evaluator_llm = ChatBedrockConverse(
    model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region_name="us-east-1"
)
ragas_evaluator_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

# Controls RAGAS's internal concurrency. Bedrock enforces a transactions-per-second
# quota per model, and RAGAS's default concurrency (many parallel jobs) blows through
# that quota, causing widespread ThrottlingException failures and NaN scores.
# Lowering max_workers trades speed for reliability; raise max_retries so any
# still-throttled call gets more chances to eventually succeed.
run_config = RunConfig(
    max_workers=2,
    timeout=180,
    max_retries=8,
)


# ===================== ANSWER COMMITMENT CLASSIFIER (for no_answer.json) =====================
class Answer_commitment(BaseModel):
    made_confident_claim: bool = Field(
        description="True if the response asserts specific facts as if answering the question "
                    "directly. False if the response declines, hedges, says it couldn't find "
                    "information, or explicitly states the question can't be answered from "
                    "available context."
    )


commitment_llm = pipeline_llm.with_structured_output(Answer_commitment)


def classify_commitment(question: str, response: str) -> bool:
    """Returns True if the response committed to a confident specific answer."""
    if not response.strip():
        return False
    prompt = f"""
        Question: {question}
        Response: {response}

        Does this response assert specific facts as a confident answer to the question,
        or does it decline / hedge / say the information isn't available?
    """
    result = commitment_llm.invoke(prompt)
    return result.made_confident_claim


# ===================== RUN ONE QUESTION THROUGH THE GRAPH =====================
def run_query_for_eval(question: str) -> dict:
    """
    Runs a single question through the compiled graph exactly as a real user
    would — no document_index or file_ids are passed, so retrieval must find
    the right context on its own.

    Wrapped in try/except so a transient LLM/tool-call failure on one question
    doesn't crash the entire eval run and lose everything computed so far.
    """
    try:
        initial_state = RAGState(query=question)
        final_state = compiled_graph.invoke(initial_state)

        final_answer = final_state.get("final_answer")
        answer_text = final_answer.answer if final_answer else ""
        contexts = final_state.get("relevant_context") or final_state.get("context") or []
        context_texts = [c.content for c in contexts]

        return {
            "user_input": question,
            "response": answer_text,
            "retrieved_contexts": context_texts,
            "_had_relevant_context": bool(final_state.get("relevant_context")),
            "_error": None,
        }
    except Exception as e:
        print(f"⚠️  Failed on question: {question!r} — {type(e).__name__}: {e}")
        return {
            "user_input": question,
            "response": "",
            "retrieved_contexts": [],
            "_had_relevant_context": False,
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
def filter_failed(samples: list) -> list:
    ok = [s for s in samples if s.get("_error") is None]
    failed_count = len(samples) - len(ok)
    if failed_count:
        print(f"⚠️  {failed_count}/{len(samples)} question(s) failed and were excluded from scoring")
    return ok


# ===================== REFUSAL / HALLUCINATION-RISK CHECK (for no_answer.json) =====================
def check_refusal_rate(samples: list) -> dict:
    """
    Classifies each no_answer.json sample into three categories:
      - correctly_refused: no relevant context was found (retrieval correctly found nothing)
      - declined_gracefully: context was found, but the response still declined/hedged
      - confident_hallucination_risk: context was found AND the response committed to a
        specific confident answer — these need manual verification against retrieved_contexts

    _had_relevant_context (retrieval-level) and answer commitment (generation-level) are
    different signals — a system can find tangential context and still correctly decline,
    so both are checked rather than treating "found some context" as equivalent to hallucination.
    """
    correctly_refused = []
    declined_gracefully = []
    confident_hallucination_risk = []

    for s in samples:
        had_context = s["_had_relevant_context"]

        if not had_context:
            correctly_refused.append(s)
            continue

        committed = classify_commitment(s["user_input"], s["response"])
        if committed:
            confident_hallucination_risk.append(s)
        else:
            declined_gracefully.append(s)

    total = len(samples)
    print(f"Correctly refused (no context found): {len(correctly_refused)}/{total}")
    print(f"Found context but declined/hedged anyway: {len(declined_gracefully)}/{total}")
    print(f"Found context AND gave a confident answer (needs manual review): {len(confident_hallucination_risk)}/{total}")

    if confident_hallucination_risk:
        print("\nQuestions needing manual review against retrieved_contexts:")
        for s in confident_hallucination_risk:
            print(f"  Q: {s['user_input']}\n  A: {s['response']}\n")

    return {
        "correctly_refused": correctly_refused,
        "declined_gracefully": declined_gracefully,
        "confident_hallucination_risk": confident_hallucination_risk,
    }


def strip_internal(samples: list) -> list:
    """
    Removes any key starting with '_' from each sample dict before handing
    it to RAGAS — those are our own bookkeeping fields (e.g. _had_relevant_context,
    _error, _document_index), not part of RAGAS's expected schema
    (user_input, response, retrieved_contexts, reference).
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
    single_scores = evaluate(
        dataset=single_dataset,
        metrics=metrics,
        llm=ragas_evaluator_llm,
        embeddings=ragas_evaluator_embeddings,
        run_config=run_config
    )
    print(single_scores)
    single_df = single_scores.to_pandas()
    single_df["document_index"] = [s["_document_index"] for s in single_samples]
    print(single_df[["user_input", "faithfulness", "answer_relevancy", "context_precision", "context_recall", "document_index"]])

    # ---- multi.json ----
    print("\n=== Running multi.json ===")
    multi_samples = build_eval_samples(multi, has_reference=True)
    multi_samples = filter_failed(multi_samples)
    multi_dataset = EvaluationDataset.from_list(strip_internal(multi_samples))
    multi_scores = evaluate(
        dataset=multi_dataset,
        metrics=metrics,
        llm=ragas_evaluator_llm,
        embeddings=ragas_evaluator_embeddings,
        run_config=run_config
    )
    print(multi_scores)
    multi_df = multi_scores.to_pandas()
    multi_df["document_index"] = [s["_document_index"] for s in multi_samples]
    print(multi_df[["user_input", "faithfulness", "answer_relevancy", "context_precision", "context_recall", "document_index"]])

    # ---- no_answer.json ----
    print("\n=== Running no_answer.json ===")
    no_answer_samples = build_eval_samples(no_ans, has_reference=False)
    no_answer_samples = filter_failed(no_answer_samples)
    no_answer_dataset = EvaluationDataset.from_list(strip_internal(no_answer_samples))
    no_answer_scores = evaluate(
        dataset=no_answer_dataset,
        metrics=[Faithfulness(llm=ragas_evaluator_llm)],
        llm=ragas_evaluator_llm,
        embeddings=ragas_evaluator_embeddings,
        run_config=run_config
    )
    print(no_answer_scores)

    print("\n=== no_answer.json refusal / hallucination-risk check ===")
    check_refusal_rate(no_answer_samples)