## Theory behind this pipeline — in depth, building on BM25 + vector search

You already know BM25 and vector search individually. What this code is actually doing is solving a *harder* problem than either one alone: **how do you combine two retrieval signals that measure fundamentally different things, without one drowning out the other, and then how do you clean up the fused result before it reaches the LLM.** Let's go through each theoretical layer the code implements, in the order data flows through it.

---

### 1. Why hybrid retrieval exists at all — the theoretical gap between BM25 and vectors

BM25 and dense vector search are not two versions of the same idea — they fail in *opposite, complementary* ways.

- **BM25 fails on synonymy and paraphrase.** If the document says "the Treaty of Versailles imposed reparations" and the query says "financial penalties after WW1," BM25 sees almost zero lexical overlap and scores it low — even though it's exactly the right chunk. BM25 has no concept of *meaning*, only surface tokens.
- **Vector search fails on precision and rare/exact terms.** Embedding models compress meaning into a fixed-size vector, and in that compression, specific identifiers — model numbers, exact names, rare technical terms, numbers, acronyms — get blurred together with semantically "nearby" concepts. If your query contains an exact clause name or a specific date, a dense vector might rank a *topically similar but factually wrong* chunk above the one containing the exact match.

These failure modes are **statistically close to independent** — a chunk that BM25 misses due to paraphrase is often exactly the kind of chunk vector search *catches*, and vice versa. This is the core theoretical justification for hybrid retrieval: when two weak, independent, imperfect signals both point at the same document, your confidence in that document should compound. When they disagree, you want a fusion method that doesn't let one signal's failure mode silently veto the other's success.

This is exactly what `rag_node_hybrid` sets up: two retrievers built over the *same underlying chunk set*, run independently, each returning its own ranked list.

### 2. Why the BM25 corpus is rebuilt from Chroma metadata, not stored separately

Look at this part of the code:

```python
raw = VECTOR_DB.get(where={"pdf_id": {"$in": active_pdf_ids}}, include=["documents", "metadatas"])
bm25_corpus = [Document(page_content=text, metadata=meta) for text, meta in zip(raw["documents"], raw["metadatas"])]
bm25_retriever = BM25Retriever.from_documents(bm25_corpus)
```

This reflects an important structural fact about BM25 that vector search doesn't share: **BM25 is corpus-relative, not chunk-relative.** Recall the IDF term — $df(t)$ and $N$ are properties of the *whole corpus*, not any single chunk. A term's IDF changes depending on what other documents exist alongside it. This means BM25 can't be precomputed and stored per-chunk the way an embedding vector can (an embedding is a fixed representation of one chunk in isolation, computed once and stored forever). BM25 has to be **rebuilt in memory, scoped to exactly the active corpus**, every time the query set changes — which is why the code pulls the raw text for the *specific* `active_pdf_ids` fresh from Chroma and reconstructs a `BM25Retriever` from scratch each run. If a chunk from a document *not* in `active_pdf_ids` were included, it would silently pollute every IDF calculation for every term.

This is a structural asymmetry between the two retrieval paradigms that's easy to miss: vectors are embedded once, independently, and reused; BM25 statistics are a *property of the query-time corpus* and must be recomputed whenever that corpus changes.

### 3. Over-fetching (`RETRIEVAL_K = 15`) — the funnel theory

```python
RETRIEVAL_K = 15   # wider net; reranker will trim this down later
```

This encodes a specific theoretical tradeoff in multi-stage retrieval systems, often called the **retrieve-then-rerank funnel**. The intuition: cheap, approximate retrieval methods (BM25, vector cosine similarity) are *fast* but *imprecise* — they're good at roughly separating "plausibly relevant" from "definitely irrelevant," but bad at fine-grained ordering *within* the plausibly-relevant set. A precise method (cross-encoder, covered next) is *accurate* but *expensive* — too slow to run over your entire corpus, but perfectly suited to re-judging a small shortlist.

So the architecture deliberately **over-retrieves** at the cheap stage (K=15, wider than the final K=5 needed) specifically to maximize the odds that the *true* best chunks are somewhere in that shortlist — even if BM25 or vector search ranked them 12th instead of 1st individually — and then hands that wider net to a more expensive, more accurate method to do the fine sorting. This is a recall/precision handoff: stage 1 optimizes for recall (don't lose the right chunk), stage 2 optimizes for precision (put the right chunk first).

### 4. Reciprocal Rank Fusion (RRF) — the actual fusion theory inside `EnsembleRetriever`

```python
hybrid_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.4, 0.6])
```

This is the theoretically interesting part. You might expect fusion to average the *scores* from BM25 and vector search — but that's a trap, because **BM25 scores and cosine-similarity scores live on completely different, incomparable scales.** BM25 scores are unbounded (they can be 0.1 or 15.7 depending on IDF magnitudes and corpus size); cosine similarity is bounded in $[-1, 1]$ (or $[0,1]$ for normalized embeddings). Averaging two numbers from different scales is meaningless — a BM25 score of 8 isn't "more relevant" than a cosine score of 0.9 in any comparable sense.

RRF sidesteps this entirely by **throwing away the raw scores and using only rank position.** For each retriever's result list, RRF computes:

$$RRF\_score(d) = \sum_{\text{retriever } r} \frac{w_r}{k + rank_r(d)}$$

where $rank_r(d)$ is the position of document $d$ in retriever $r$'s ranked list (1st, 2nd, 3rd...), and $k$ is a small constant (commonly 60) that dampens the effect of very high ranks so the function doesn't blow up as rank→1.

Why this works theoretically: rank position is a **scale-free** measure of relevance — "3rd most relevant according to BM25" and "3rd most relevant according to vector search" are directly comparable statements, even though the underlying score magnitudes that produced those ranks are not. A document that ranks highly in *both* lists accumulates a large sum from two large fractions (small rank → large $1/(k+rank)$). A document that only one retriever found gets a contribution from just one term. This is precisely the "independent evidence compounding" argument from section 1, implemented in a scale-agnostic way.

The `weights=[0.4, 0.6]` scales each retriever's contribution to that sum — meaning the code has decided vector search's rankings should count for somewhat more than BM25's when they disagree, without needing either retriever's raw scores to be on the same numeric scale. This weighting is a design decision, not something derivable from the math — it reflects a belief about which retrieval mode is more trustworthy for this corpus/query distribution (PDF documents with prose content — semantic matching is probably more useful more often than exact keyword matching for a question like "what happened to Germany after WW1," which doesn't hinge on rare exact tokens).

### 5. Cross-encoder reranking — why it's a different theoretical model, not just "vector search again"

```python
RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
...
pairs = [(state.query, c["content"]) for c in state.context]
scores = RERANKER.predict(pairs)
```

This is the most important conceptual distinction in the whole pipeline, and it's worth being precise about it since you already know vector search: **a cross-encoder is architecturally different from the embedding model used for vector search, not just a fancier version of the same thing.**

Vector search (a **bi-encoder**) encodes the query and each document **separately**, into two independent fixed vectors, and then compares them with a cheap operation (cosine similarity) afterward. Critically, when the document's embedding was computed, the model had *no idea what the query would be* — the document vector is query-agnostic, computed once and reused for every future query. This is *why* vector search scales — you can precompute and store embeddings for a million chunks once, then just do cheap similarity math at query time.

A cross-encoder does the opposite: it takes the **query and document concatenated together as a single input** (`[query] [SEP] [document]`) and passes that *pair* jointly through the full transformer in one forward pass. Every layer of attention can now directly compare specific words in the query against specific words in the document — the model can notice, for instance, that the query says "Germany after WW1" and this specific chunk's third sentence says "the Weimar Republic" is a period-appropriate concept, and weigh that interaction directly. A bi-encoder's two separate vectors can never capture that kind of fine-grained, query-specific interaction — it's forced to compress "everything this document could ever be relevant to" into one static vector ahead of time.

The cost of this power is that a cross-encoder can't be precomputed — it has to be run at query time, once per (query, candidate) pair, and it's computationally heavier per pair than a cosine similarity lookup. This is *exactly* why it appears at the reranking stage rather than the initial retrieval stage: you can't afford to cross-encode every chunk in a large corpus against every query, but you can absolutely afford to cross-encode the 15 shortlisted candidates that survived the cheap BM25+vector funnel. This is the retrieve-then-rerank funnel from section 3, made concrete: cheap-and-approximate casts a wide net, expensive-and-precise does the final, accurate sort over a small candidate set.

Notice also *why* reranking specifically after fusion (not before, not instead of) matters theoretically: RRF fuses two *rank-based* signals that each have blind spots; a chunk could rank well in the fused list because it satisfied BM25's keyword match while being only tangentially relevant, or vice versa for vector similarity. The cross-encoder is the first point in the pipeline that actually reads the query and the candidate content *jointly* and re-judges "is this chunk really relevant," independent of whichever retrieval mechanism surfaced it. It's a check against both retrievers' blind spots simultaneously, not a redundant third vote.

---

### The full theoretical shape, end to end

Two cheap, independent, complementary-but-imprecise retrieval signals (BM25's lexical statistics, vector search's semantic geometry) get fused by a scale-free rank-based method (RRF) that lets their agreements compound and their individual blind spots get outvoted rather than trusted blindly — deliberately over-fetching to protect recall. That fused, still-imprecise shortlist then passes through an expensive, jointly-attending model (the cross-encoder) that can only afford to run over a small candidate set, but in exchange produces a genuinely query-aware relevance judgment neither retrieval stage could produce alone. The whole thing sits inside a state-machine graph specifically so that future stages — self-correction, iterative retrieval, agentic loops — have somewhere to plug in without a rewrite.