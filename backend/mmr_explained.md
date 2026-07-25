## Standard Similarity Search vs. MMR — Theoretical Breakdown

### 1. What plain similarity search (`search_type="similarity"`) does

This is the simplest retrieval strategy. Given a query embedding **q**, it computes a similarity score (cosine similarity, or sometimes L2 distance) between **q** and every candidate document embedding **dᵢ** in the vector store (after applying your metadata filter, e.g. `pdf_id`), then returns the top-k documents by that score alone.

Cosine similarity formula:

```
sim(q, d) = (q · d) / (||q|| ||d||)
```

The retriever ranks all filtered candidates by `sim(q, dᵢ)` descending and returns the top 5. That's it — a single ranking criterion, relevance to the query. There is no mechanism that looks at how similar the returned documents are *to each other*.

**The core weakness:** if your document set has several near-duplicate chunks (common in RAG when a PDF repeats a definition, or when overlapping chunk windows create redundant text), the top-5 results can all cluster around the same semantic point. You get 5 chunks that are all highly relevant but low-information as a set — a lot of that context is spent restating the same thing.

---

### 2. What MMR (`search_type="mmr"`) does

MMR (Maximal Marginal Relevance) was designed exactly to fix this. Instead of a single relevance score, it optimizes a combined objective at *each selection step*:

```
MMR = argmax_{d ∈ (C \ S)} [ λ · sim(q, d) − (1 − λ) · max_{dⱼ ∈ S} sim(d, dⱼ) ]
```

Where:
- **C** = the candidate pool (your `fetch_k=20` documents, pre-fetched by plain similarity)
- **S** = the set of documents already selected so far
- **λ (lambda_mult)** = trade-off knob between relevance and diversity
- The first term rewards similarity to the query
- The second term **penalizes** similarity to documents *already chosen* — it takes the max similarity to any already-picked doc, so a candidate that's redundant with even one selected doc gets punished

This is inherently **iterative and greedy** — after each pick, the "diversity penalty" term has to be recomputed against the updated set S, because now there's one more document to be different from. That's fundamentally different from plain similarity search, which just sorts once and slices the top k.

Your config: `fetch_k=20` (candidate pool), `k=5` (final output), `lambda_mult=0.9` (90% relevance weight, 10% diversity weight — mild diversity pressure, mostly relevance-driven).

---

### 3. Worked numeric example

Say your query is *"What is gradient descent?"* and after embedding + filtering by `pdf_id`, your `fetch_k=20` search returns candidates. Let's simplify to 5 candidates competing for `k=3` slots, with cosine similarities to the query already computed:

| Doc | sim(q, d) |
|---|---|
| A | 0.92 |
| B | 0.91 (near-duplicate of A — repeated definition) |
| C | 0.85 |
| D | 0.80 |
| E | 0.75 |

**Plain similarity search** just takes top-3 by `sim(q,d)`: **A, B, C**. Notice A and B are basically the same content — you've burned 2 of your 3 context slots on redundancy.

**MMR with λ = 0.9**, computed step by step:

*Step 1:* S is empty, so the penalty term is 0 for everyone. Pick the highest `sim(q,d)`: **A** (0.92). S = {A}.

*Step 2:* Now compute the MMR score for each remaining candidate, using sim(d, A) as the redundancy term (assume these pairwise similarities: sim(B,A)=0.95 since B is a near-duplicate, sim(C,A)=0.40, sim(D,A)=0.35, sim(E,A)=0.30):

```
MMR(B) = 0.9(0.91) − 0.1(0.95) = 0.819 − 0.095 = 0.724
MMR(C) = 0.9(0.85) − 0.1(0.40) = 0.765 − 0.040 = 0.725
MMR(D) = 0.9(0.80) − 0.1(0.35) = 0.720 − 0.035 = 0.685
MMR(E) = 0.9(0.75) − 0.1(0.30) = 0.675 − 0.030 = 0.645
```

**C edges out B** (0.725 vs 0.724) — even though B had higher raw relevance (0.91 vs 0.85), B's near-duplication with A knocks it down just enough. Pick **C**. S = {A, C}.

*Step 3:* Now redundancy is measured as the max similarity to *either* A or C. Say sim(B,C)=0.42, sim(D,A)=0.35, sim(D,C)=0.30, sim(E,A)=0.30, sim(E,C)=0.55:

```
MMR(B) = 0.9(0.91) − 0.1·max(0.95, 0.42) = 0.819 − 0.095 = 0.724
MMR(D) = 0.9(0.80) − 0.1·max(0.35, 0.30) = 0.720 − 0.035 = 0.685
MMR(E) = 0.9(0.75) − 0.1·max(0.30, 0.55) = 0.675 − 0.055 = 0.620
```

Pick **B** (0.724 is still highest here despite the redundancy penalty, since λ=0.9 favors relevance heavily). Final set: **A, C, B**.

Compare the two outputs:
- **Similarity:** A, B, C → two of three chunks nearly identical
- **MMR:** A, C, B → same three chunks selected in this small example, but notice *why* — MMR still picked B eventually because λ=0.9 is relevance-dominant, but it *reordered the selection logic* and would have dropped B entirely in favor of D or E if B's redundancy with A had been more extreme, or if λ were lower (e.g. λ=0.5).

To see MMR actually *change the result set* (not just the order), lower λ to 0.5 and redo step 2:

```
MMR(B) = 0.5(0.91) − 0.5(0.95) = 0.455 − 0.475 = −0.020
MMR(C) = 0.5(0.85) − 0.5(0.40) = 0.425 − 0.200 =  0.225
```

Now C beats B by a wide margin, and B likely never makes it into the top-3 at all — this is where MMR earns its keep: it actively excludes near-duplicates rather than just nudging their rank.

---
