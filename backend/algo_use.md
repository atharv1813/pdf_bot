## TF-IDF and its role in RAG retrieval

In a RAG pipeline, before your LLM ever sees a chunk of text, you need to *find* the right chunks out of potentially thousands sitting in your vector store or a keyword index. TF-IDF (and its descendant BM25) is the classic lexical/sparse retrieval method — it scores every document in the corpus against the query based on **word overlap**, weighted so that rare, discriminative words matter more than common ones. This is exactly the "BM25 side" of your ensemble retrieval (BM25 + vector search → RRF → cross-encoder rerank). Vector search catches *semantic* similarity ("car" ≈ "automobile"), while BM25/TF-IDF catches *exact term* importance — that's why combining them via RRF gives you better recall than either alone.

The two components from your image:

- **TF(t,d)** = (times term *t* appears in doc *d*) / (total terms in *d*) — rewards documents that use the term a lot.
- **IDF(t,D)** = log( total docs / docs containing *t* ) — punishes terms that appear everywhere (they're not discriminative), rewards rare terms.

**TF-IDF(t,d) = TF(t,d) × IDF(t,D)**

You sum this over all query terms to get a document's final score.

---

## Worked example — Corpus of 3 documents

Let's use a mini "RAG-flavored" corpus (small vocabulary on purpose, so every number is checkable by hand):

- **D1**: "vector database stores embeddings for retrieval" → 6 terms
- **D2**: "BM25 ranking function improves retrieval scoring" → 6 terms
- **D3**: "RAG uses vector retrieval for context generation" → 7 terms

**Query**: "vector retrieval"

### Step 1 — Document frequency (df) of query terms

| Term | Appears in | df | N (total docs) |
|---|---|---|---|
| vector | D1, D3 | 2 | 3 |
| retrieval | D1, D2, D3 | 3 | 3 |

### Step 2 — IDF (using natural log, as in your formula)

$$IDF(\text{vector}) = \ln(3/2) = \ln(1.5) \approx 0.405$$
$$IDF(\text{retrieval}) = \ln(3/3) = \ln(1) = 0$$

Notice this already: **"retrieval" appears in every document, so its IDF collapses to exactly zero.** It contributes *nothing* to the score no matter how many times it appears anywhere. Hold onto this — it's the exact weakness BM25 fixes below.

### Step 3 — TF for each term in each document

| Doc | TF(vector) | TF(retrieval) |
|---|---|---|
| D1 (6 terms) | 1/6 = 0.1667 | 1/6 = 0.1667 |
| D2 (6 terms) | 0/6 = 0 | 1/6 = 0.1667 |
| D3 (7 terms) | 1/7 = 0.1429 | 1/7 = 0.1429 |

### Step 4 — TF-IDF per term, then sum per document

**D1**: (0.1667 × 0.405) + (0.1667 × 0) = **0.0675** + 0 = **0.0675**

**D2**: (0 × 0.405) + (0.1667 × 0) = 0 + 0 = **0.0000**

**D3**: (0.1429 × 0.405) + (0.1429 × 0) = **0.0579** + 0 = **0.0579**

### Final TF-IDF ranking

| Rank | Doc | Score |
|---|---|---|
| 🥇 1 | D1 | 0.0675 |
| 2 | D3 | 0.0579 |
| 3 | D2 | 0.0000 |

**D1 wins.** Makes intuitive sense — it's short and dense with "vector." But look at D2: it's the *runner-up in relevance by common sense* (it's about BM25/ranking/retrieval — arguably more topically related to a "retrieval" query than you'd expect) yet TF-IDF gives it a **flat zero**, tying it with a totally irrelevant document would score. That's the flaw.

---

## Why BM25 improves on this — in depth, same corpus and query

BM25 fixes TF-IDF in **three concrete ways**, all visible in this example:

1. **IDF smoothing** — never lets IDF hit exactly 0 (or go negative for very common terms), so a term appearing in every doc still contributes *something*.
2. **Term-frequency saturation** — TF-IDF's raw TF grows linearly forever (10 occurrences = 10× the score of 1 occurrence). BM25 caps the benefit of repeating a word via the `k1` parameter — the 5th occurrence of a word barely adds more than the 4th.
3. **Length normalization** — via parameter `b`, BM25 explicitly penalizes/rewards documents based on how their length compares to the *average* document length in the corpus, rather than just dividing by length once (which TF already does, but BM25 tunes how strongly this matters).

### The BM25 formula

$$BM25(D,Q) = \sum_{i} IDF(q_i) \cdot \frac{f(q_i, D) \cdot (k_1+1)}{f(q_i,D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{avgdl}\right)}$$

Standard values: **k1 = 1.5**, **b = 0.75** (these are the defaults in Elasticsearch, Lucene, and most BM25 implementations).

**BM25's IDF formula** (this is the fix for problem #1 above):

$$IDF_{BM25}(t) = \ln\left(\frac{N - df(t) + 0.5}{df(t) + 0.5} + 1\right)$$

The `+1` inside the log is exactly what prevents IDF from ever reaching zero, even when a term is in *every* document in the corpus.

### Step 1 — avgdl (average document length)

$$avgdl = \frac{6+6+7}{3} = \frac{19}{3} \approx 6.333$$

### Step 2 — BM25 IDF for each query term

**vector** (df=2, N=3):
$$IDF = \ln\left(\frac{3-2+0.5}{2+0.5}+1\right) = \ln\left(\frac{1.5}{2.5}+1\right) = \ln(1.6) \approx 0.470$$

**retrieval** (df=3, N=3):
$$IDF = \ln\left(\frac{3-3+0.5}{3+0.5}+1\right) = \ln\left(\frac{0.5}{3.5}+1\right) = \ln(1.1429) \approx 0.1335$$

**Compare to TF-IDF:** "retrieval" went from IDF = **0.000** → IDF = **0.1335**. It's now small (correctly reflecting that it's a common, less-discriminative term), but not zero. This is the whole ballgame.

### Step 3 — Score each document

**D1** (|D|=6): 
Length normalization factor = $1 - 0.75 + 0.75 \times \frac{6}{6.333} = 0.25 + 0.7105 = 0.9605$
Denominator (same for both terms since both have f=1, |D|=6) = $1 + 1.5 \times 0.9605 = 2.4408$
Numerator (f=1) = $1 \times 2.5 = 2.5$

- vector: $0.470 \times \frac{2.5}{2.4408} = 0.470 \times 1.0243 = 0.4814$
- retrieval: $0.1335 \times 1.0243 = 0.1367$

**D1 total = 0.4814 + 0.1367 = 0.6181**

**D2** (|D|=6): same length-norm factor as D1 (same length) → denom = 2.4408

- vector: f=0, so numerator=0 → contributes **0**
- retrieval: f=1 → $0.1335 \times 1.0243 = 0.1367$

**D2 total = 0 + 0.1367 = 0.1367**

**D3** (|D|=7):
Length normalization factor = $0.25 + 0.75 \times \frac{7}{6.333} = 0.25 + 0.8290 = 1.0790$
Denominator = $1 + 1.5 \times 1.0790 = 2.6185$

- vector: $0.470 \times \frac{2.5}{2.6185} = 0.470 \times 0.9548 = 0.4487$
- retrieval: $0.1335 \times 0.9548 = 0.1275$

**D3 total = 0.4487 + 0.1275 = 0.5762**

### Final BM25 ranking

| Rank | Doc | BM25 Score | TF-IDF Score (for comparison) |
|---|---|---|---|
| 🥇 1 | D1 | 0.6181 | 0.0675 |
| 2 | D3 | 0.5762 | 0.0579 |
| 3 | D2 | **0.1367** | **0.0000** |

---

## What actually changed, and why it matters for your RAG pipeline

- **D1 still wins** — the ranking order didn't flip in this toy example, but that's not the point.
- **The real fix is D2**: TF-IDF said D2 has *zero* relevance to "vector retrieval" — indistinguishable from a document that shares no words with the query at all. BM25 correctly says "D2 shares one query term, so it deserves some non-zero relevance signal" (0.1367), while still ranking it well below D1 and D3 which share both terms. This matters a lot in Reciprocal Rank Fusion — if BM25 assigns D2 a real rank position (say rank 3 with a nonzero score) instead of a tied zero-score bucket with every other unrelated doc, RRF can merge it more sensibly with the vector-search ranking instead of losing that signal entirely.
- **Saturation**: if D1 instead said "vector vector vector vector vector database," raw TF-IDF would keep scaling its score up almost linearly. BM25's denominator (`f(q,D) + k1×...`) means going from 1→2 occurrences helps a lot, 4→5 occurrences helps very little — this stops keyword-stuffed chunks from dominating retrieval, which is a real risk in PDF-chunked corpora where boilerplate or headers repeat a term.
- **Length normalization (`b`)**: D3 is the longest doc (7 terms vs 6). Even though D3 has vector and retrieval at the same raw counts as D1, its slightly longer length pulls its score down a bit relative to D1 (0.5762 vs 0.6181) — BM25 is compensating for the fact that longer documents naturally have more chances to contain any given word by chance.
