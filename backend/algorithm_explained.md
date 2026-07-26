# Hybrid Retrieval Theory: TF-IDF → BM25 → RRF → Cross-Encoder Reranking

A from-first-principles walkthrough of why `pdf_bot`'s retrieval pipeline (BM25 + vector search → Reciprocal Rank Fusion → cross-encoder rerank) is built the way it is, with a fully worked numerical example.

---

## 1. TF-IDF: the starting point

### The problem

We need one number per document that says "how relevant is this document to this query?" Two requirements fall out immediately:

1. **Word overlap matters, but not equally.** A word that appears in every document (like "the") tells you nothing. A word that appears in only a few documents is a strong signal when it matches.
2. **Repetition within a document signals emphasis**, but should be normalized so long documents don't win just by having more words.

### Term Frequency (TF)

```
TF(t, d) = (number of times term t appears in document d) / (total number of terms in d)
```

Normalizing by document length turns raw counts into a measure of *concentration* — how much of this document is "about" the term — independent of document length.

### Inverse Document Frequency (IDF)

```
IDF(t, D) = log( N / df(t) )
```

Where `N` = total documents, `df(t)` = number of documents containing term `t`.

- Rare terms (small `df`) → large ratio → high IDF (informative).
- Common terms (`df` close to `N`) → ratio close to 1 → IDF close to 0.
- The **log** is what converts multiplicative rarity ("10x rarer") into additive score differences, and prevents one ultra-rare term from exploding the score.
- **The flaw:** if a term appears in *every* document (`df = N`), IDF hits **exactly zero**. TF-IDF has no way to express "slightly useful" — only "useful" down to "worthless," with a hard cliff at `df = N`.

### Combining them

```
TF-IDF(t, d) = TF(t, d) × IDF(t, D)
```

Multiplication (not addition) encodes an AND relationship: both "this document emphasizes the term" and "this term is globally meaningful" must hold for the term to contribute. If IDF is 0, no amount of TF rescues it.

---

## 2. BM25: fixing TF-IDF's two cracks

**Crack 1 — IDF can hit a hard zero.** A term in every document should be "barely useful," not "provably useless."

**Crack 2 — raw TF grows forever, linearly.** The 100th occurrence of a word shouldn't count for 10x what the 10th occurrence did. There's no saturation.

### BM25's smoothed IDF

```
IDF_BM25(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
```

- The `+0.5` terms are smoothing constants (same idea as Laplace smoothing) that soften the extremes.
- The outer `+1` is the critical fix: even when `df(t) = N`, this expression stays above zero. The floor is raised off the ground — universally common terms now contribute a small, non-zero amount instead of literally nothing.

### BM25's saturating term frequency

```
TF_BM25(t, d) = ( f(t,d) × (k1 + 1) ) / ( f(t,d) + k1 × norm(d) )
```

where:

```
norm(d) = 1 - b + b × ( |d| / avgdl )
```

- `f(t,d)` = raw count of term `t` in document `d`
- `k1` (typically ~1.2–2.0) controls how fast repetition saturates — this is a rectangular-hyperbola shape (same math as Michaelis–Menten enzyme kinetics): steep at first, then flattens toward a ceiling of `k1 + 1` and never grows further no matter how many times the term repeats.
- `b` (0 to 1) is a dial for length normalization: `b=0` ignores document length entirely; `b=1` fully normalizes against the corpus's average length (`avgdl`); values in between blend the two.

### Full BM25 score

```
BM25(D, Q) = Σ over query terms t:  IDF_BM25(t) × TF_BM25(t, D)
```

Same skeleton as TF-IDF (sum of IDF × TF across query terms) — every constant exists to patch a specific flaw, not as an arbitrary tuning knob.

| | TF-IDF | BM25 |
|---|---|---|
| IDF at df=N | exactly 0 (cliff) | small positive (smoothed floor) |
| Effect of repeated term | grows linearly, unbounded | saturates toward a ceiling |
| Length correction | one-time division inside TF | tunable dial (`b`), relative to corpus average |

---

## 3. Why hybrid retrieval exists

BM25 and vector (embedding) search fail in **opposite, largely independent** ways:

- **BM25** misses paraphrase and synonymy — no lexical overlap means no match, even if the meaning matches.
- **Vector search** blurs exact/rare terms — specific identifiers, acronyms, and numbers get compressed into "semantically nearby" territory and can lose precision.

When both signals agree a document is relevant, that agreement should compound. When they disagree, a good fusion method shouldn't let one signal's blind spot silently veto the other's catch.

---

## 4. Reciprocal Rank Fusion (RRF)

Raw BM25 scores and cosine similarities live on **incompatible scales** — averaging them is meaningless. RRF sidesteps this by using only **rank position**, which is scale-free and directly comparable across retrievers:

```
RRF(d) = Σ over retrievers r:  w_r / (k + rank_r(d))
```

- `rank_r(d)` = the position of document `d` in retriever `r`'s ranked list (1 = best)
- `k` = a damping constant, commonly 60
- `w_r` = a weight for retriever `r` — in `pdf_bot`, `weights=[0.4, 0.6]` for `[bm25, vector]`

A document ranking near the top in *both* lists accumulates two large terms. A document only one retriever found gets just one contribution. The weights let you decide which signal to trust more when they disagree.

---

## 5. Cross-encoder reranking

**Vector search (bi-encoder):** query and document are embedded **separately**, into two independent vectors, then compared cheaply (cosine similarity) afterward. The document's vector is computed without knowing what the query will be — this is what makes it cheap to precompute at scale.

**Cross-encoder:** query and document are concatenated and passed **jointly** through the transformer in a single forward pass. Every attention layer can directly compare specific query words against specific document words. This produces a far more precise relevance judgment — but it can't be precomputed, so it's too expensive to run over an entire corpus.

**Why it sits after fusion, not instead of retrieval:** cheap-and-approximate (BM25 + vector) casts a wide net over the whole corpus to protect recall; expensive-and-precise (cross-encoder) does the final, accurate sort over only the small shortlist that survived. This is the retrieve-then-rerank funnel — exactly why `pdf_bot` over-fetches (`RETRIEVAL_K = 15`) before reranking down to `TOP_N = 5`.

---

## 6. Full worked example

### Corpus (5 chunks, tokenized — lowercased, punctuation stripped, all tokens kept for length)

```
C1: "BM25 uses term frequency saturation and length normalization to score documents."
C2: "Vector embeddings capture semantic meaning using dense numerical representations."
C3: "Cross encoders jointly attend to query and document tokens to improve ranking."
C4: "Reciprocal rank fusion combines rankings from multiple retrievers using reciprocal scores."
C5: "PDF loaders split large files into smaller overlapping text chunks for indexing."
```

| Doc | Length \|d\| |
|---|---|
| C1 | 11 |
| C2 | 9 |
| C3 | 12 |
| C4 | 11 |
| C5 | 12 |

```
avgdl = (11 + 9 + 12 + 11 + 12) / 5 = 55 / 5 = 11
```

**Query:** "How does BM25 score keyword ranking?" → content terms: `bm25`, `score`, `keyword`, `ranking` (exact match only — no stemming, so "score" ≠ "scores", "ranking" ≠ "rankings"/"rank").

### Step 1 — BM25

**Document frequencies and IDF**

| Term | df | IDF_BM25 |
|---|---|---|
| bm25 | 1 (C1 only) | ln(4.5/1.5 + 1) = ln(4) = **1.3863** |
| score | 1 (C1 only) | **1.3863** |
| keyword | 0 (nowhere) | ln(5.5/0.5 + 1) = ln(12) = **2.4849** |
| ranking | 1 (C3 only) | **1.3863** |

`keyword` has the highest IDF of all (rarest possible), but it appears in zero documents — so `TF = 0` everywhere, and it contributes nothing to any score. **High IDF measures potential informativeness, not actual relevance in this corpus.**

**Length normalization (b = 0.75), norm(d) = 0.25 + 0.75 × (|d| / avgdl)**

| Doc | \|d\|/avgdl | norm(d) |
|---|---|---|
| C1 | 1.0 | 1.0000 |
| C3 | 1.0909 | 1.0682 |
| C4 | 1.0 | 1.0000 |

**Per-term contribution — IDF(t) × [ f×(k1+1) / (f + k1×norm) ], k1 = 1.5**

*C1* (bm25 f=1, score f=1; norm=1.0, denom = 1 + 1.5×1.0 = 2.5):
- bm25: 1.3863 × (1×2.5 / 2.5) = **1.3863**
- score: 1.3863 × (2.5/2.5) = **1.3863**
- **C1 total = 2.7726**

*C3* (ranking f=1; norm=1.0682, denom = 1 + 1.5×1.0682 = 2.6023):
- ranking: 1.3863 × (2.5/2.6023) = 1.3863 × 0.9607 = **1.3323**
- **C3 total = 1.3323**

*C4* — contains none of `bm25`/`score`/`ranking` as **exact** tokens ("rankings", "rank", "scores" all fail exact match): **total = 0**

*C2, C5* — no matches at all: **total = 0**

**BM25 final ranking**

| Rank | Doc | Score |
|---|---|---|
| 1 | C1 | 2.7726 |
| 2 | C3 | 1.3323 |
| 3 (tied) | C2, C4, C5 | 0.0000 |

C4 — which is *actually* about "reciprocal rank fusion... combines rankings... using reciprocal scores," nearly a paraphrase of the query — scores **zero**, indistinguishable from two totally unrelated chunks. This is BM25's blind spot to morphological variants (rank/ranking/rankings, score/scores) made concrete.

### Step 2 — Vector search (illustrative cosine similarities)

| Doc | Cosine similarity | Why |
|---|---|---|
| C1 | 0.81 | Directly explains BM25 scoring mechanics |
| C4 | 0.74 | Semantically about ranking/scores via fusion — close in meaning despite zero shared words |
| C3 | 0.68 | About improving ranking, but via reranking specifically |
| C2 | 0.40 | Different topic (embeddings, not ranking) |
| C5 | 0.25 | Unrelated (PDF chunking) |

**Vector ranking: C1 > C4 > C3 > C2 > C5**

Vector search rescues C4 to 2nd place — exactly where BM25 buried it at zero.

### Step 3 — RRF fusion (k=60, weights [0.4 bm25, 0.6 vector] — matching `pdf_bot`'s `EnsembleRetriever`)

```
RRF(d) = 0.4/(60+rank_bm25) + 0.6/(60+rank_vector)
```

| Doc | BM25 rank | Vector rank | Calculation | RRF score |
|---|---|---|---|---|
| C1 | 1 | 1 | 0.4/61 + 0.6/61 | **0.016393** |
| C3 | 2 | 3 | 0.4/62 + 0.6/63 | **0.015976** |
| C4 | 4 | 2 | 0.4/64 + 0.6/62 | **0.015927** |
| C2 | 3 | 4 | 0.4/63 + 0.6/64 | **0.015724** |
| C5 | 5 | 5 | 1.0/65 | **0.015385** |

**Fused ranking: C1 > C3 > C4 > C2 > C5**

C4 climbs from a zero-score, bottom-tier BM25 result to a near-tie for 2nd place (0.015927 vs C3's 0.015976 — a gap of only 0.000049), purely on the strength of vector search's rank-2 vote at the larger 0.6 weight.

### Step 4 — Cross-encoder reranking

| Doc | Cross-encoder score | Why |
|---|---|---|
| C1 | 6.3 | Direct, literal match to "BM25 score" mechanics |
| C4 | 3.8 | Joint attention correctly reads "rank fusion / rankings / reciprocal scores" as substantively about ranking, despite zero shared tokens with the query |
| C3 | 2.5 | Related, but through reranking specifically — one inferential hop further |
| C2 | -1.2 | Off-topic |
| C5 | -3.7 | Unrelated |

**Final reranked order: C1 > C4 > C3 > C2 > C5**

The cross-encoder flips C3 and C4 with a clear, confident gap (3.8 vs 2.5) — a judgment neither BM25 (zero lexical overlap) nor RRF (rank position only) could make.

### Step 5 — Generation

Top-2 after rerank (**C1, C4**) go into the LLM prompt. The generated answer correctly explains BM25's own mechanics *and* how it fits into rank fusion — while C3's reranking-specific content stays just outside the context window, and the two unrelated chunks never appear.

---

## 7. The one-line summary

| Stage | Question it answers | Weakness it covers for |
|---|---|---|
| BM25 | Do the exact words match? | Vector search can miss rare terms/IDs/acronyms |
| Vector search | Does the meaning match? | BM25 can miss paraphrases, synonyms, and morphological variants |
| RRF fusion | What do both retrievers agree is likely relevant? | Neither retriever alone is reliable enough; scores aren't comparable, so fuse by rank |
| Cross-encoder rerank | Given query AND chunk together, how relevant is this really? | Retrieval scores are approximate; this is the precise, expensive judge that only runs on the small shortlist |

**The result seen in this example, for one document (C4): BM25 rank ~4-5 (score 0) → RRF rank ~3 → cross-encoder rank 2.** Three independent mechanisms, three different blind spots, one document correctly recovered.