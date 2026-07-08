### Combining vector search similarity with cross encoder reranking to do Hybrid Search to get both speed and accuracy
Let's walk through the algorithm step by step with concrete numbers: **query = "what is gravity"**, initial retrieval **k = 5**, final **n = 3**.

## Step 1: Bi-encoder retrieval (the "candidate" stage)

Your vector store (ChromaDB) already has embeddings for every chunk, computed **independently** of any query — that's what makes it fast. At query time:

1. Embed the query: `vec_q = embed("what is gravity")`
2. Compare `vec_q` against all stored chunk vectors using cosine similarity / MMR
3. Return the top **k=5** closest chunks

Say you get back these 5 chunks (this is `candidate_docs` in the graph):

| # | Chunk text (snippet) | Bi-encoder similarity score |
|---|---|---|
| D1 | "Gravity is a force that attracts two bodies with mass toward each other." | 0.81 |
| D2 | "Newton's law of universal gravitation states F = Gm₁m₂/r²." | 0.79 |
| D3 | "The gravitational field on Earth's surface is about 9.8 m/s²." | 0.76 |
| D4 | "Einstein's general relativity describes gravity as curvature of spacetime." | 0.74 |
| D5 | "Apples fall from trees due to gravitational pull." | 0.70 |

Notice: these scores come from comparing **query embedding vs document embedding separately** — the model never actually looks at the query and document *together*. It's fast because embeddings are precomputed, but the similarity is a rough proxy for relevance.

## Step 2: Cross-encoder reranking (the "precision" stage)

This is architecturally different. Instead of two separate embeddings compared by cosine distance, the cross-encoder takes the **pair** `(query, document)` as a single joint input:

```
input = "[CLS] what is gravity [SEP] Gravity is a force that attracts two bodies... [SEP]"
```

and passes it through a transformer that outputs **one relevance score** for that exact pair. It does this **once per candidate**, so for k=5 candidates, that's 5 forward passes:

```
pairs = [
  ["what is gravity", D1_text],
  ["what is gravity", D2_text],
  ["what is gravity", D3_text],
  ["what is gravity", D4_text],
  ["what is gravity", D5_text],
]
scores = cross_encoder.predict(pairs)
```

Because the model sees query and doc **jointly**, it can pick up on things cosine similarity misses — word overlap patterns, whether the doc actually *answers* "what is X" versus just mentioning the topic, etc. Say the output scores are:

| # | Chunk | Bi-encoder score | Cross-encoder score |
|---|---|---|---|
| D1 | Gravity = force attracting masses | 0.81 | **9.2** |
| D2 | Newton's law F = Gm₁m₂/r² | 0.79 | 6.8 |
| D4 | Einstein's spacetime curvature | 0.74 | **8.9** |
| D3 | Earth's gravitational field 9.8 m/s² | 0.76 | 5.1 |
| D5 | Apples falling | 0.70 | 4.3 |

Notice the **reordering**: D4 (Einstein/spacetime) jumped from rank 4 → rank 2, because the cross-encoder judged it as a genuinely strong direct answer to "what is gravity," even though its raw embedding similarity was lower than D2 and D3. This is the whole point of reranking — it corrects mistakes the bi-encoder makes.

## Step 3: Sort and truncate to n=3

```python
sorted_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
top_n = sorted_docs[:3]
```

Result — the 5 candidates get cut down to the **final 3** sent to the LLM:

1. D1 — "Gravity is a force that attracts two bodies..." (9.2)
2. D4 — "Einstein's general relativity describes gravity..." (8.9)
3. D2 — "Newton's law of universal gravitation..." (6.8)

D3 and D5 get dropped even though they *were* in the original top-5 by embedding similarity — the cross-encoder decided they were less directly relevant.

## Why not just retrieve 3 directly and skip reranking?

Because bi-encoder similarity is a **cheap approximation** — it's optimized for speed across millions of vectors, not accuracy on a handful of candidates. Retrieving a wider pool (k=5, or in production usually k=15–20) and letting the cross-encoder — which is slower but far more accurate — make the final judgment call on just those few candidates gives you the best of both: fast search over the whole corpus, accurate ranking of the shortlist.

That's exactly what `retrieve_node → rerank_node` does in the graph from before: `k` controls the width of the net, `n` (called `top_n` in the code) controls how many survive to reach the LLM.
