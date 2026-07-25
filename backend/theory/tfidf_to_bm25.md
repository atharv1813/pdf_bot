## Building TF-IDF from first principles

### The problem we're trying to solve

Imagine you have a corpus of documents and a query. You need a single number per document that says "how relevant is this document to this query?" What would a good scoring function *need* to capture? Let's reason it out from scratch, one requirement at a time.

### Requirement 1: word overlap matters, but not equally

The naive idea: count how many query words appear in the document. But this immediately breaks. Consider the query "the vector retrieval system." The word "the" appears in nearly every English document ever written. It tells you *nothing* about relevance. Meanwhile "vector" appears in maybe 2% of documents — if a document contains "vector," that's a strong signal.

So intuitively: **a word's usefulness for distinguishing relevant from irrelevant documents is inversely related to how common that word is across the corpus.** A word that appears everywhere carries near-zero information. A word that appears rarely carries a lot of information. This is a direct echo of an idea from information theory. Shannon entropy quantifies the average information content of a random variable as $H(X) = -\sum_i p(x_i)\log p(x_i)$, where each term's contribution is built from $-\log p(x_i)$ — the "surprise" or self-information of a single outcome, higher when that outcome is rarer. IDF is essentially reinventing this per-outcome piece, $-\log(P(\text{term appears in a doc}))$, applied to a single term rather than summed over a whole distribution.

This is where **IDF (Inverse Document Frequency)** comes from — a measure of how rare, and therefore how informative, a term is across the whole corpus:

$$IDF(t) = \log\left(\frac{N}{df(t)}\right)$$

Walk through *why* each piece is shaped this way:

- $df(t)/N$ is the *fraction* of documents containing the term — essentially $P(t)$, the probability a random document contains it.
- We invert it ($N/df(t)$) because we want rare terms (small $df$) to produce a *large* number, and common terms (large $df$, up to $df=N$) to produce a number approaching 1.
- We take the $\log$ for two reasons. First, **diminishing returns**: going from a term appearing in 1000 documents to 100 documents (10x rarer) should matter about as much as going from 100 to 10 (another 10x). Rarity matters on a *multiplicative*, not additive, scale — logs convert multiplicative relationships into additive ones, matching human intuition about "how much rarer." Second, **it caps the range down to something reasonable** — without the log, a term appearing in 1 document out of a million would produce a score of a million, letting one ultra-rare term completely dominate the sum regardless of anything else. The log tames that explosion.
- When $df(t) = N$ (term is in every document), $IDF = \log(1) = 0$. This is a deliberate design choice: a term with zero discriminative power should contribute *zero* to the score. We'll come back to this — it's exactly the flaw BM25 fixes.

### Requirement 2: repetition within a document signals emphasis

If a document mentions "retrieval" once versus twenty times, the twenty-mention document is probably more centrally *about* retrieval. So we want to reward higher raw counts. That gives us **raw term frequency**: $f(t,d)$ = count of term $t$ in document $d$.

But raw counts alone are unfair to short documents. A one-sentence document that mentions "vector" once is arguably *more* about vectors than a 10,000-word document that happens to mention "vector" once amid everything else. So we normalize by document length, giving **TF (Term Frequency)** — a measure of how much a given document emphasizes a term, relative to its own length:

$$TF(t,d) = \frac{f(t,d)}{|d|}$$

This turns the raw count into something like "the probability that a randomly picked word from this document is the query term" — a length-independent measure of *concentration/emphasis*, not just occurrence.

### Requirement 3: combine them multiplicatively, not additively

Why multiply TF and IDF instead of adding them? Think about what each factor *means*:

- TF answers: "how much does this document emphasize term $t$?"
- IDF answers: "how much does term $t$ matter at all, corpus-wide?"

If IDF is 0 (term is meaningless — appears everywhere), it doesn't matter *how much* a document emphasizes it — the relevance contribution should be zero. Addition wouldn't achieve this (a high TF + zero IDF would still leave a nonzero score). Multiplication correctly encodes "both factors must be non-trivial for the term to matter" — it's an AND relationship, not an OR relationship. This is the same logic as multiplying independent probabilities.

$$TF\text{-}IDF(t,d) = TF(t,d) \times IDF(t)$$

Then for a multi-word query, you sum this across query terms — you're accumulating independent evidence from each word.

### The crack in TF-IDF's foundation

Follow the logic to its natural conclusion and a problem emerges. IDF was built to make common words contribute *less*. But at the extreme — a term in literally every document — IDF doesn't just get *small*, it hits exactly **zero**. That's a much stronger claim than "this word isn't very useful." Zero says "this word carries no information whatsoever, treat it identically to a word that doesn't appear at all." That's too aggressive — a word appearing in every document might still be *somewhat* more concentrated in relevant ones. TF-IDF's IDF function can't express "slightly useful"; it only has "very useful" down to "exactly useless," with the cliff sitting right at $df=N$.

Similarly, raw TF grows *linearly forever*. A document mentioning "retrieval" 100 times scores 10x higher than one mentioning it 10 times, in TF-IDF's math. But does the 91st–100th mention really tell you 10x more about relevance than the first 10 mentions did? Intuitively, no — once a document has clearly established it's about a topic, additional repetitions (often just because the doc is long, or repetitive, or keyword-stuffed) shouldn't keep scaling the score up proportionally. There should be **diminishing marginal returns on repetition**, the same way there are diminishing returns on rarity in IDF. TF-IDF captures diminishing returns for *rarity* but forgot to apply the same logic to *frequency*.

This is the logical seed BM25 grows from: **take TF-IDF's structure, and fix both of these diminishing-returns gaps properly.**

---

## Building BM25 from TF-IDF's cracks

BM25 keeps the same skeleton — sum over query terms of (an IDF-like factor) × (a TF-like factor) — but redesigns both factors to behave the way our intuition actually demanded.

### Fix 1: IDF that never truly zeroes out

$$IDF_{BM25}(t) = \ln\left(\frac{N - df(t) + 0.5}{df(t) + 0.5} + 1\right)$$

Reason through each modification:

- The $+0.5$ terms (a *smoothing constant*) prevent division-by-zero edge cases and soften the curve near the extremes — this is a standard statistical smoothing trick (related to Laplace/add-one smoothing), acknowledging that with finite data, "this term appeared in 0% of documents I've seen" shouldn't be treated as mathematical certainty that it's infinitely rare.
- The $N - df(t)$ in the numerator, instead of just $N$, measures "the number of documents *without* this term" rather than "total documents." This reframes IDF as literally a log-odds: $\ln(\frac{\text{docs without } t}{\text{docs with } t})$ — the odds that a random document does *not* contain the term. This is a more principled quantity than the plain ratio TF-IDF used, and it's the natural next step once you're already in "rare = informative" territory: log-odds is the standard way statisticians quantify how strongly evidence favors one outcome over another (it's literally the same math as logistic regression).
- The **outer $+1$** is the critical fix. Even when $df(t) = N$ (term in every document), $\frac{N-df+0.5}{df+0.5} \to$ a small positive fraction, and $\ln(\text{small fraction} + 1)$ is a small *positive* number — not zero. The floor has been raised off the ground. A universally common word now contributes a small but nonzero amount, matching the intuition that "slightly useful" and "completely useless" shouldn't be mathematically identical.

### Fix 2: TF that saturates instead of growing forever

We want a function of raw count $f(t,d)$ that:
- increases with $f$ (more mentions → more relevant), matching TF-IDF's original instinct,
- but grows quickly at first and then flattens — the difference between 0 and 1 occurrence should matter a lot, the difference between 20 and 21 occurrences should matter almost nothing.

This is a classic saturating-curve shape, and BM25 achieves it with:

$$\frac{f(t,d) \cdot (k_1+1)}{f(t,d) + k_1}$$

Trace what happens as $f$ grows: when $f$ is small relative to $k_1$, the expression behaves close to linear (each new occurrence adds meaningful value). As $f \to \infty$, the expression approaches $(k_1+1)$ — a hard ceiling — no matter how many times the word repeats. $k_1$ (typically ~1.2–2.0) controls *how fast* saturation kicks in: a smaller $k_1$ means the curve flattens sooner (repetition stops mattering quickly); larger $k_1$ lets frequency matter for longer before flattening. This is the same mathematical shape used in enzyme kinetics (Michaelis-Menten) and other "diminishing returns" phenomena in nature — a $x/(x+k)$ curve is the generic way to express "more is better, but with saturation."

### Fix 3: length normalization becomes a deliberate, tunable knob

TF-IDF already divided by document length once (inside TF), baking in a *fixed, total* length correction. BM25 questions whether length should be fully corrected for, not at all, or something in between — because both extremes are arguably wrong:

- No correction at all → long documents win by accident (more words = more chances to match).
- Full correction (dividing completely by length) → short documents are over-rewarded, since a short doc matching one term concentrates 100% of its content on it, which may or may not reflect genuine relevance versus just being terse.

BM25 introduces a parameter $b \in [0,1]$ that *dials* between these extremes:

$$1 - b + b \cdot \frac{|d|}{avgdl}$$

- If $b=0$: the whole expression equals 1 — no length normalization at all, exactly like ignoring length.
- If $b=1$: full normalization relative to the *average* document length in this specific corpus (not an absolute count, which is smarter than TF-IDF's raw division — it adapts to whatever "normal" looks like for this dataset).
- Values in between blend the two. This factor then gets folded into the denominator of the saturation function above, so that a document *longer than average* needs proportionally more occurrences to hit the same saturation point, and a document *shorter than average* saturates faster (short docs get a bit of a relevance boost for matching at all).

### Putting the rebuilt pieces together

$$BM25(D,Q) = \sum_{t \in Q} IDF_{BM25}(t) \cdot \frac{f(t,D)\cdot(k_1+1)}{f(t,D) + k_1\left(1-b+b\cdot\frac{|D|}{avgdl}\right)}$$

Notice the skeleton is *identical* to TF-IDF's "sum of IDF × TF across query terms" — BM25 isn't a different idea, it's the same idea with both factors upgraded from "linear, unbounded, occasionally degenerate" to "smoothed, saturating, tunable." Every constant in the formula ($0.5$, $k_1$, $b$) exists because a specific logical flaw in TF-IDF needed patching, not as arbitrary tuning knobs bolted on afterward.

### The one-sentence intuition to hold onto

TF-IDF says: *"rare words matter more, frequent occurrence matters more, multiply them."* BM25 says: *"...but nothing should ever be worth exactly zero, repetition should have diminishing returns, and 'long' is relative to your corpus, not absolute."* It's the same core insight, matured by asking "what does this formula do at the extremes?" and fixing every place the answer was uncomfortable.