## The core mathematical shapes underlying TF-IDF and BM25

Every formula we built up rests on a small number of *curve shapes* — the actual visual behavior of a function as its input grows. Understanding these shapes is what makes the formulas feel inevitable rather than arbitrary. Let's walk through each one, in the order they get used.

---

### 1. The linear function — the "naive" baseline everything else improves on

$$f(x) = x$$

Shape: a straight line through the origin, constant slope, no curvature. Every unit increase in $x$ produces exactly the same unit increase in $f(x)$ — the 1st increment matters exactly as much as the 1,000th.

This is raw term count, $f(t,d)$, before any correction. Its graph is a plain diagonal ray from (0,0) going up and to the right forever, unbounded. This is the *default assumption we're rejecting*. When we said "raw TF grows forever and treats the 100th occurrence the same as the 1st," this straight-line graph is exactly what's wrong: it has no memory of "I've already made my point," it just keeps climbing at the same rate indefinitely. Every fix that follows exists to *bend this line* into something with a conscience.

---

### 2. The logarithm curve — the backbone of IDF

$$f(x) = \log(x)$$

This is the single most important shape in the whole system, so let's look at it closely.

**Shape**: For $x$ between 0 and 1, $\log(x)$ is negative and plunges steeply toward $-\infty$ as $x \to 0$. At $x=1$, $\log(x) = 0$ exactly — this is the crossing point. For $x > 1$, the curve rises, but its slope *keeps shrinking* as $x$ grows — it climbs fast at first, then flattens into a long, lazy, ever-rising but nearly-horizontal tail. It never turns downward and never hits a ceiling, but it grows unboundedly *slowly*.

**Key property — concavity**: the log curve is concave (bends downward/flattens as you move right). This is the mathematical signature of "diminishing returns." The derivative of $\log(x)$ is $1/x$ — meaning the *rate of increase* is largest when $x$ is small and shrinks as $x$ grows. Concretely: $\log(2) - \log(1) \approx 0.693$, but $\log(1000) - \log(999) \approx 0.001$. Doubling from 1 to 2 moves the curve a huge amount; moving from 999 to 1000 barely moves it at all. That's exactly the "10x rarer matters about the same regardless of where you start" property IDF needed — because $\log$ converts *ratios* into equally-spaced *differences*. $\log(N/df)$ for $df$ shrinking by a factor of 10 each time (1000 → 100 → 10 → 1) produces IDF values that increase by the *same fixed amount* each step, not an accelerating amount. That's the graph doing the "multiplicative → additive" conversion described earlier, made visible.

**Where IDF sits on this curve**: since $N/df(t) \geq 1$ always (a term can't appear in more documents than exist), IDF's input never dips below 1, so we're always on the rising-but-flattening branch, output always $\geq 0$. As $df \to N$ (term in every doc), input $\to 1$, and IDF $\to 0$ — this is literally the curve crossing zero at $x=1$. As $df \to$ small numbers relative to $N$, input grows large, IDF grows too, but slowly, thanks to the flattening tail — so no single ultra-rare term can blow the score up to infinity.

---

### 3. The saturating rational curve — BM25's fix for term frequency

$$f(x) = \frac{x}{x+k}$$

(BM25's actual TF factor is $\frac{x \cdot (k_1+1)}{x+k_1}$, a rescaled version of this same shape — the $(k_1+1)$ just stretches the ceiling upward.)

**Shape**: starts at $f(0) = 0$, rises steeply at first, then bends over into a horizontal asymptote. As $x \to \infty$, $f(x) \to 1$ (or $\to k_1+1$ in BM25's rescaled version) — a hard ceiling the curve approaches but never touches. This is sometimes called a "rectangular hyperbola" shape, and it's the exact curve used in the Michaelis-Menten model of enzyme reaction rates in biochemistry — which is not a coincidence; both are modeling the identical underlying phenomenon: *a resource with increasing input but a saturating effect because something else becomes the bottleneck*. In enzyme kinetics the bottleneck is the enzyme's available binding sites; in BM25 the "bottleneck" is a deliberate modeling choice that repetition should stop adding new relevance signal.

**Contrast with the log curve**: both curves are concave/diminishing-returns shaped, but they behave very differently at the extreme. Log climbs forever, just slower and slower. This rational curve actually **caps out at a hard ceiling** and essentially stops moving once $x$ passes a few multiples of $k$. That's a stronger, more literal form of "enough is enough" than log provides — which is exactly what TF needs (repetition truly should stop mattering past some point), versus what IDF needs (rarity should keep mattering a little, forever, just less and less).

**Role of $k_1$**: $k_1$ sets *where* the knee of the curve sits — the input value at which the curve is roughly halfway to its ceiling. At $x = k_1$, $f(x) = \frac{k_1}{2k_1} = 0.5$ of the max (in the unscaled version). Small $k_1$ → the knee sits near the origin → saturation kicks in almost immediately (even 2-3 occurrences are "enough"). Large $k_1$ → the knee sits further out → the curve stays closer to the linear line for longer before bending over, meaning more repetitions continue to matter before diminishing returns set in.

---

### 4. The log-odds curve — BM25's refined IDF

$$f(x) = \ln\left(\frac{1-x}{x}\right) \quad \text{(where } x = df(t)/N \text{, the fraction of docs containing } t\text{)}$$

This is the *logit function*, the same curve that underlies logistic regression in statistics.

**Shape**: defined for $x$ strictly between 0 and 1 (a probability). As $x \to 0$ (term almost never appears — extremely rare), $f(x) \to +\infty$ — steep upward blow-up. As $x \to 1$ (term appears in almost every document), $f(x) \to -\infty$ — steep downward blow-up. At $x = 0.5$ (term in exactly half the documents), $f(x) = \ln(1) = 0$ — the natural zero-crossing, a genuine "no information either way" point. The curve is a mirror-image S-shape rotated 90°, steep at both ends, and it is **antisymmetric** around $x=0.5$: a term in 10% of docs gets the exact opposite-sign score of a term in 90% of docs.

**Why this is a better shape for IDF than plain $\log(N/df)$**: plain TF-IDF's IDF only has one "danger zone" — the top end, where $df \to N$ and the score collapses to exactly 0. It never goes negative, because $\log(N/df)$ can't go below $\log(1)=0$ (since $df \leq N$ always). BM25's smoothed log-odds version, by contrast, *can* dip slightly negative for extremely common terms (via the $+0.5$ smoothing, in practice it stays just above zero for realistic corpora, but the *shape* it's built from is this fully-symmetric log-odds curve) — meaning the formula has room to express "this term is actively *unhelpful* for distinguishing documents," not just "unhelpful, floor at zero." This is the graph-level reason BM25's IDF avoids TF-IDF's hard zero-cliff: log-odds simply doesn't have a cliff, it has a smooth, symmetric, ever-so-gently-sloping curve through its whole domain.

---

### 5. The linear blend / interpolation line — length normalization

$$f(x) = (1-b) + b \cdot x, \quad x = \frac{|d|}{avgdl}$$

**Shape**: this one actually *is* a straight line — but its *purpose* is different from the raw-count line in section 1. Here, $x=1$ (document length equals corpus average) always gives $f(x)=1$, the neutral point, regardless of $b$. The **slope of this line is controlled by $b$**: at $b=0$ the line is perfectly flat (horizontal, always equal to 1, i.e., length is ignored entirely). At $b=1$ the line has the steepest slope (full proportional scaling with length). Values of $b$ between 0 and 1 give intermediate slopes — the line pivots around the point $(1,1)$ like a hinge, tilting flatter or steeper as $b$ moves. This is why $b$ is described as a "dial" — graphically, you're literally choosing the slope of a line that pivots at "average length."

This linear term then sits inside the denominator of the saturating curve from section 3, which has an important compounding effect: making the denominator larger (long document, $x>1$, $b>0$) pushes the whole saturation curve to the *right* — meaning a long document needs a *higher* raw term count $f(t,d)$ to reach the same saturation point that a short document reaches with fewer occurrences. Graphically, you're not just adding a correction — you're *stretching the x-axis* of the saturation curve differently depending on document length.

---

### How the five shapes fit together into one picture

If you were to sketch the full BM25 term contribution for a single query word as a function of raw count $f(t,d)$, holding everything else fixed, you'd see: a curve that starts at the origin, rises with a slope proportional to $IDF_{BM25}(t)$ (itself pinned at a height determined by the log-odds curve, section 4), climbs quickly for the first few occurrences, and then bends over into a horizontal ceiling near $(k_1+1)\times IDF_{BM25}(t)$ — the saturation shape from section 3. Change the document's length, and the *whole curve stretches sideways* (section 5) before you even get to plug in the occurrence count. Compare that entire family of curves against TF-IDF's contribution — a straight, unbounded, ever-rising line (section 1) scaled by an IDF that can hit a hard zero (section 2) — and the difference in *shape*, not just formula, is the real story: TF-IDF is built from a line and a curve with a cliff; BM25 replaces both with curves that bend, saturate, and never fully die out.