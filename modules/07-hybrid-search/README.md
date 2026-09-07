# Module 07 — Hybrid Search

**Concept:** Vector search and keyword search fail in opposite directions. Run both and fuse the
results.

## The blind spot from Module 03

We measured this already: `E-204` and `E-311` scored ~0.9 cosine similarity despite being
unrelated faults. Embeddings compress text into a few hundred floats, and that compression
discards exactly the thing an identifier *is* — its precise surface form.

So a user searching `E-311` may get chunks about E-204, E-101, and E-450, all "about error
codes," none the right one. The failure is quiet: you get confident, well-formed, wrong answers.

Now flip it. Keyword search on "why won't my robot charge" scores zero against the E-311 chunk,
because the chunk never uses the word "won't" and the query never uses "fault."

**Neither method is better. They fail on different queries.**

| | Dense (vectors) | Sparse (BM25) |
| --- | --- | --- |
| Paraphrase, synonyms | Strong | Fails |
| Exact IDs, codes, names | Weak | Strong |
| Rare/out-of-vocabulary terms | Weak — unseen tokens embed poorly | Strong — a rare term is a *stronger* signal |
| Typos | Tolerant | Brittle |
| Needs training data | Yes | No |
| Cost | Embedding call per query | Nothing |

## BM25 in one paragraph

BM25 scores a document by how often the query's terms appear in it, with two corrections that
make it work in practice. **Saturation:** the tenth occurrence of "charge" adds far less than the
second — term frequency has diminishing returns. **Length normalisation:** a long document is not
more relevant just because it has more room to contain your words. Terms rare across the whole
corpus count for more than common ones, which is why BM25 latches onto identifiers like `E-311`.

It is a bag-of-words method with no notion of meaning, and it has been the information-retrieval
baseline for thirty years because it is fast, needs no training, and is very hard to beat on
exact-term queries.

## Fusing two ranked lists

Here is the trap: **you cannot average the scores.** BM25 scores are unbounded and
corpus-dependent (0 to 15, or 0 to 40 — it depends). Cosine similarity is bounded to [-1, 1].
Adding them lets BM25's scale silently dominate the result.

Two ways out:

**Reciprocal Rank Fusion (RRF)** — discard the scores, keep only the ranks:

```
score(doc) = Σ  1 / (k + rank_in_that_list)        k ≈ 60 by convention
           lists
```

Because it only uses ranks, RRF fuses any number of retrievers with no normalisation, no tuning,
and no assumptions about score distributions. The constant `k` damps the influence of top
positions so one list cannot completely dictate the outcome. It is the default for good reason:
robust, parameter-free, and it works.

**Weighted score fusion** — normalise each list's scores to [0, 1], then take a weighted sum.
Lets you express "trust vectors 70%, keywords 30%," but you have to tune the weights per corpus,
and min-max normalisation is unstable when a list's scores are tightly clustered.

Start with RRF. Move to weighted fusion only if you have evaluation data (Module 09) proving it
helps.

## Run it

```bash
pip install rank_bm25
python modules/07-hybrid-search/hybrid_search.py
```

## What to look for

- **`E-311` — BM25 nails it, dense search misses.** The exact-identifier failure, live.
- **"robot won't charge" — dense finds it, BM25 returns nothing useful.** The mirror image.
- **Hybrid handles both.** That is the whole argument for the extra complexity.
- **The raw score ranges are wildly different.** Look at them before you consider averaging.

## Exercises

1. Find a query where hybrid is *worse* than either method alone. They exist — fusion can pull a
   mediocre-but-agreed-upon result above a great one that only one retriever found.
2. Change RRF's `k` from 60 to 1 and to 500. What does it do to how much the top rank matters?
3. BM25 needs the whole corpus in memory. Sketch what changes at 10M documents. (Answer:
   Elasticsearch/OpenSearch, or a store with native hybrid support like Qdrant or Weaviate.)

## Next

Hybrid search improves the *candidate set*. Module 08 adds a second pass that reorders those
candidates with a much more accurate — and much slower — model.
