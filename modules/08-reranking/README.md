# Module 08 — Reranking

**Concept:** Retrieve wide with a fast, sloppy method. Then rerank narrow with a slow, accurate
one. Two stages beat one.

## Why the first stage has to be sloppy

Every retriever so far has been a **bi-encoder**: the query is embedded, each chunk is embedded,
and the two vectors are compared.

The critical property is that chunk embeddings are computed **before** any query exists. That is
what makes search fast — you precompute once and reuse forever. It is also what makes it
inaccurate: the chunk's vector must summarise it for *every possible future question*, so it
cannot emphasise the part you actually asked about.

A **cross-encoder** does the opposite. It takes `(query, chunk)` as a single joined input and
runs them through a transformer together, with full attention across both. It sees which specific
words in the chunk relate to which words in the query. That is dramatically more accurate.

And it is unusable as your only retriever, because nothing can be precomputed. Scoring a query
against 1 million chunks means 1 million forward passes.

| | Bi-encoder | Cross-encoder |
| --- | --- | --- |
| When encoded | Chunks ahead of time | Query and chunk together, at query time |
| Cost per query | 1 embedding + fast ANN search | One model pass **per candidate** |
| Accuracy | Good | Substantially better |
| Scales to | Millions of chunks | Tens of candidates |

## The two-stage pattern

```
1M chunks  --bi-encoder (fast)-->  top 25 candidates  --cross-encoder (accurate)-->  top 4
```

You get most of the cross-encoder's accuracy at a fraction of the cost, because it only ever
sees 25 candidates.

**This is usually the single highest-leverage upgrade to a naive RAG pipeline.** More than better
chunking, more than a bigger embedding model.

### Picking the candidate count

The first stage only has to get the right chunk *somewhere* in its list — the reranker fixes the
ordering. So widen it: `k=25` or `k=50` where you would have used `k=4`.

More candidates means better recall and more reranking cost. Past ~50 the returns flatten while
latency does not. Start at 25.

Note the interaction with Module 07: a wide hybrid first stage plus a reranker is the standard
strong-baseline architecture. Hybrid maximises the chance the right chunk is in the pool;
reranking makes sure it surfaces.

## Three ways to rerank

**Local cross-encoder** (`sentence-transformers`) — models like
`cross-encoder/ms-marco-MiniLM-L-6-v2` are small, run on CPU in tens of milliseconds, and cost
nothing per call. Requires torch. **The default choice.**

**Hosted rerank API** (Cohere Rerank, Voyage) — best quality, no infrastructure, per-call cost
and a network round-trip. Good when quality matters more than latency and you do not want to
host models.

**LLM-as-reranker** — ask an LLM to score relevance. No extra dependencies, works with the API
key you already have, and is easy to reason about. Slower and pricier per candidate than a
dedicated cross-encoder, but genuinely competitive on quality, and it is the technique used in
`ContextualCompressionRetriever`-style filters. The script uses this as its primary example so
it runs with no heavy installs.

## The cost of not reranking

Without a reranker you must set `k` low, because everything retrieved goes into the prompt and
context is expensive. Low `k` means one bad retrieval and the answer is wrong.

With a reranker you retrieve 25 and *pass 4*. The extra 21 cost you a cheap scoring pass, not
prompt tokens. You are buying recall without paying for it in context.

## Run it

```bash
python modules/08-reranking/reranking.py

# optional, for the local cross-encoder section (large download — torch)
pip install sentence-transformers
```

## What to look for

- **The rank ordering changes.** Chunks the bi-encoder put 7th land at 1st. That reordering is
  the entire value.
- **Relevance scores are better separated** than cosine similarities, which bunch up around
  0.3-0.5 and discriminate poorly.
- **A relevance threshold can return nothing** — unlike similarity search, which always returns
  `k`. This is how you finally get "I don't know" from the retrieval layer instead of the prompt.

## Exercises

1. Sweep the candidate pool from 5 to 50. Where does the final answer stop improving?
2. Set a relevance threshold and ask an unanswerable question. Does the pipeline correctly
   return zero chunks?
3. Compare a wide-hybrid + rerank pipeline against dense-only + rerank. Which contributes more?

## Next

You now have a lot of dials — chunk size, `k`, weights, thresholds. Module 09 is about how to
tell whether turning any of them actually helped.
