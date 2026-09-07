# Module 14 — Production Readiness

**Concept:** Everything between a pipeline that works on your machine and one you can put in
front of users.

## Caching

Three distinct caches, often confused:

**Embedding cache.** The same text embeds to the same vector, always. Cache by content hash and
re-indexing an unchanged document costs nothing. This is the highest-value cache in RAG, because
re-ingestion is where your embedding spend actually goes. `CacheBackedEmbeddings` does it in two
lines.

**Exact-match LLM cache.** Identical prompt in, cached response out. Trivially correct, and it
only fires on byte-identical prompts — which in RAG means identical retrieved context too. Hit
rates are lower than you would guess.

**Semantic cache.** Embed the query; if a previous query is close enough, reuse its answer. Much
higher hit rate, and genuinely risky: "What is the Atlas payload?" and "What is the Scout
payload?" are semantically close and have different answers. Set the threshold high (0.95+), scope
the cache per user or tenant, and never cache anything personalised.

## Know your numbers

You cannot optimise what you have not measured. Instrument per request:

- Retrieval latency vs generation latency — they have different fixes
- Tokens in and out, and cost
- Number of chunks retrieved, and which documents
- Cache hit or miss

In a typical pipeline, generation dominates latency (~80%). That points at streaming, a smaller
model, or shorter context — not at a faster vector store.

Where the money goes is less obvious. Indexing is a large one-time cost; queries are small and
recurring. Re-indexing a big corpus on every deploy quietly costs more than serving it.

## Streaming

Time to *first* token matters more to users than time to last. A streamed answer starting in 400ms
feels faster than a complete one at 2s, even though it finishes later.

Retrieval must complete before generation starts, so it is on the critical path. Stream the
answer, and consider showing sources as soon as retrieval returns — users get feedback while the
model is still writing.

## Guardrails

**Prompt injection through retrieved content is the RAG-specific vulnerability.** Your pipeline
takes text from documents and puts it in a prompt. If an attacker can influence any indexed
document — a support ticket, a wiki page, a scraped site — they can attempt to inject
instructions.

Defences, layered:

- Keep retrieved content in a **user** message, never merged into the system prompt.
- Delimit it clearly and state that it is data, not instructions.
- Never let retrieved text decide tool calls without validation (see Module 12).
- Validate output: check citations resolve to chunks you actually retrieved.
- Control who can write to your index. This is the real fix — treat index write access with the
  same seriousness as database write access.

Also worth handling: **PII** (redact at ingestion, not at query time — once it is embedded it is
in your index), and **permissions** (filter by the *requesting user's* access at query time; a
shared index with no access filter will eventually leak).

## Observability

Log enough to debug a bad answer after the fact. The minimum: query, condensed query, retrieved
chunk IDs, scores, the final prompt, the response, and latency.

When a user reports a wrong answer, the first question is always "what did it retrieve?" Without
that logged, you are guessing. LangSmith gives you this for free with two environment variables;
plain callbacks work fine too.

## Keeping the index fresh

Full re-indexing is simple and wasteful. Incremental indexing tracks content hashes and only
re-embeds what changed — LangChain's `index()` API handles the bookkeeping, including deleting
chunks whose source document disappeared.

Decide the freshness requirement explicitly. Hourly, daily, and real-time are very different
architectures, and picking one by accident is expensive.

## Failure modes worth handling

| Failure | Response |
| --- | --- |
| Vector store unreachable | Fail loudly. Answering without retrieval is worse than an error |
| LLM rate limited | Exponential backoff, then a queue |
| Nothing retrieved above threshold | Say so. Do not generate from nothing |
| Retrieved context exceeds budget | Truncate by relevance, keep the top chunks |
| Embedding model changed | Version your index; never mix embedding spaces |

That last one is a real outage waiting to happen. Stamp the embedding model name into your
collection metadata and refuse to query on a mismatch.

## A pre-launch checklist

- [ ] Evaluation set exists and runs in CI (Module 09)
- [ ] Retrieved chunks logged for every request
- [ ] Cost per query measured and budgeted
- [ ] Retrieval and generation latency tracked separately
- [ ] Embedding cache in place for re-indexing
- [ ] Index write access controlled and audited
- [ ] Retrieved content isolated from system instructions
- [ ] "I don't know" verified on a real unanswerable question
- [ ] Index version pinned to an embedding model
- [ ] Rate-limit and timeout behaviour tested under load

## Run it

```bash
python modules/14-production-readiness/production.py
```

## What to look for

- **The second identical query is near-instant** — the LLM cache.
- **Re-embedding an unchanged corpus makes zero API calls** — the embedding cache.
- **Generation dominates the latency breakdown.** That is where to optimise.
- **The injection demo.** A poisoned document tries to hijack the answer; watch whether the
  hardened prompt holds.

## Exercises

1. Add a semantic cache and find two questions close enough to collide wrongly. How high does the
   threshold need to go?
2. Write the citation validator: check every `[n]` in an answer against the chunks actually
   retrieved. How often does it catch something?
3. Take the checklist above and score a pipeline you have built. What is missing?

## You've finished the course

You can now build a RAG system, measure it, improve it deliberately, and reason about what it
will do in production. The remaining depth is corpus-specific — and Module 09 is how you find it.
