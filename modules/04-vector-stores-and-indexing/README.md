# Module 04 — Vector Stores & Indexing

**Concept:** A vector store is not just a place to put embeddings. It decides how fast you can
search, how much accuracy you trade for that speed, and what you can filter on.

## Why you cannot just compare everything

Module 03 scored a query against every chunk with a loop. That is **exact** search — perfect
recall, and O(n) per query. At 500 chunks it is instant. At 50 million chunks, with a 1,536-float
vector each, you are reading ~300 GB per query.

So real vector stores use **approximate nearest neighbour** (ANN) search: give up a small,
tunable amount of accuracy for orders of magnitude more speed.

This is the trade you are making, and it should be a deliberate one:

- **Recall@k** — of the true top-k nearest vectors, what fraction did the index actually return?
- 100% recall means exact search. Production systems typically run 95-99%.
- Missing 2% of true neighbours is usually invisible in answer quality. Being 100x slower is not.

## The index types

**Flat (brute force)** — compare against everything. Perfect recall, no build time, linear
search. Correct choice under ~10k vectors, and the baseline you measure the others against.

**HNSW (Hierarchical Navigable Small World)** — a layered proximity graph. Search enters at a
sparse top layer, greedily walks toward the query, then descends into denser layers to refine.
Fast and high-recall, at the cost of memory (the graph is held in RAM) and slower inserts. **This
is the default in most modern stores**, Chroma included.

Two knobs matter:
- `ef_construction` — effort spent building the graph. Higher = better graph, slower build.
- `ef_search` — how many candidates to explore per query. **This is your recall/latency dial at
  runtime**, and the one you will actually tune.

**IVF (Inverted File)** — cluster the vectors, then search only the clusters nearest the query.
Cheaper memory than HNSW, needs a training pass over representative data, and recall depends on
`nprobe` (how many clusters to visit). Common in FAISS at very large scale.

**Product Quantisation (PQ)** — compress vectors into codes rather than storing floats. Often
layered onto IVF (`IVF-PQ`) for billion-scale indexes. Big memory savings, real accuracy cost.

Rule of thumb: **flat under 10k, HNSW up to tens of millions, IVF-PQ beyond that.**

## Metadata filtering is the underrated feature

"What is the payload?" is ambiguous across the Atlas and Scout specs. Filtering by
`source == "product-atlas-r5.md"` removes the ambiguity entirely — and it is far more reliable
than hoping the embedding picks the right document.

Design your metadata at ingestion time. Retrofitting it means re-indexing.

One caveat worth knowing: **pre-filtering vs post-filtering**. Post-filtering fetches k results
then discards non-matching ones — so a narrow filter can leave you with almost nothing.
Pre-filtering restricts the search space first and returns a full k. Chroma pre-filters. Not all
stores do; check before you rely on it.

## MMR: fixing redundancy

Plain similarity search happily returns four near-identical chunks. If they all say the same
thing you have spent your context budget on one fact.

**Maximal Marginal Relevance** re-ranks candidates to balance relevance against diversity: fetch
a wider pool, then greedily pick results that are relevant *and* unlike what is already chosen.
The `lambda_mult` parameter sets the balance (1.0 = pure relevance, 0.0 = pure diversity).

Use it when your corpus has redundancy — overlapping chunks, duplicated boilerplate, multiple
documents covering the same ground. Which is to say: usually.

## Choosing a store

| Store | Use when |
| --- | --- |
| **Chroma** | Prototyping, local dev, small-to-mid corpora. Zero setup. Used throughout this course |
| **FAISS** | Library not a server; maximum control over index type; in-process, no persistence layer of its own |
| **pgvector** | You already run Postgres and want vectors alongside relational data in one transaction |
| **Qdrant / Weaviate** | Self-hosted production, rich filtering, horizontal scale |
| **Pinecone** | Fully managed, do not want to operate anything |

Do not agonise over this early. LangChain's `VectorStore` interface makes these largely
swappable — build on Chroma, migrate when you have real numbers.

## Run it

```bash
python modules/04-vector-stores-and-indexing/vector_stores.py
```

Note that this module writes a persisted store to `chroma_db/` (gitignored) to demonstrate that
an index survives process restarts.

## What to look for

- **Persistence.** The second run loads the existing index instead of re-embedding — the whole
  point of separating indexing from querying.
- **The ambiguous payload question** returns Scout chunks until you filter to the Atlas doc.
- **MMR vs similarity** on a question with redundant coverage: same query, visibly less
  repetition.
- **Updating a chunk by ID** changes the retrieved text without rebuilding the index.

## Exercises

1. Delete the Atlas document from the store, then ask an Atlas question. What comes back? This is
   how deletion behaves in production when a document is retracted.
2. Set `lambda_mult` to 0.1 and then 0.9. Where does diversity start hurting relevance?
3. Time exact search against HNSW on the corpus. At 500 chunks, is the index earning its keep?

## Next

Retrieval is now solid. Module 05 turns to the other half: what you actually do with the chunks
once you have them.
