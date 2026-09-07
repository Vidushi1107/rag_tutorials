# Module 03 — Embeddings & Similarity

**Concept:** Retrieval is nearest-neighbour search in a space where distance means
dissimilarity of meaning. Understanding that space tells you what your retriever can and cannot
find.

## What an embedding is

An embedding model maps text to a fixed-length list of floats — 1,536 of them for OpenAI's
`text-embedding-3-small`. The coordinates are not individually interpretable; there is no
"charging-ness" dimension. What matters is the *arrangement*: texts about similar things end up
in similar directions, because the model was trained so that related text pairs land close
together.

That gives you the property the whole field rests on: **you can compare meaning with
arithmetic**. "How long does the battery last?" and "What is the runtime?" share no content
words, but their vectors point nearly the same way.

This is also why RAG beats keyword search on paraphrase — and, importantly, why it *loses* on
exact identifiers. `E-204` and `E-311` are near-identical strings that mean completely different
things, and an embedding model trained on natural language has little reason to separate them
sharply. Module 07 fixes this by putting keyword search back in alongside vectors.

## Measuring "close"

Three metrics show up in practice:

**Cosine similarity** — the angle between two vectors, ignoring their lengths. Range -1 to 1,
higher is more similar. This is the default for text, and the reason is length: a two-sentence
chunk and a two-paragraph chunk about the same topic have different vector magnitudes but
similar *directions*. Cosine looks only at direction, so it does not penalise a chunk for being
short.

**Dot product** — cosine, but scaled by both magnitudes. Cheaper (no normalisation step) and
equivalent to cosine *when the vectors are already unit length*. OpenAI returns normalised
vectors, so for those two, dot product and cosine give identical rankings. If your model does
not normalise, dot product will bias toward longer, higher-magnitude text.

**Euclidean (L2) distance** — straight-line distance. Lower is more similar, which is the
opposite convention and a common source of confusion when reading scores. On normalised vectors
L2 and cosine produce the same *ranking* — they are monotonically related — so the choice only
matters if your vectors are not normalised.

The practical summary: **use cosine unless you have a specific reason not to.** Chroma's default
is L2; the code in this module shows how to switch it, and why the ranking does not change for
OpenAI embeddings.

## Choosing an embedding model

| Factor | What to weigh |
| --- | --- |
| Quality | Check the MTEB retrieval leaderboard, not the overall average — a model can be great at classification and mediocre at retrieval |
| Dimensions | More dimensions = more nuance, more storage, slower search. `text-embedding-3-large` is 3,072d vs `-small`'s 1,536d for ~6x the price |
| Cost | You embed the whole corpus once, then one query per request. Corpus size drives cost, not traffic |
| Domain | General models underperform on legal, medical, and code. A domain model or a fine-tune can beat a bigger general model |
| Hosting | API models are simplest; local models (via `sentence-transformers`) cost nothing per call and keep data in-house |

Two rules that are not optional:

1. **Query and documents must use the same model.** Different models produce incomparable
   spaces. Mixing them does not error — it silently returns nonsense.
2. **Changing the model means re-embedding everything.** Budget for it.

`text-embedding-3-small` is the right default: cheap, fast, and strong enough that your chunking
will be the bottleneck long before your embedding model is.

## Matryoshka: shortening vectors on purpose

OpenAI's v3 models are trained so that the *first* N dimensions are independently useful. Pass
`dimensions=256` and you get a 6x smaller vector that retains most of the retrieval quality —
a real lever when your index gets large. The script measures the quality drop directly.

## Run it

```bash
python modules/03-embeddings-and-similarity/embeddings.py
```

## What to look for

- **Paraphrases score high without sharing words.** "How long does the battery last" vs "What is
  the runtime" — no overlap, high similarity.
- **`E-204` vs `E-311` score high despite meaning different things.** This is the failure that
  motivates hybrid search in Module 07.
- **Cosine and L2 produce the same ranking** on these normalised vectors, but the scores read
  in opposite directions.
- **Truncating to 256 dimensions barely changes the top result.**

## Exercises

1. Embed a sentence in English and its German translation. How similar are they? What does that
   tell you about multilingual corpora?
2. Find two chunks in the corpus with high similarity but different sources. Would retrieving
   both be useful, or redundant?
3. Compare `text-embedding-3-small` against `-large` on the same questions. Is the quality
   difference worth 6x the cost for this corpus?

## Next

Module 04 moves from "how are vectors compared" to "how do you search millions of them without
comparing against all of them" — indexing and vector stores.
