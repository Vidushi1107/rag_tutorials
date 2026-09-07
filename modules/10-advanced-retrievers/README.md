# Module 10 — Advanced Retrievers

**Concept:** Stop assuming the text you *search* has to be the text you *return*. Decoupling them
dissolves the chunk-size dilemma.

## The dilemma, restated

Module 02 framed chunk size as a trade-off with no good answer:

- **Small chunks** embed precisely — the vector represents one idea — but arrive at the LLM
  stripped of context. A chunk saying "42 minutes to 80%" does not say what charges in 42 minutes.
- **Large chunks** carry context but embed imprecisely, because the vector averages everything in
  them, and a specific query matches that average poorly.

Every technique in this module makes the same move: **index one representation, return a
different one.** Search over something small and precise; hand the LLM something large and
complete.

Once you see it, the four techniques below are variations on one idea.

## Parent document retrieval

Split twice. Index the small children; return the parent.

```
parent (2000 chars) ──┬── child (400) ──> embedded and searched
                      ├── child (400) ──> embedded and searched
                      └── child (400) ──> embedded and searched

a child matches  ->  return the whole parent
```

Precision of small chunks, context of large ones. **The best default upgrade** over plain
chunking, and the one to reach for first.

## Sentence-window retrieval

The extreme version: index individual sentences, return the sentence plus its *k* neighbours.

Maximum retrieval precision, and the returned window is contiguous prose rather than a fixed
block. Works well on flowing documents; less well on tables and lists, where "neighbouring
sentence" is not meaningful.

## Multi-vector retrieval

The most flexible: store several *derived* representations per chunk, all pointing back to the
original.

Useful derivations:

- **Summaries** — a dense chunk becomes a clean summary that embeds well. Search summaries,
  return the full text.
- **Hypothetical questions** — ask an LLM "what questions does this passage answer?", index those.
  Now you are matching questions to questions, which sidesteps the asymmetry problem from Module
  06 at *index* time rather than query time.
- **Tables** — index a natural-language description, return the raw table. This is the backbone
  of Module 13.

Costs an LLM pass over the corpus at ingestion. That is a one-time cost, and it buys the largest
quality gain of anything here.

## Self-query retrieval

Different axis. The LLM reads the natural-language question and extracts **metadata filters**
from it.

> "What error codes are in the support FAQ?"
> → query: `"error codes"`, filter: `source == "support-faq.md"`

Module 04 showed metadata filtering was powerful but required *you* to know the filter. Self-query
infers it from the question. Excellent when your corpus has strong structured attributes — dates,
categories, prices, versions.

The requirement: you must describe your metadata schema to the LLM up front, accurately. A wrong
description produces filters that silently match nothing.

## Choosing

| Situation | Use |
| --- | --- |
| General upgrade over fixed chunking | Parent document |
| Flowing prose, need tight precision | Sentence window |
| Dense/jargon text, or tables | Multi-vector with summaries |
| Questions that name attributes ("2024 policies") | Self-query |
| Strong structured metadata | Self-query + parent document |

These compose. A multi-vector retriever indexing hypothetical questions, feeding a reranker
(Module 08), is close to state of the art for text corpora.

## Run it

```bash
pip install lark        # needed by the self-query retriever
python modules/10-advanced-retrievers/advanced_retrievers.py
```

## What to look for

- **Parent retrieval returns far more text than it matched on.** Compare the matched child
  against the returned parent.
- **Hypothetical questions look strikingly like real user queries.** That is why they retrieve
  well.
- **The self-query retriever prints the filter it inferred.** Watch it turn a phrase into a
  structured constraint.
- **Sentence windows read as prose,** not as chunks cut at arbitrary boundaries.

## Exercises

1. Ask a question answerable only by combining a specific number with surrounding context. Which
   retriever handles it best?
2. Index hypothetical questions and inspect them. Are any wrong? A bad generated question is a
   permanent bad entry in your index — how would you catch that at ingestion time?
3. Give the self-query retriever a question with no metadata constraint at all. Does it correctly
   apply no filter, or invent one?

## Next

Module 11 adds the dimension every real deployment needs: conversation.
