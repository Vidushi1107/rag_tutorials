# Module 01 — The Naive RAG Pipeline

**Concept:** RAG is five stages glued together. Learn the stages, and everything else in this
course is a refinement of one of them.

## The problem RAG solves

An LLM only knows what was in its training data. Ask GPT-4o-mini about the payload rating of
the Helix Atlas R5 and it has three options, all bad: refuse, guess, or confidently invent a
number. It has never seen our corpus.

There are three ways to fix that:

| Approach | What it does | Cost of a change |
| --- | --- | --- |
| Longer prompt | Paste all your documents into every request | Pay for every token, every call; breaks past the context limit |
| Fine-tuning | Retrain weights on your data | Hours to days, and a full retrain per update |
| **RAG** | Fetch only the relevant passages, then ask | Re-index one document, seconds |

RAG wins whenever your knowledge changes more often than you want to retrain, which is almost
always. It also gives you something fine-tuning cannot: **provenance**. You know which document
produced the answer, so the answer can be checked.

## The five stages

```
                    ┌── INDEXING (done once, ahead of time) ──┐
   documents  →  1. LOAD  →  2. SPLIT  →  3. EMBED + STORE
                                                    │
                                                    ▼
                                              [ vector store ]
                                                    │
   question   →  4. RETRIEVE  ──────────────────────┘
                      │
                      ▼
                 5. GENERATE  →  answer
                    └── QUERYING (done per question) ──┘
```

**1. Load** — turn source files into `Document` objects (text + metadata). Metadata matters:
it is how you later filter and cite.

**2. Split** — cut documents into chunks. Whole documents are too coarse; a 3,000-word spec
sheet stuffed into the prompt buries the one sentence you needed. Chunks are the unit of
retrieval, so chunking decides the granularity of everything downstream. This is Module 02.

**3. Embed + store** — convert each chunk into a vector that encodes its meaning, and store it
in a vector database. Similar meanings land near each other in vector space.

**4. Retrieve** — embed the user's question with the *same* model, then find the nearest chunk
vectors. "Nearest" is a similarity metric, covered in Module 03.

**5. Generate** — paste the retrieved chunks into a prompt with the question and let the LLM
compose the answer, grounded in what you gave it.

Stages 1-3 happen once, ahead of time. Stages 4-5 happen per question. Confusing these two
phases is the most common beginner mistake — re-embedding your whole corpus on every query is
slow and expensive, and buys you nothing.

## Why this one is called "naive"

The pipeline in `pipeline.py` is the simplest thing that works, and it has real weaknesses:

- Fixed-size chunking splits mid-sentence and mid-table.
- It retrieves a fixed `k` chunks whether or not they are any good.
- Pure vector similarity misses exact keyword matches like the error code `E-204`.
- Nothing checks whether the answer is actually supported by the retrieved text.

Every one of those is a later module. Start here so you can feel the failure modes before you
learn the fixes.

## Run it

```bash
python modules/01-naive-rag-pipeline/pipeline.py
```

The script runs the same question twice — once with no retrieval, once through the full
pipeline — so you can see the difference directly.

## What to look for

- **The baseline answer is wrong or evasive.** That is the point. The model has no idea what an
  Atlas R5 is.
- **The retrieved chunks are printed before the answer.** Always inspect these. When a RAG
  system gives a bad answer, the cause is bad retrieval far more often than bad generation, and
  you cannot tell which without looking.
- **The grounded answer cites its source document.** Provenance is a feature you get for free
  here; later modules make it stricter.

## Exercises

1. Ask something the corpus cannot answer — "who is the Helix Robotics CFO?" A naive pipeline
   still retrieves its `k` nearest chunks and hands them over, because similarity search always
   returns *something*. Does the model admit it does not know, or does it improvise?
2. Set `CHUNK_SIZE` to 200, then to 3000. Watch the retrieved chunks change: too small loses the
   surrounding context, too large drowns the answer in noise.
3. Ask "what causes E-311?" and then "why won't my robot charge?" — same underlying issue,
   different words. The second phrasing has no keyword overlap with the corpus. Does vector
   search still find it?

## Next

Module 02 takes the crudest part of this pipeline — the fixed-size splitter — and shows what
better chunking is worth.
