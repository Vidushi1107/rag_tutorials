# Module 06 — Query Transformation

**Concept:** The user's question is rarely the best thing to search with. Transform it first.

## The asymmetry problem

Vector retrieval compares a question to a set of answers. But questions and answers do not look
alike:

> **Question:** "why won't my robot charge"
> **Answer text:** "E-311 — Charge fault. The robot docked but did not draw current. Check the
> dock contacts for debris..."

These share almost no vocabulary and, more importantly, no *shape*. One is a short informal
complaint; the other is a formal diagnostic entry. Embedding models place text by meaning, but
the meaning of a terse question and the meaning of a detailed answer land in noticeably
different regions. That gap is called the **asymmetry problem**, and it is the single most
common reason a retriever "just misses."

Real user queries make it worse. They are vague ("it's broken"), overloaded ("compare X and Y
and tell me about Z"), full of pronouns, or written in vocabulary the documents never use.

Query transformation fixes the query rather than the index.

## Four techniques

### 1. Query rewriting

Ask an LLM to turn a messy question into a clean search query. "why won't my robot charge" →
"robot charging fault dock not drawing current error code". Cheapest technique, works well on
conversational input, and it is essentially free insurance against sloppy phrasing.

### 2. Multi-query

Generate several rephrasings, retrieve for each, and union the results.

The insight: any single embedding is one point in vector space, and the "right" chunk might be
near a *different* phrasing of your question. Firing three or four differently-worded probes
covers more of the space. Recall goes up, at the cost of N retrievals.

Use it when recall matters more than latency. Deduplicate the union by chunk ID.

### 3. HyDE (Hypothetical Document Embeddings)

The clever one. Instead of making the query look better, make it look like *an answer*:

1. Ask the LLM to hallucinate a plausible answer to the question, ignoring that it does not know.
2. Embed **that fake answer** instead of the question.
3. Retrieve with it.

The hallucination is factually worthless — it will invent wrong numbers — but that does not
matter. It has the right *shape* and vocabulary: it reads like a document, so it lands near real
documents. You are searching answer-space with an answer-shaped probe rather than a
question-shaped one, which sidesteps the asymmetry directly.

Strong on technical corpora with distinctive vocabulary. Weaker when the model has no idea what
a plausible answer looks like — a fabrication about a domain it has never seen can land you
further away, not closer.

### 4. Step-back prompting

Ask a more general question first, retrieve for both, and combine.

"Can I run an Atlas R5 in a freezer at -20°C?" is narrow, and a literal search may find nothing.
Step back to "What are the Atlas R5 operating temperature limits?" and you retrieve the spec
that actually answers it. Good for questions requiring a principle rather than a stated fact.

### And: decomposition

Multi-part questions retrieve badly because their embedding is an average of several topics,
which lands between the relevant clusters instead of inside any of them. Split into
sub-questions, retrieve per sub-question, and answer from the merged context.

## What it costs

Every technique here trades LLM calls for retrieval quality:

| Technique | Extra LLM calls | Extra retrievals | Use when |
| --- | --- | --- | --- |
| Rewriting | 1 | 0 | Conversational or messy input |
| Multi-query | 1 | N | Recall matters more than latency |
| HyDE | 1 | 0 | Technical corpus, jargon-heavy |
| Step-back | 1 | 1 | Specific questions needing general context |
| Decomposition | 1 | N | Multi-part questions |

That is roughly +1 to +2 seconds of latency. Do not apply them unconditionally — Module 12
shows how to let the model decide when a query needs transforming.

## Run it

```bash
python modules/06-query-transformation/query_transformation.py
```

## What to look for

- **The vague query retrieves badly, then well after rewriting.** Same index, better probe.
- **HyDE's hypothetical answer contains invented numbers.** Read it. Then note that retrieval
  still improves — you are matching on shape and vocabulary, not facts.
- **Multi-query's variations overlap but not completely.** The union is the point.
- **Decomposition splits the compound question** and each part retrieves cleanly on its own.

## Exercises

1. Write a query where HyDE makes retrieval *worse*. Hint: pick something the model will
   confidently fabricate in the wrong vocabulary.
2. Push multi-query to 8 variations. Does recall keep improving, or does it plateau?
3. Combine step-back with HyDE. Does stacking help, or do the gains overlap?

## Next

Query transformation makes vector search better at what it already does. Module 07 addresses
what it fundamentally cannot do: exact matching.
