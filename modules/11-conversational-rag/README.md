# Module 11 — Conversational RAG

**Concept:** Follow-up questions are not self-contained. "What about the Scout?" means nothing to
a retriever. Rewrite it against the history before you search.

## What breaks

Every pipeline so far treated each question as standalone. Real conversations do not work that
way:

> **User:** What's the payload of the Atlas R5?
> **Assistant:** 450 kg.
> **User:** How about the Scout?

Embed `"How about the Scout?"` and search. You will get something — similarity search always
does — but the query carries none of the actual intent. The words "payload," "capacity," and
"kg" appear nowhere in it.

Three ways human conversation breaks retrieval:

- **Pronouns:** "how long does *it* take to charge?"
- **Ellipsis:** "and the Scout?" — the verb and object are simply gone
- **Implicit context:** "is that covered?" — *that* refers to something three turns back

## The fix: condense before retrieving

Insert a step. Given the history and the new question, ask an LLM to rewrite it as a standalone
question, then retrieve with the rewrite.

```
history + "How about the Scout?"
        │
        ▼  LLM condensation
"What is the maximum payload of the Scout M2?"
        │
        ▼  retrieve with this
```

Two rules that matter more than they look:

**Rewrite only — never answer.** The condenser's job is one sentence of search text. A condenser
that starts answering produces long, contaminated queries.

**Pass through what is already standalone.** If the question stands on its own, return it
unchanged. Over-eager rewriting damages good queries by "helpfully" injecting stale context from
earlier turns.

Note that the *rewritten* question goes to the retriever, but the **original** question and the
full history go to the answering prompt. The rewrite is a search artefact, not a replacement for
what the user said.

## Managing history growth

History grows without bound; context windows and budgets do not.

| Strategy | How | Trade-off |
| --- | --- | --- |
| **Full history** | Send everything | Simple; cost grows every turn |
| **Windowed** | Keep the last N turns | Cheap, predictable; drops old references |
| **Summary** | Summarise old turns, keep recent ones verbatim | Retains long-range context; costs an LLM call, and summaries lose detail |
| **Summary + window** | Both | The usual production answer |

Start windowed with N=5. Add summarisation when users actually reference things from far back.

One caveat: condensation quality degrades with a long history, because the model has more chances
to pull in irrelevant earlier context. Condense against the last few turns even when you send
more history to the answering prompt.

## Not every turn needs retrieval

"Thanks!" and "can you rephrase that?" do not need a vector search. Retrieving anyway costs
latency and can drag irrelevant chunks into the answer.

You can gate this with a cheap classifier call, or let the model decide — which is exactly what
Module 12 is about.

## Session isolation

Obvious but easy to get wrong: histories must be keyed per session. Leaking one user's history
into another's is both a correctness bug and a privacy incident. `RunnableWithMessageHistory`
handles the plumbing; the store behind it is yours to make durable.

## Run it

```bash
python modules/11-conversational-rag/conversational_rag.py
```

## What to look for

- **The naive follow-up retrieves the wrong document.** Printed first, so you can see the failure
  before the fix.
- **The condensed question is fully standalone** — it names the product and the attribute that
  were only implied.
- **An already-standalone question passes through unchanged.**
- **Token count per turn climbs** with full history and stays flat when windowed.

## Exercises

1. Build a five-turn conversation where turn 5 refers to turn 1. At what window size does it
   break?
2. Ask a follow-up that changes topic entirely. Does the condenser wrongly drag in old context?
3. Add a gate that skips retrieval when the condenser output is near-identical to a greeting.
   How much latency does that save on a chatty conversation?

## Next

Module 12 removes the last hardcoded assumption: that retrieval happens exactly once, always,
before generation.
