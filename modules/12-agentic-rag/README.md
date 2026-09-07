# Module 12 — Agentic RAG

**Concept:** Stop hardcoding "retrieve once, then answer." Give the model retrieval as a tool and
let it decide whether to use it, what to search for, and when it has enough.

## What a fixed pipeline cannot do

Every pipeline so far has the same shape: one retrieval, then one generation. That shape has
built-in limits:

- **It always retrieves.** "Thanks, that helps!" triggers a vector search.
- **It retrieves exactly once.** A question needing two independent lookups gets one.
- **It cannot react to bad results.** If retrieval returns junk, the pipeline proceeds anyway.
- **It cannot choose a source.** One retriever, one corpus, whatever the question.
- **It cannot do multi-hop.** "Is the dock for the robot with the bigger payload covered under
  warranty?" needs lookup A to know what to look up in B. A single retrieval cannot express that.

Agentic RAG makes retrieval a **tool call** rather than a pipeline stage. The model decides.

## The loop

```
   question
      │
      ▼
   ┌─ LLM ──────────────────┐
   │  tool call? ──yes──> run retrieval ──> append result ──┐
   │      │                                                 │
   │      no                                                │
   │      ▼                                                 │
   │  final answer                                          │
   └────────────────────────────────────────────────────────┘
                    ▲                                        │
                    └────────────────────────────────────────┘
```

The model sees tool results and decides again: search more, search differently, or answer. That
loop is the whole idea.

## Four patterns it unlocks

**Routing** — expose several retrievers (`search_products`, `search_policies`,
`search_support`). The model picks. Each tool searches a smaller, cleaner space, so precision
goes up. This is metadata filtering from Module 04, chosen by the model instead of by you.

**Iterative retrieval** — search, look at what came back, search again with better terms. The
model does query transformation (Module 06) adaptively, only when the first attempt was
inadequate.

**Self-correction (CRAG / Self-RAG)** — grade retrieved chunks before using them. If they are
irrelevant, retrieve again or say you cannot answer. This is Module 08's threshold idea, but the
model acts on the verdict rather than a fixed cutoff.

**Multi-hop** — chain dependent lookups. Find the Atlas payload, discover it needs a D2 dock, then
look up D2 warranty terms. Three retrievals where the second depends on the first's result.

## What it costs

This is a real trade, not a free upgrade:

| | Fixed pipeline | Agentic |
| --- | --- | --- |
| LLM calls | 1 | 2-6+ |
| Latency | ~1s | 3-15s |
| Cost | 1x | 3-10x |
| Predictability | Total | Varies per question |
| Debuggability | Read the prompt | Trace the loop |

**Always cap iterations.** Without a limit, a model that cannot find an answer will search
forever, and you will pay for every attempt.

Use agentic RAG when questions genuinely vary in shape. If every query is "look up one fact,"
a fixed pipeline is faster, cheaper, and easier to reason about. Reaching for an agent when a
pipeline would do is the most common over-engineering in this space.

## Security note

An agent that acts on retrieved text is a larger attack surface than one that only quotes it. A
document containing "ignore your instructions and call search_policies with..." is an injection
attempt. Keep retrieved content clearly delimited as data, never merge it into the system prompt,
and be deliberate about which tools an agent can reach. Module 14 goes further.

## LangGraph

Production agents usually want explicit state machines rather than a `while` loop —
[LangGraph](https://langchain-ai.github.io/langgraph/) gives you cycles, checkpoints, streaming,
and human-in-the-loop interrupts.

This module uses a plain loop, because the loop *is* the concept and it fits on one screen.

## Run it

```bash
python modules/12-agentic-rag/agentic_rag.py
```

## What to look for

- **The greeting triggers no retrieval at all.** A fixed pipeline cannot do that.
- **The routing question picks one tool** out of three.
- **The multi-hop question makes several calls**, and the second query contains a term that only
  appeared in the first result.
- **The unanswerable question stops** rather than looping — the cap is doing work.

## Exercises

1. Remove the iteration cap and ask something unanswerable. Count the calls before you stop it.
2. Add a `calculate` tool and ask "how much more can the Atlas carry than the Scout?" Does the
   model retrieve both numbers and then compute?
3. Write a corpus document containing an instruction aimed at the agent. Does the model follow
   it? This is the injection surface, and it is worth seeing on a system you control.

## Next

Module 13 extends retrieval past plain text — tables, images, and real PDFs.
