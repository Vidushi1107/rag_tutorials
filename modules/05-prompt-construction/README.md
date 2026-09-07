# Module 05 — Prompt Construction & Context Assembly

**Concept:** Retrieval hands you a pile of chunks. How you format, order, budget, and instruct
around them determines whether the model uses them, ignores them, or contradicts them.

This is the last Tier 1 module, and it completes the picture: modules 01-04 got the right text
into your hands, this one gets it used correctly.

## Formatting: chunks are not a blob

The laziest assembly — `"\n".join(chunk_texts)` — throws away everything the model needs to
reason about *provenance*. Where did each claim come from? Are these two passages from the same
document or two contradictory ones?

Give each chunk a visible boundary and a label:

```
[1] source: product-atlas-r5.md | section: Power
Battery: 48 V lithium iron phosphate...

[2] source: support-faq.md | section: Error codes
E-311 — Charge fault...
```

That costs a handful of tokens per chunk and buys you two things: the model can cite `[1]`
instead of paraphrasing vaguely, and *you* can check the citation. Ungrounded claims become
visible rather than plausible.

## Ordering: lost in the middle

Models attend unevenly across a long context. Accuracy is highest for material at the very
beginning and the very end, and sags in the middle — the "lost in the middle" effect. A fact
buried at position 5 of 10 is measurably more likely to be missed than the same fact at
position 1.

Retrievers return results in descending relevance, so naive assembly puts your best chunk first
(good) and your second-best in the middle (bad). The fix is to **reorder so the strongest chunks
sit at both ends**, weakest in the middle. LangChain ships `LongContextReorder` for exactly this.

The effect is small at 4 chunks and significant at 15+. Do it anyway; it costs nothing.

## Budgeting: count tokens, not chunks

`k=5` is not a budget. Five chunks might be 400 tokens or 8,000 depending on what got retrieved.
A budget is expressed in tokens, enforced before the call:

1. Decide the context allowance (model limit, minus prompt, minus expected answer, minus safety
   margin).
2. Walk retrieved chunks in relevance order, adding while they fit.
3. Stop when the next one would overflow.

Fitting more into a large context window is not automatically better. More context costs more,
adds latency, and dilutes attention. Retrieving 20 chunks where 4 would do makes answers *worse*
as well as slower.

## Instructions that actually change behaviour

Three that earn their tokens:

**Ground strictly.** "Answer using only the context below." Without it the model blends
retrieved text with its pretrained priors, and you cannot tell which produced any given
sentence.

**Permit refusal explicitly.** "If the context does not contain the answer, say you don't know."
Models are strongly biased toward being helpful, and helpfulness under missing information looks
exactly like hallucination. You have to make refusal an allowed, named outcome.

**Require citations.** "Cite the chunk number for each fact." This is verification you get for
free — and it also improves grounding, since the model has to locate a supporting passage before
asserting something.

Set `temperature=0`. Creativity is not the goal; faithfully reading supplied text is.

## System vs user message

Put durable instructions (role, grounding rules, citation format, refusal policy) in the
**system** message. Put the context and question in the **user** message.

The separation is not cosmetic: it keeps stable instructions in a stable prefix — which is what
makes prompt caching effective — and it makes the boundary between *your* instructions and
*retrieved text* explicit. Retrieved content is data, not instruction. A document containing
"ignore previous instructions" should not be read as a command, and keeping instructions in a
different message is the first line of defence.

## Run it

```bash
python modules/05-prompt-construction/prompting.py
```

Five prompt variants against the same retrieved chunks, so the only variable is the prompt.

## What to look for

- **The bare prompt answers the unanswerable question anyway.** The strict prompt refuses. Same
  retrieval, same model — the difference is one instruction.
- **Citations make errors checkable.** Trace a `[2]` back to the printed chunk.
- **Token counts per variant.** The structured prompt costs more; decide if it is worth it.
- **The reordered context puts the top chunk first and second-best last.**

## Exercises

1. Remove "say you don't know" and ask about the CFO. Count how many sentences of plausible
   fiction you get.
2. Set the token budget to 300 and ask a question needing several chunks. Does it degrade
   gracefully or fail outright?
3. Add a document to the corpus that contradicts the Atlas spec on payload, retrieve both, and
   see how the model handles conflict. Does it notice, pick one silently, or average them?

## Next

Tier 1 is complete — you can build a working RAG system end to end. Tier 2 attacks the weakest
link: retrieval quality. Module 06 starts by questioning the assumption that the user's question
is the right thing to search with.
