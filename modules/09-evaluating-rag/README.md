# Module 09 — Evaluating RAG

**Concept:** Every previous module gave you a dial to turn. Without measurement you are guessing,
and "it seems better" does not survive contact with a real corpus.

## Evaluate the two halves separately

This is the most important idea in the module. RAG has two failure modes and they need different
fixes:

1. **Retrieval failed** — the right chunk never arrived. No prompt can recover it.
2. **Generation failed** — the right chunk arrived and the model ignored, misread, or
   contradicted it.

A single end-to-end "is the answer good" score cannot tell these apart, so it cannot tell you
what to fix. Measure them separately, always.

## Retrieval metrics

These need only a list of which documents *should* be retrieved. No LLM, no cost, fast enough to
run on every commit.

**Hit rate @ k** — did any correct document appear in the top k? The blunt one. Answers "is the
information even reaching the model?"

**Recall @ k** — what fraction of all relevant documents were retrieved? Matters for questions
needing several sources; hit rate would call it a success when you found one of three.

**MRR (Mean Reciprocal Rank)** — 1/rank of the first correct result, averaged. Rewards putting
the right chunk *first*, not merely somewhere. This is the metric that moves when you add
reranking.

**Precision @ k** — what fraction of retrieved chunks were relevant? Low precision wastes context
and dilutes attention.

Hit rate and MRR together cover most needs: "did we find it" and "did we rank it well."

## Generation metrics

These need an LLM to judge, so they cost money and time.

**Faithfulness** — is every claim in the answer supported by the retrieved context? This is the
hallucination metric, and the one that matters most. Critically, it is measured *against the
retrieved context*, not against the truth: an answer can be perfectly faithful and still wrong,
if retrieval supplied the wrong passage. That distinction is what makes the metric diagnostic.

**Answer relevancy** — does it address the question actually asked? Catches answers that are
faithful and true but off-topic.

**Answer correctness** — does it match the ground-truth answer? Needs a hand-written reference,
which is expensive, and is the closest thing to "is it right."

**Context precision / recall** — a middle layer: was the retrieved context useful, and was it
complete? Bridges the two halves.

## The golden dataset

You need questions with known answers. There is no way around it.

`data/eval/golden_set.json` holds 14 hand-written examples covering single-fact lookups, exact
identifiers, multi-document synthesis, inference, and — critically — **two unanswerable
questions**. Include unanswerables. A system that scores 100% on answerable questions while
confidently inventing answers to the rest is not working, and only these cases expose it.

**Synthetic generation** scales this: prompt an LLM to write questions from each chunk, keeping
the chunk as the ground-truth source. Cheap, and biased — the questions come out phrased like the
documents, which is exactly the easy case. Use synthetic data for breadth and hand-written data
for the cases you actually care about.

Fifty good examples beat five hundred careless ones. Start at twenty.

## LLM-as-judge: use it, but know the failure modes

Grading with an LLM is the only practical way to score faithfulness at volume. It is also
biased: judges prefer longer answers, prefer their own model family's output, and are sensitive
to option order in comparisons.

Mitigations that actually help: ask for a binary or small-integer verdict rather than a
continuous score, require the reason *before* the verdict, set `temperature=0`, and spot-check
against human judgement on a sample. Never treat judge scores as ground truth — treat them as a
consistent yardstick for comparing two versions of your own system.

## RAGAS

[RAGAS](https://docs.ragas.io) packages these metrics. It is worth using once you are past
prototyping.

This module implements the metrics by hand instead, because the formulas are simple and knowing
what "faithfulness = 0.8" is actually counting matters more than importing it.

```bash
pip install ragas    # optional
```

## Run it

```bash
python modules/09-evaluating-rag/evaluation.py
```

It evaluates three configurations — naive dense, hybrid, and hybrid + reranking — across the
golden set, and prints a comparison table. Expect a couple of minutes and a few cents.

## What to look for

- **Retrieval metrics rank the configurations differently than generation metrics.** They are
  measuring different things.
- **MRR improves most from reranking** — that is precisely what reranking does.
- **The unanswerable questions.** Does faithfulness stay high because the system correctly
  refuses, or does it collapse because the system invented something?
- **Per-question-type breakdown.** Exact-identifier questions should show the biggest gain from
  hybrid search — the Module 07 argument, quantified.

## Exercises

1. Add five questions covering a corpus area currently untested. Do the rankings change?
2. Build a synthetic set with an LLM and compare its difficulty against the hand-written one.
   Which is easier, and why should that worry you?
3. Deliberately break retrieval (`k=1`, tiny chunks) and watch which metric moves first. That
   metric is your early warning signal in production.

## Next

Tier 2 is done: retrieval is better and you can prove it. Tier 3 opens with retrieval
architectures that change *what a chunk is*.
