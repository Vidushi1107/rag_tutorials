# Module 02 — Chunking Strategies

**Concept:** The chunk is the unit of retrieval. You can only ever retrieve a whole chunk, so
how you cut your documents sets a hard ceiling on what your system can answer.

## Why this is not a minor detail

Module 01 used `RecursiveCharacterTextSplitter(chunk_size=800)` without justifying it. That one
line has more effect on answer quality than your choice of LLM.

Consider the Atlas R5 spec table:

```
| Maximum payload | 450 kg |
| Unloaded weight | 210 kg |
```

Cut between those rows and you get a chunk containing `450 kg` with no indication of what is
450 kg, and another containing `Maximum payload` with no number. Both chunks are individually
useless, and no amount of prompt engineering downstream recovers the lost association.

## The size trade-off

There is no correct chunk size, only a trade-off you are choosing between:

| | Small chunks (~200 chars) | Large chunks (~2000 chars) |
| --- | --- | --- |
| Embedding precision | High — vector represents one idea | Low — vector is an average of many ideas |
| Context for the LLM | Poor — facts arrive stranded | Good — surrounding explanation included |
| Retrieval hit rate | Misses facts that span a boundary | Catches more, but buries it in noise |
| Cost per query | Lower | Higher |

The failure modes are different, which is why you cannot tune this by feel. Small chunks *lose*
answers. Large chunks *contain* answers the model then fails to notice — the "lost in the
middle" effect, where models attend less to the centre of a long context.

**Overlap** softens boundary loss: each chunk repeats the last N characters of its predecessor,
so a fact split across a boundary survives whole in one of them. Typical overlap is 10-20% of
chunk size. It is a hedge, not a fix — it costs storage and retrieval redundancy.

## Five strategies

**1. Fixed-size character** (`CharacterTextSplitter`) — cut every N characters. Fast, and
respects nothing. Splits mid-word, mid-sentence, mid-table. Use it as a baseline to beat.

**2. Recursive character** (`RecursiveCharacterTextSplitter`) — try to split on `\n\n`, then
`\n`, then `. `, then ` `, falling back only when a piece is still too big. Keeps paragraphs
intact when it can. **This is the right default** for most prose.

**3. Token-based** — split on tokenizer tokens rather than characters. Characters are a proxy
for what you actually care about (context window and cost), and the proxy is bad: code and
tables run ~2 characters per token while English prose runs ~4. If you are tight against a
context limit, count the thing you are actually limited by.

**4. Structure-aware** (`MarkdownHeaderTextSplitter`, and equivalents for HTML/code) — split on
the document's own structure, and promote headers into metadata. A chunk from the Atlas spec
then carries `{"Header 2": "Power"}`, which is both better context and a filterable field. For
documents with real structure this usually beats generic splitting outright.

**5. Semantic** (`SemanticChunker`) — embed sentences, then cut where consecutive sentences
diverge in meaning. Chunks end up topically coherent regardless of length. It costs an embedding
pass over the whole corpus to build, and the quality gain over structure-aware splitting is
often small. Reach for it on unstructured text — transcripts, scraped prose — where there is no
structure to exploit.

## How to actually choose

1. Start with recursive, 800-1000 chars, 10-15% overlap.
2. If your documents have headers, switch to structure-aware and keep the headers as metadata.
3. Only then tune size — and measure it, do not eyeball it. Module 09 covers how.

## Run it

```bash
pip install langchain-experimental       # only needed for the semantic chunker section
python modules/02-chunking-strategies/chunking.py
```

The script splits the same document five ways, prints the boundaries each strategy produces,
then builds a separate index per strategy and runs identical questions through each so you can
compare what comes back.

## What to look for

- **The fixed-size splitter cuts the spec table mid-row.** Find it in the printed output. That
  chunk can never answer a payload question.
- **The markdown splitter's chunks carry header metadata.** Look at `chunk.metadata`.
- **The table question fails on some strategies and works on others.** Same corpus, same
  embedding model, same LLM — the only variable is the cut.

## Exercises

1. Drop overlap to 0 and re-run. Which questions break?
2. The `deployment-guide.md` numbered steps are semantically one unit. Which strategy keeps
   them together?
3. Try chunk_size=150 with the markdown splitter. Structure-aware splitting still has to obey a
   size cap — what happens to a section longer than the cap?

## Next

Module 03 opens up the step this module took for granted: what an embedding actually is, and
what "nearest" means when we search.
