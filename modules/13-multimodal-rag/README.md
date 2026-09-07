# Module 13 — Multi-modal RAG

**Concept:** Real documents are not clean prose. They are PDFs full of tables, charts, diagrams,
and scans. The trick that makes all of it work is the same one from Module 10: index a text
representation, return the original.

## Why plain text extraction fails

Take this table:

```
| Property        | Value  |
| Maximum payload | 450 kg |
| Unloaded weight | 210 kg |
```

Naive PDF text extraction typically yields:

```
Property Value Maximum payload 450 kg Unloaded weight 210 kg
```

Row and column structure is gone. Chunk that mid-way and you get `450 kg Unloaded weight` — a
fragment that reads as though 450 kg *is* the unloaded weight. This is worse than losing the
data, because it is confidently wrong and nothing downstream flags it.

Charts are worse still: extraction yields axis labels and nothing else. The trend line — the
actual content — is invisible to a text pipeline.

## The unifying strategy

For every non-text element:

1. Generate a **text description** of it (LLM, or a vision model for images).
2. **Index the description** — it embeds well, because it is natural language.
3. **Return the original** element — the raw table markdown, or the image itself.

This is Module 10's multi-vector retriever with a different input type. One `MultiVectorRetriever`
can hold text chunks, table summaries, and image descriptions side by side, all resolving to
their original content.

## Tables

Two things matter.

**Keep the structure.** Extract tables as markdown or HTML, never as flattened text. Markdown
survives chunking and models read it well.

**Never split a table across chunks.** A table is atomic. Extract it separately from surrounding
prose so no splitter can cut it.

Then summarise for indexing: "Atlas R5 physical specifications including payload of 450 kg,
weight, dimensions, speed, and IP rating." That sentence embeds far better than a grid of pipes
and numbers, while retrieval still returns the intact table.

## Images

Three approaches, in the order you should consider them:

**1. Vision-model description (recommended).** Send the image to a vision model, get a
description, index the description, return the image. Works with your existing text embedding
model and text vector store — no new infrastructure. Description quality is the limit, so prompt
for the *content* ("what values does this chart show?"), not the appearance.

**2. Multi-modal embeddings (CLIP and similar).** Embed images and text into one shared space and
search across both. Elegant, and weaker in practice for document retrieval: CLIP-family models are
trained on natural photographs and captions, and handle dense charts and diagrams poorly.

**3. Pass images at generation time.** Retrieve by description, then send the actual image to a
vision model alongside the question. Most accurate for "what does this diagram show?", and the
most expensive.

In practice 1 and 3 combine well: retrieve on descriptions, generate on the real image.

## PDFs

The extraction library is the decision that matters most:

| Library | Good for | Trade-off |
| --- | --- | --- |
| `pypdf` | Simple text-only PDFs | No table or layout awareness |
| `pymupdf` | Fast text plus image extraction | Tables still need work |
| `unstructured` | Layout-aware: titles, tables, figures | Heavy install, slow |
| `pdfplumber` | Table extraction specifically | Text-focused otherwise |
| Cloud (AWS Textract, Azure DI) | Scans, forms, handwriting | Cost, data leaves your network |

Scanned PDFs are images and need OCR (`pytesseract`, or a cloud service) before anything else
applies.

Budget real time for this. **Document parsing is usually the single largest source of quality
loss in a production RAG system** — larger than chunking, embedding choice, or reranking. Garbage
extraction cannot be fixed downstream.

## Run it

```bash
python modules/13-multimodal-rag/multimodal_rag.py

# optional — generates a sample chart for the vision section
pip install matplotlib
```

The table sections need no extra packages. The image section generates a chart if matplotlib is
available and skips cleanly otherwise.

## What to look for

- **The flattened table** — read it and try to work out which number goes with which property.
- **The table summary reads like a sentence,** and retrieves on questions the raw grid misses.
- **Retrieval returns the intact markdown table,** not the summary it matched on.
- **The vision description of the chart** contains values that exist nowhere in the file's text.

## Exercises

1. Ask a question answerable only by reading two columns of a table together. Does the summary-
   indexed version handle it?
2. Feed the same chart to a vision model twice with different prompts — one asking for appearance,
   one for data. Which description retrieves better?
3. Take a real PDF you own, extract it with `pypdf`, and count how many tables survive.

## Next

Module 14 closes the course: everything between a working prototype and something you can put in
front of users.
