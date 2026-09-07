# RAG Tutorials

A module-by-module course on Retrieval-Augmented Generation. Each module teaches **one**
concept, with a short written explanation and a runnable script you can read top to bottom.

Built with Python + LangChain, using OpenAI for both chat and embeddings, and Chroma as the
local vector store (no database to install, no extra API keys).

## Who this is for

You can write Python and have used an LLM API at least once. You do not need prior RAG,
vector search, or LangChain experience — the modules build up from nothing.

## Setup

```bash
git clone https://github.com/shantam21/rag_tutorials.git
cd rag_tutorials
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then paste your OpenAI key into .env
```

Every module reads `OPENAI_API_KEY` from `.env`. Running the full Tier 1 set costs well under
$0.10 in API usage.

## How to run a module

Each module is a single script with `# %%` cell markers. That means you can either:

```bash
python modules/01-naive-rag-pipeline/pipeline.py
```

...or open the same file in VS Code / Jupyter, where the markers render it as a notebook so you
can run it cell by cell. Reading the module README first is recommended — the code assumes you
know why each step exists.

## The sample corpus

All modules retrieve over `data/corpus/`: six documents about **Helix Robotics**, a fictional
Dutch warehouse-robotics company.

The company is fictional on purpose. If you ask an LLM "what is the Atlas R5's payload?" with
no retrieval, it cannot know — so any correct answer is provably coming from your retrieval
pipeline and not from the model's memory. That makes it obvious when RAG is working, and
equally obvious when it silently is not.

## Modules

### Tier 1 — Foundations

| # | Module | Concept |
| --- | --- | --- |
| 01 | [Naive RAG Pipeline](modules/01-naive-rag-pipeline/) | The five stages of RAG, end to end |
| 02 | [Chunking Strategies](modules/02-chunking-strategies/) | How you split documents decides what you can retrieve |
| 03 | [Embeddings & Similarity](modules/03-embeddings-and-similarity/) | What vectors capture, and how "closeness" is measured |
| 04 | [Vector Stores & Indexing](modules/04-vector-stores-and-indexing/) | Storing, filtering, and searching vectors at scale |
| 05 | [Prompt Construction](modules/05-prompt-construction/) | Assembling context, budgeting tokens, forcing citations |

### Tier 2 — Intermediate Retrieval

| # | Module | Concept |
| --- | --- | --- |
| 06 | [Query Transformation](modules/06-query-transformation/) | Rewriting the question to retrieve better |
| 07 | [Hybrid Search](modules/07-hybrid-search/) | Combining keyword (BM25) and vector retrieval |
| 08 | [Reranking](modules/08-reranking/) | A second, slower pass that fixes ranking errors |
| 09 | [Evaluating RAG](modules/09-evaluating-rag/) | Measuring faithfulness, relevancy, and recall |

### Tier 3 — Advanced & Production

| # | Module | Concept |
| --- | --- | --- |
| 10 | Advanced Retrievers | Parent-document, sentence-window, self-querying |
| 11 | Conversational RAG | Multi-turn history and query condensation |
| 12 | Agentic RAG | Letting the model decide when and what to retrieve |
| 13 | Multi-modal RAG | Tables, images, and real-world PDF layouts |
| 14 | Production Readiness | Caching, cost, latency, guardrails, observability |

Modules are self-contained: each one builds the pipeline it needs, so you can jump straight to
module 08 without having run 01-07. Working in order is still the intended path.
