# %% [markdown]
# # Module 03 — Embeddings & Similarity
#
# What the vectors look like, how "close" is measured, and where the abstraction leaks.
# Read `README.md` in this folder first.

# %%
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def find_corpus() -> Path:
    try:
        start = Path(__file__).resolve().parent
    except NameError:
        start = Path.cwd()
    for candidate in [start, *start.parents]:
        corpus = candidate / "data" / "corpus"
        if corpus.is_dir():
            return corpus
    raise FileNotFoundError("Could not find data/corpus above " + str(start))


CORPUS = find_corpus()

# %% [markdown]
# ## What a vector actually looks like

# %%
vector = embeddings.embed_query("The Atlas R5 has a maximum payload of 450 kg.")
arr = np.array(vector)

print(f"Dimensions: {len(vector)}")
print(f"First 8 values: {np.round(arr[:8], 4)}")
print(f"L2 norm: {np.linalg.norm(arr):.6f}   <- OpenAI returns unit-length vectors")

# %% [markdown]
# ## The three similarity metrics
#
# Implemented directly so there is no mystery about what the vector store is doing.

# %%
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def embed(texts: list[str]) -> np.ndarray:
    return np.array(embeddings.embed_documents(texts))


# %% [markdown]
# ## Semantic similarity vs word overlap
#
# The pairs below are chosen to separate the two. A model doing lexical matching would score
# pair 1 low and pair 4 high; a model doing semantic matching does the opposite.

# %%
pairs = [
    ("How long does the battery last?", "What is the runtime on a full charge?"),
    ("Atlas R5 payload capacity", "How much weight can the robot carry?"),
    ("The robot won't charge", "Error E-311 charge fault"),
    ("Atlas R5 maximum payload", "Atlas R5 maximum speed"),
    ("Warranty coverage period", "Floor flatness requirements"),
    ("E-204 localisation lost", "E-311 charge fault"),
]

print(f"\n{'cosine':>8}  {'dot':>8}  {'L2':>8}   pair")
print("-" * 78)
for left, right in pairs:
    a, b = embed([left, right])
    print(f"{cosine(a, b):>8.4f}  {dot(a, b):>8.4f}  {euclidean(a, b):>8.4f}   "
          f"{left!r} / {right!r}")

# %% [markdown]
# Two things to take from that table:
#
# - Pairs 1-3 score high on **meaning** with little or no shared vocabulary. That is the whole
#   value proposition of vector retrieval.
# - Pair 6 also scores high — but `E-204` and `E-311` are unrelated faults. The embedding model
#   sees two near-identical short strings and cannot tell them apart. Exact identifiers are a
#   known blind spot; Module 07 puts keyword search back alongside vectors to cover it.
# - `cosine` and `dot` are identical here because the vectors are unit length. `L2` runs the
#   other direction: lower means closer.

# %% [markdown]
# ## Ranking is what matters, not the score
#
# Different metrics produce different numbers but — on normalised vectors — the same order.
# Below, one query ranked against every chunk by each metric.

# %%
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
docs = [
    Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})
    for p in sorted(CORPUS.glob("*.md"))
]
chunks = splitter.split_documents(docs)
chunk_vectors = embed([c.page_content for c in chunks])

query = "How long does it take to charge an Atlas robot?"
query_vector = np.array(embeddings.embed_query(query))

scores = {
    "cosine": [cosine(query_vector, v) for v in chunk_vectors],
    "dot": [dot(query_vector, v) for v in chunk_vectors],
    "euclidean": [-euclidean(query_vector, v) for v in chunk_vectors],  # negate: higher = closer
}

print(f"\nQuery: {query}\n")
for metric, values in scores.items():
    top = np.argsort(values)[::-1][:3]
    print(f"{metric:>10}: chunks {list(top)}  <- same ranking, different scale")

print("\nTop chunk:")
best = int(np.argmax(scores["cosine"]))
print(f"  [{chunks[best].metadata['source']}] "
      f"{' '.join(chunks[best].page_content.split())[:200]}...")

# %% [markdown]
# ## Telling Chroma which metric to use
#
# Chroma defaults to L2. For OpenAI vectors the ranking is identical either way, but being
# explicit is worth the one line — and it matters immediately if you swap in a model that does
# not normalise.

# %%
store = Chroma.from_documents(
    chunks,
    embeddings,
    collection_name="m03-cosine",
    collection_metadata={"hnsw:space": "cosine"},
)

for doc, score in store.similarity_search_with_score(query, k=3):
    print(f"  distance={score:.4f}  [{doc.metadata['source']}] "
          f"{' '.join(doc.page_content.split())[:90]}...")

print("\nNote: Chroma reports *distance* (lower = better), not similarity. "
      "Getting this backwards silently inverts your ranking.")
store.delete_collection()

# %% [markdown]
# ## Matryoshka: shorter vectors, most of the quality
#
# The v3 models are trained so the leading dimensions carry the most signal. Truncating is a
# direct storage/speed lever — and here it costs almost nothing in ranking quality.

# %%
print(f"\n{'dims':>6}  {'top chunk':>10}  {'cosine':>8}")
print("-" * 30)
for dims in [1536, 512, 256, 64]:
    small = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=dims)
    q = np.array(small.embed_query(query))
    vecs = np.array(small.embed_documents([c.page_content for c in chunks]))
    sims = [cosine(q, v) for v in vecs]
    winner = int(np.argmax(sims))
    print(f"{dims:>6}  {winner:>10}  {max(sims):>8.4f}")

print("\nIf the winning chunk index stays the same as dimensions fall, you can store a "
      "fraction of the floats for the same retrieval quality.")
