# %% [markdown]
# # Module 04 — Vector Stores & Indexing
#
# Persistence, metadata filtering, updates, and diversity-aware retrieval.
# Read `README.md` in this folder first.

# %%
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def find_root() -> Path:
    try:
        start = Path(__file__).resolve().parent
    except NameError:
        start = Path.cwd()
    for candidate in [start, *start.parents]:
        if (candidate / "data" / "corpus").is_dir():
            return candidate
    raise FileNotFoundError("Could not find repo root above " + str(start))


ROOT = find_root()
CORPUS = ROOT / "data" / "corpus"
PERSIST_DIR = ROOT / "chroma_db"

# %% [markdown]
# ## Build a persistent index
#
# The key detail: indexing is separate from querying. Pass `persist_directory` and the index
# survives the process. Re-run this script and the second run skips embedding entirely.
#
# Stable IDs (`source:chunk_index`) let us update and delete individual chunks later. Without
# them your only option is rebuilding the whole collection.

# %%
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)

docs = [
    Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})
    for p in sorted(CORPUS.glob("*.md"))
]

chunks, ids = [], []
for doc in docs:
    for i, chunk in enumerate(splitter.split_documents([doc])):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["doc_type"] = (
            "product" if chunk.metadata["source"].startswith("product") else "policy"
        )
        chunks.append(chunk)
        ids.append(f"{chunk.metadata['source']}:{i}")

store = Chroma(
    collection_name="helix",
    embedding_function=embeddings,
    persist_directory=str(PERSIST_DIR),
    collection_metadata={"hnsw:space": "cosine"},
)

existing = store.get()["ids"]
if existing:
    print(f"Loaded existing index: {len(existing)} chunks (no embedding calls made)")
else:
    store.add_documents(chunks, ids=ids)
    print(f"Built new index: {len(chunks)} chunks -> {PERSIST_DIR}")

# %% [markdown]
# ## The ambiguity problem
#
# "What is the maximum payload?" is a reasonable question with two different correct answers in
# this corpus — 450 kg for Atlas, 12 kg for Scout. Unfiltered search has to guess.

# %%
query = "What is the maximum payload?"

print(f"\nQ: {query}")
print("\nUnfiltered:")
for doc in store.similarity_search(query, k=3):
    print(f"  [{doc.metadata['source']}] {' '.join(doc.page_content.split())[:80]}...")

print("\nFiltered to the Atlas spec:")
for doc in store.similarity_search(query, k=3, filter={"source": "product-atlas-r5.md"}):
    print(f"  [{doc.metadata['source']}] {' '.join(doc.page_content.split())[:80]}...")

# %% [markdown]
# Filtering turned an ambiguous question into a precise one without touching the embedding
# model or the prompt. Chroma also supports operators — `$eq`, `$ne`, `$in`, `$gte`, and
# `$and` / `$or` for composition.

# %%
print("\nPolicy documents only ($ne on doc_type):")
for doc in store.similarity_search(
    "What is not covered?", k=3, filter={"doc_type": {"$ne": "product"}}
):
    print(f"  [{doc.metadata['source']}] {' '.join(doc.page_content.split())[:80]}...")

# %% [markdown]
# ## MMR: relevance without redundancy
#
# Similarity search will happily hand back four chunks that say the same thing. MMR fetches a
# wider pool (`fetch_k`) then greedily picks results that are relevant *and* dissimilar to what
# it has already chosen.

# %%
charging_query = "charging and docking"

print(f"\nQ: {charging_query}")
print("\nPlain similarity (k=4):")
for doc in store.similarity_search(charging_query, k=4):
    print(f"  [{doc.metadata['source']:<24}] {' '.join(doc.page_content.split())[:70]}...")

print("\nMMR (k=4, fetch_k=20, lambda_mult=0.5):")
for doc in store.max_marginal_relevance_search(
    charging_query, k=4, fetch_k=20, lambda_mult=0.5
):
    print(f"  [{doc.metadata['source']:<24}] {' '.join(doc.page_content.split())[:70]}...")

# %% [markdown]
# ## Updating and deleting by ID
#
# Documents change. With stable IDs you can correct one chunk in place — no rebuild, no
# duplicate. Upserting the same ID overwrites rather than appending, which is what you want
# for a re-ingestion pipeline.

# %%
target_id = "product-atlas-r5.md:0"
before = store.get(ids=[target_id])["documents"][0]
print(f"\nBefore: {' '.join(before.split())[:100]}...")

store.update_document(
    target_id,
    Document(
        page_content="PRICE UPDATE 2026: The Atlas R5 base unit is now priced at 82,000 EUR.",
        metadata={"source": "product-atlas-r5.md", "chunk_index": 0, "doc_type": "product"},
    ),
)

print("After:", store.get(ids=[target_id])["documents"][0])
print("\nSearching for the new content:")
for doc in store.similarity_search("Atlas R5 2026 price", k=1):
    print(f"  {doc.page_content}")

# restore, so re-running this script starts from a clean state
store.update_document(
    target_id,
    Document(
        page_content=before,
        metadata={"source": "product-atlas-r5.md", "chunk_index": 0, "doc_type": "product"},
    ),
)
print("\n(restored original chunk)")

# %% [markdown]
# ## Is the index earning its keep?
#
# At this corpus size, no — and that is worth seeing. ANN indexing pays off at scale; below
# ~10k vectors a flat scan is competitive and gives you perfect recall for free.

# %%
stored = store.get(include=["embeddings"])
vectors = np.array(stored["embeddings"])
query_vector = np.array(embeddings.embed_query(charging_query))

start = time.perf_counter()
exact_top = np.argsort(vectors @ query_vector)[::-1][:4]
exact_ms = (time.perf_counter() - start) * 1000

start = time.perf_counter()
store.similarity_search(charging_query, k=4)
hnsw_ms = (time.perf_counter() - start) * 1000

print(f"\n{len(vectors)} vectors, {vectors.shape[1]} dimensions")
print(f"  exact (numpy dot product): {exact_ms:.2f} ms   <- 100% recall")
print(f"  HNSW via Chroma:           {hnsw_ms:.2f} ms   <- includes query embedding call")
print("\nThe HNSW number is dominated by the network round-trip to embed the query. "
      "Index choice only starts to matter when the scan itself is the bottleneck.")
