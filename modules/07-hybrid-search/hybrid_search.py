# %% [markdown]
# # Module 07 — Hybrid Search
#
# BM25 and vector search fail on opposite queries. Fuse them with RRF.
# Read `README.md` in this folder first.
#
# Needs one extra package: `pip install rank_bm25`

# %%
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
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


splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
docs = [
    Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})
    for p in sorted(find_corpus().glob("*.md"))
]
chunks = splitter.split_documents(docs)

# Both retrievers index the *same* chunks — the only difference is how they match.
store = Chroma.from_documents(chunks, embeddings, collection_name="m07")
dense = store.as_retriever(search_kwargs={"k": 4})

sparse = BM25Retriever.from_documents(chunks)
sparse.k = 4

print(f"Indexed {len(chunks)} chunks into both a vector store and a BM25 index")


def preview(chunk: Document, width: int = 68) -> str:
    return f"[{chunk.metadata['source']:<24}] {' '.join(chunk.page_content.split())[:width]}..."


def compare(query: str, expect: str) -> None:
    print("\n" + "=" * 76)
    print(f"QUERY: {query!r}")
    print(f"looking for: {expect}")
    print("=" * 76)
    print("\n  DENSE (vectors):")
    for chunk in dense.invoke(query):
        print(f"    {preview(chunk)}")
    print("\n  SPARSE (BM25):")
    for chunk in sparse.invoke(query):
        print(f"    {preview(chunk)}")


# %% [markdown]
# ## Case 1: an exact identifier
#
# The Module 03 blind spot, in a retriever. BM25 treats `E-311` as a rare, high-signal token.
# The embedding model sees a short alphanumeric string much like every other error code.

# %%
compare("E-311", expect="the E-311 charge fault entry in support-faq.md")

# %% [markdown]
# ## Case 2: a conversational paraphrase
#
# Now the mirror image. The E-311 entry never contains the words "won't" or "broken", so BM25
# has nothing to match on. The embedding has no such problem.

# %%
compare("my robot won't charge when I dock it", expect="the same E-311 entry, found by meaning")

# %% [markdown]
# ## Why you cannot just average the scores
#
# Look at the ranges before assuming a weighted sum is safe.

# %%
scored = store.similarity_search_with_score("E-311", k=4)
print("\nChroma distances (cosine distance, lower = closer):")
for doc, score in scored:
    print(f"  {score:.4f}  {preview(doc, 50)}")

bm25_raw = sparse.vectorizer.get_scores("E-311".split())
print(f"\nBM25 raw scores across all {len(bm25_raw)} chunks:")
print(f"  min={bm25_raw.min():.4f}  max={bm25_raw.max():.4f}  mean={bm25_raw.mean():.4f}")
print("\nOne is bounded [0,2], the other is unbounded and corpus-dependent. "
      "Summing them lets BM25's scale decide everything.")

# %% [markdown]
# ## Reciprocal Rank Fusion, implemented
#
# RRF throws the scores away and keeps only positions, which is exactly why it needs no
# normalisation. Written out here so there is no magic in it.

# %%
def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]], k: int = 60
) -> list[tuple[Document, float]]:
    scores: dict[str, float] = defaultdict(float)
    by_key: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = f"{doc.metadata['source']}::{doc.page_content[:60]}"
            scores[key] += 1.0 / (k + rank + 1)
            by_key[key] = doc

    return sorted(
        ((by_key[key], score) for key, score in scores.items()),
        key=lambda pair: pair[1],
        reverse=True,
    )


for query in ["E-311", "my robot won't charge when I dock it"]:
    print("\n" + "=" * 76)
    print(f"RRF FUSION: {query!r}")
    print("=" * 76)
    fused = reciprocal_rank_fusion([dense.invoke(query), sparse.invoke(query)])
    for doc, score in fused[:4]:
        print(f"  {score:.5f}  {preview(doc)}")

# %% [markdown]
# ## The same thing via LangChain
#
# `EnsembleRetriever` does RRF internally. The `weights` scale each list's contribution — equal
# weights are the sane default until evaluation tells you otherwise.

# %%
hybrid = EnsembleRetriever(retrievers=[dense, sparse], weights=[0.5, 0.5])

for query in ["E-311", "my robot won't charge when I dock it", "battery warranty period"]:
    print("\n" + "-" * 76)
    print(f"HYBRID: {query!r}")
    for chunk in hybrid.invoke(query)[:4]:
        print(f"  {preview(chunk)}")

# %% [markdown]
# ## Weighting is a real lever
#
# Same query, three weightings. A corpus full of part numbers wants more BM25; a corpus of
# conversational prose wants more vector. There is no universal answer — measure it (Module 09).

# %%
print("\n" + "=" * 76)
print("WEIGHT SWEEP on 'E-311 charge fault dock'")
print("=" * 76)
for dense_w, sparse_w in [(0.9, 0.1), (0.5, 0.5), (0.1, 0.9)]:
    tuned = EnsembleRetriever(retrievers=[dense, sparse], weights=[dense_w, sparse_w])
    top = tuned.invoke("E-311 charge fault dock")[0]
    print(f"\n  dense={dense_w} sparse={sparse_w} -> {preview(top)}")

store.delete_collection()
