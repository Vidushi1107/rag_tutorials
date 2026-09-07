# %% [markdown]
# # Module 08 — Reranking
#
# Retrieve 25 candidates cheaply, then reorder them with something that actually reads them.
# Read `README.md` in this folder first.
#
# Optional extra for the local cross-encoder section: `pip install sentence-transformers`

# %%
from pathlib import Path

from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

CANDIDATES = 20
FINAL_K = 4


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

store = Chroma.from_documents(chunks, embeddings, collection_name="m08")
dense = store.as_retriever(search_kwargs={"k": CANDIDATES})
sparse = BM25Retriever.from_documents(chunks)
sparse.k = CANDIDATES

# Wide hybrid first stage: maximise the chance the right chunk is in the pool at all.
first_stage = EnsembleRetriever(retrievers=[dense, sparse], weights=[0.5, 0.5])

QUESTION = "What do I do if an Atlas robot docks but doesn't charge?"


def preview(chunk: Document, width: int = 66) -> str:
    return f"[{chunk.metadata['source']:<24}] {' '.join(chunk.page_content.split())[:width]}..."


# %% [markdown]
# ## Stage 1: retrieve wide
#
# Deliberately over-fetch. These are candidates, not answers — the ordering here is not
# trustworthy, which is the whole reason stage 2 exists.

# %%
candidates = first_stage.invoke(QUESTION)

print("=" * 78)
print(f"QUESTION: {QUESTION}")
print("=" * 78)
print(f"\nSTAGE 1 — hybrid retrieval returned {len(candidates)} candidates:\n")
for i, chunk in enumerate(candidates[:10], 1):
    print(f"  {i:>2}. {preview(chunk)}")
if len(candidates) > 10:
    print(f"      ... and {len(candidates) - 10} more")

# %% [markdown]
# ## Stage 2: LLM reranking
#
# Score each candidate against the query on a 0-10 scale, then keep the best few. Two details
# that matter: `temperature=0` for stable scores, and a strict output format so parsing does not
# become the weak link.

# %%
rerank_prompt = ChatPromptTemplate.from_template(
    """Rate how well this passage answers the question, on a scale of 0 to 10.

10 = directly and completely answers it
5  = related topic, does not answer it
0  = irrelevant

Respond with a single integer and nothing else.

Question: {question}

Passage:
{passage}

Score:"""
)
scorer = rerank_prompt | llm | StrOutputParser()


def llm_rerank(question: str, docs: list[Document], top_k: int) -> list[tuple[Document, float]]:
    scored = []
    for doc in docs:
        raw = scorer.invoke({"question": question, "passage": doc.page_content})
        try:
            score = float(raw.strip().split()[0])
        except (ValueError, IndexError):
            score = 0.0
        scored.append((doc, score))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


reranked = llm_rerank(QUESTION, candidates, top_k=FINAL_K)

print(f"\nSTAGE 2 — reranked to top {FINAL_K}:\n")
for i, (chunk, score) in enumerate(reranked, 1):
    print(f"  {i}. score={score:>4.1f}  {preview(chunk)}")

# %% [markdown]
# ## Did the order actually change?
#
# If reranking returns the same top 4 the bi-encoder already had, it bought you nothing. Usually
# it does not.

# %%
before = [preview(c, 40) for c in candidates[:FINAL_K]]
after = [preview(c, 40) for c, _ in reranked]

print("\n" + "=" * 78)
print("RANK MOVEMENT")
print("=" * 78)
print(f"\n{'position':<10} {'stage 1 (hybrid)':<46} stage 2 (reranked)")
for i, (b, a) in enumerate(zip(before, after), 1):
    marker = "  " if b == a else "->"
    print(f"{i:<10} {b:<46} {marker} {a}")

moved = sum(1 for b, a in zip(before, after) if b != a)
print(f"\n{moved} of {FINAL_K} positions changed.")

# %% [markdown]
# ## Thresholding: the retriever can now say "nothing here"
#
# Similarity search always returns `k` results, however bad. Relevance *scores* let you drop
# everything below a bar — so an unanswerable question yields zero chunks rather than four
# irrelevant ones for the LLM to work with.

# %%
UNANSWERABLE = "What is the CEO's home address?"

print("\n" + "=" * 78)
print(f"THRESHOLDING: {UNANSWERABLE!r}")
print("=" * 78)

unanswerable_candidates = first_stage.invoke(UNANSWERABLE)
unanswerable_scored = llm_rerank(UNANSWERABLE, unanswerable_candidates, top_k=5)

THRESHOLD = 6.0
print(f"\nTop scores (threshold = {THRESHOLD}):")
for chunk, score in unanswerable_scored:
    verdict = "keep" if score >= THRESHOLD else "drop"
    print(f"  score={score:>4.1f}  [{verdict}]  {preview(chunk, 50)}")

kept = [c for c, s in unanswerable_scored if s >= THRESHOLD]
print(f"\n{len(kept)} chunks passed the threshold.")
if not kept:
    print("Retrieval itself concluded the corpus does not cover this — "
          "no prompt engineering required.")

# %% [markdown]
# ## Local cross-encoder
#
# A purpose-built reranker: smaller, faster, and cheaper per candidate than an LLM. This is what
# you would actually deploy. Skipped automatically if the package is not installed.

# %%
try:
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=CrossEncoderReranker(model=cross_encoder, top_n=FINAL_K),
        base_retriever=first_stage,
    )

    print("\n" + "=" * 78)
    print("LOCAL CROSS-ENCODER (ms-marco-MiniLM-L-6-v2)")
    print("=" * 78 + "\n")
    for i, chunk in enumerate(compression_retriever.invoke(QUESTION), 1):
        print(f"  {i}. {preview(chunk)}")
except ImportError:
    print("\n[skipped] local cross-encoder needs: pip install sentence-transformers")

# %% [markdown]
# ## End to end: does the answer improve?
#
# Same question, same corpus. One pipeline passes the bi-encoder's top 4 straight through; the
# other passes the reranked top 4.

# %%
answer_prompt = ChatPromptTemplate.from_template(
    """Answer using only the context. Cite the source document for each fact.
If the context does not answer the question, say so.

Context:
{context}

Question: {question}
Answer:"""
)
answer_chain = answer_prompt | llm | StrOutputParser()


def as_context(docs: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)


print("\n" + "=" * 78)
print("ANSWER COMPARISON")
print("=" * 78)

print("\n--- no reranking (hybrid top 4) ---")
print(answer_chain.invoke(
    {"context": as_context(candidates[:FINAL_K]), "question": QUESTION}
))

print("\n--- with reranking (top 4 of 20 candidates) ---")
print(answer_chain.invoke(
    {"context": as_context([c for c, _ in reranked]), "question": QUESTION}
))

store.delete_collection()
