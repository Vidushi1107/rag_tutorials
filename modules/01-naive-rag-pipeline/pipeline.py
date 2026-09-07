# %% [markdown]
# # Module 01 — The Naive RAG Pipeline
#
# Five stages, end to end: load, split, embed+store, retrieve, generate.
# Read `README.md` in this folder first.

# %%
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4

QUESTION = "What is the maximum payload of the Atlas R5, and how long does it take to charge?"


def find_corpus() -> Path:
    """Locate data/corpus whether this runs as a script or as notebook cells."""
    try:
        start = Path(__file__).resolve().parent
    except NameError:  # notebook cell — no __file__
        start = Path.cwd()
    for candidate in [start, *start.parents]:
        corpus = candidate / "data" / "corpus"
        if corpus.is_dir():
            return corpus
    raise FileNotFoundError("Could not find data/corpus above " + str(start))


# %% [markdown]
# ## Baseline: what the model says with no retrieval
#
# Helix Robotics is fictional, so the model has never seen it. Whatever comes back here is
# either a refusal or an invention — that is the gap RAG closes.

# %%
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print("=" * 70)
print("BASELINE (no retrieval)")
print("=" * 70)
print(llm.invoke(QUESTION).content)

# %% [markdown]
# ## Stage 1 — Load
#
# Read the corpus into `Document` objects. Each carries `page_content` (the text) and
# `metadata` (here, the source path). Metadata is what makes citation possible later.

# %%
loader = DirectoryLoader(
    str(find_corpus()),
    glob="*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
documents = loader.load()

print(f"\nLoaded {len(documents)} documents:")
for doc in documents:
    print(f"  {Path(doc.metadata['source']).name:<28} {len(doc.page_content):>6,} chars")

# %% [markdown]
# ## Stage 2 — Split
#
# Chunks, not documents, are the unit of retrieval. `RecursiveCharacterTextSplitter` tries to
# break on paragraph boundaries first, then sentences, then words — so it damages meaning less
# than a blind character cut. The overlap keeps a sentence that straddles a boundary from being
# lost to both chunks.

# %%
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True,
)
chunks = splitter.split_documents(documents)

print(f"\n{len(documents)} documents -> {len(chunks)} chunks")
print(f"Average chunk: {sum(len(c.page_content) for c in chunks) // len(chunks):,} chars")

# %% [markdown]
# ## Stage 3 — Embed and store
#
# Each chunk becomes a 1,536-dimension vector. Chunks about charging land near other chunks
# about charging, which is what makes semantic search work. This is the only expensive step,
# and it runs once — not per question.

# %%
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)

print(f"\nIndexed {len(chunks)} chunks into Chroma (in memory)")

# %% [markdown]
# ## Stage 4 — Retrieve
#
# The question is embedded with the *same* model, then compared against every stored vector.
# The `k` nearest come back. Note that similarity search always returns `k` results — it has no
# concept of "nothing here is relevant."

# %%
retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
retrieved = retriever.invoke(QUESTION)

print("\n" + "=" * 70)
print(f"RETRIEVED {len(retrieved)} CHUNKS")
print("=" * 70)
for i, chunk in enumerate(retrieved, 1):
    source = Path(chunk.metadata["source"]).name
    preview = " ".join(chunk.page_content.split())[:160]
    print(f"\n[{i}] {source}\n    {preview}...")

# %% [markdown]
# ## Stage 5 — Generate
#
# The retrieved text goes into the prompt as context. Two instructions matter: answer *only*
# from the context, and say so when the context is insufficient. Without them the model happily
# falls back on its own priors, which defeats the point of grounding it.

# %%
prompt = ChatPromptTemplate.from_template(
    """Answer the question using only the context below.
If the context does not contain the answer, say you don't know — do not guess.
Cite the source document for each fact you use.

Context:
{context}

Question: {question}

Answer:"""
)


def format_docs(docs) -> str:
    return "\n\n---\n\n".join(
        f"[source: {Path(d.metadata['source']).name}]\n{d.page_content}" for d in docs
    )


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\n" + "=" * 70)
print("GROUNDED ANSWER (with retrieval)")
print("=" * 70)
print(rag_chain.invoke(QUESTION))

# %% [markdown]
# ## Try it yourself
#
# Swap in your own questions. The third one is a trap — the corpus has no answer, and a naive
# pipeline will still retrieve four chunks and hand them over regardless.

# %%
for question in [
    "Can an Atlas robot charge on a Beacon D1 dock?",
    "When does the warranty period start?",
    "Who is the CFO of Helix Robotics?",
]:
    print("\n" + "-" * 70)
    print(f"Q: {question}")
    print(f"A: {rag_chain.invoke(question)}")
