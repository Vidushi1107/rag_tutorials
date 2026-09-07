# %% [markdown]
# # Module 02 — Chunking Strategies
#
# Five ways to cut the same document, and what each one costs you.
# Read `README.md` in this folder first.
#
# Needs one extra package: `pip install langchain-experimental`

# %%
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

load_dotenv()

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


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

# The Atlas spec sheet is the interesting case: it mixes prose, a markdown table, and
# bullet lists, so every strategy handles it differently.
atlas_text = (CORPUS / "product-atlas-r5.md").read_text(encoding="utf-8")
atlas_doc = Document(page_content=atlas_text, metadata={"source": "product-atlas-r5.md"})

print(f"Source document: {len(atlas_text):,} characters")


def show(strategy: str, chunks: list[Document], limit: int = 3) -> None:
    sizes = [len(c.page_content) for c in chunks]
    print("\n" + "=" * 72)
    print(f"{strategy}  —  {len(chunks)} chunks, "
          f"min {min(sizes)}, max {max(sizes)}, mean {sum(sizes) // len(sizes)}")
    print("=" * 72)
    for i, chunk in enumerate(chunks[:limit], 1):
        text = chunk.page_content
        head = " ".join(text[:110].split())
        tail = " ".join(text[-70:].split())
        print(f"\n[{i}] {len(text)} chars")
        if chunk.metadata:
            print(f"    meta: {chunk.metadata}")
        print(f"    starts: {head}...")
        print(f"    ends:   ...{tail}")


# %% [markdown]
# ## 1. Fixed-size character splitting
#
# The naive baseline. `CharacterTextSplitter` splits on a single separator and enforces the size
# cap bluntly. Watch where it lands inside the spec table.

# %%
fixed = CharacterTextSplitter(
    separator="\n",
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
fixed_chunks = fixed.split_documents([atlas_doc])
show("1. FIXED-SIZE CHARACTER", fixed_chunks)

# %% [markdown]
# ## 2. Recursive character splitting
#
# Tries `\n\n` first, then `\n`, then `. `, then ` `. Falls back to a hard cut only when a piece
# is still oversized, so paragraph boundaries survive whenever they fit.

# %%
recursive = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)
recursive_chunks = recursive.split_documents([atlas_doc])
show("2. RECURSIVE CHARACTER", recursive_chunks)

# %% [markdown]
# ## 3. Token-based splitting
#
# Same recursive logic, but the size cap counts tokens instead of characters. Compare the
# character sizes below against the 200-token cap — the ratio is not constant, because the
# table rows tokenize far denser than the prose.

# %%
token_based = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=200,
    chunk_overlap=25,
)
token_chunks = token_based.split_documents([atlas_doc])
show("3. TOKEN-BASED (200 tokens)", token_chunks)

# %% [markdown]
# ## 4. Structure-aware (markdown headers)
#
# Splits on the document's own headings and lifts them into metadata. Each chunk now knows it
# came from "Power" or "Pricing" — better context for the LLM and a field you can filter on.
#
# Note the two-pass shape: split by header first, then apply a size cap, because a single
# section can still be longer than your budget.

# %%
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "title"), ("##", "section")],
    strip_headers=False,
)
header_chunks = markdown_splitter.split_text(atlas_text)
markdown_chunks = recursive.split_documents(header_chunks)
show("4. STRUCTURE-AWARE (markdown headers)", markdown_chunks)

# %% [markdown]
# ## 5. Semantic chunking
#
# Embeds each sentence and cuts where meaning shifts. No size parameter at all — boundaries come
# from the content. Costs a full embedding pass to build.

# %%
try:
    from langchain_experimental.text_splitter import SemanticChunker

    semantic = SemanticChunker(
        OpenAIEmbeddings(model="text-embedding-3-small"),
        breakpoint_threshold_type="percentile",
    )
    semantic_chunks = semantic.split_documents([atlas_doc])
    show("5. SEMANTIC", semantic_chunks)
except ImportError:
    semantic_chunks = None
    print("\n[skipped] SemanticChunker needs: pip install langchain-experimental")

# %% [markdown]
# ## Does it actually change the answers?
#
# Build one index per strategy over the *whole* corpus, then run identical questions through
# each. Only the chunking differs — same embedding model, same `k`, no LLM involved, so any
# difference you see is caused by the cut alone.
#
# The first question is the table question: it needs "Maximum payload" and "450 kg" to have
# survived in the same chunk.

# %%
all_docs = [
    Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})
    for p in sorted(CORPUS.glob("*.md"))
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

strategies = {
    "fixed": fixed.split_documents(all_docs),
    "recursive": recursive.split_documents(all_docs),
    "token": token_based.split_documents(all_docs),
    "markdown": recursive.split_documents(
        [c for d in all_docs for c in markdown_splitter.split_text(d.page_content)]
    ),
}

questions = [
    "What is the maximum payload of the Atlas R5?",
    "What are the floor flatness requirements for a deployment?",
]

for question in questions:
    print("\n" + "=" * 72)
    print(f"Q: {question}")
    print("=" * 72)
    for name, chunks in strategies.items():
        store = Chroma.from_documents(chunks, embeddings, collection_name=f"m02-{name}")
        top = store.similarity_search(question, k=1)[0]
        preview = " ".join(top.page_content.split())[:150]
        print(f"\n  {name:<10} -> {preview}...")
        store.delete_collection()
