# %% [markdown]
# # Module 10 — Advanced Retrievers
#
# Four ways to search one representation and return a different one.
# Read `README.md` in this folder first.
#
# Needs one extra package for the self-query section: `pip install lark`

# %%
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain.retrievers import MultiVectorRetriever, ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
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
docs = [
    Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})
    for p in sorted(CORPUS.glob("*.md"))
]

QUESTION = "How long does it take to charge?"

# %% [markdown]
# ## Baseline: one representation for both jobs

# %%
baseline_store = Chroma.from_documents(
    RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50).split_documents(docs),
    embeddings,
    collection_name="m10-baseline",
)
baseline = baseline_store.similarity_search(QUESTION, k=1)[0]

print("=" * 78)
print(f"BASELINE — 400-char chunks, search and return the same text")
print("=" * 78)
print(f"\nQ: {QUESTION}")
print(f"\nreturned {len(baseline.page_content)} chars:\n")
print(f"  {' '.join(baseline.page_content.split())}")
print("\nPrecise match, but is there enough here to know *what* is charging?")

# %% [markdown]
# ## 1. Parent document retrieval
#
# Two splitters: children get embedded and searched, parents get returned. The vector store holds
# children; a separate docstore holds parents, linked by ID.

# %%
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

parent_retriever = ParentDocumentRetriever(
    vectorstore=Chroma(
        collection_name="m10-parent", embedding_function=embeddings
    ),
    docstore=InMemoryStore(),
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
parent_retriever.add_documents(docs)

matched_child = parent_retriever.vectorstore.similarity_search(QUESTION, k=1)[0]
returned_parent = parent_retriever.invoke(QUESTION)[0]

print("\n" + "=" * 78)
print("1. PARENT DOCUMENT RETRIEVAL")
print("=" * 78)
print(f"\nmatched on this child ({len(matched_child.page_content)} chars):")
print(f"  {' '.join(matched_child.page_content.split())[:200]}...")
print(f"\nreturned this parent ({len(returned_parent.page_content)} chars):")
print(f"  {' '.join(returned_parent.page_content.split())[:320]}...")
print(f"\n{len(returned_parent.page_content) // max(len(matched_child.page_content), 1)}x "
      "more context, same precision of match.")

# %% [markdown]
# ## 2. Sentence-window retrieval
#
# Index single sentences; return the sentence plus its neighbours. Built by hand here because the
# mechanism is the lesson — each sentence carries its position, and retrieval slices a window
# around it.

# %%
WINDOW = 2

sentence_docs = []
for doc in docs:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", doc.page_content) if s.strip()]
    for i, sentence in enumerate(sentences):
        sentence_docs.append(
            Document(
                page_content=sentence,
                metadata={"source": doc.metadata["source"], "position": i},
            )
        )

sentences_by_source: dict[str, list[str]] = {}
for sentence_doc in sentence_docs:
    sentences_by_source.setdefault(sentence_doc.metadata["source"], []).append(
        sentence_doc.page_content
    )

window_store = Chroma.from_documents(sentence_docs, embeddings, collection_name="m10-window")


def sentence_window_search(query: str, k: int = 1) -> list[str]:
    windows = []
    for hit in window_store.similarity_search(query, k=k):
        source = hit.metadata["source"]
        position = hit.metadata["position"]
        neighbours = sentences_by_source[source]
        start = max(0, position - WINDOW)
        end = min(len(neighbours), position + WINDOW + 1)
        windows.append(" ".join(neighbours[start:end]))
    return windows


print("\n" + "=" * 78)
print("2. SENTENCE-WINDOW RETRIEVAL")
print("=" * 78)
top_sentence = window_store.similarity_search(QUESTION, k=1)[0]
print(f"\nmatched sentence:\n  {top_sentence.page_content}")
print(f"\nreturned window (+/- {WINDOW} sentences):\n  "
      f"{' '.join(sentence_window_search(QUESTION)[0].split())[:400]}...")

# %% [markdown]
# ## 3. Multi-vector: index hypothetical questions
#
# Generate the questions each chunk answers, index *those*, return the chunk. You end up matching
# questions against questions instead of questions against statements — the asymmetry problem
# from Module 06, solved at ingestion time.

# %%
chunks = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=100
).split_documents(docs)

question_prompt = ChatPromptTemplate.from_template(
    """List 3 questions that this passage directly answers.
One per line, no numbering. Write them the way a user would ask.

Passage:
{passage}

Questions:"""
)
question_generator = question_prompt | llm | StrOutputParser()

ID_KEY = "parent_id"
multi_store = Chroma(collection_name="m10-multivector", embedding_function=embeddings)
multi_docstore = InMemoryStore()

multi_retriever = MultiVectorRetriever(
    vectorstore=multi_store, docstore=multi_docstore, id_key=ID_KEY
)

print("\n" + "=" * 78)
print("3. MULTI-VECTOR — HYPOTHETICAL QUESTIONS")
print("=" * 78)
print(f"\nGenerating questions for {len(chunks)} chunks (one LLM call each)...")

chunk_ids = [str(uuid.uuid4()) for _ in chunks]
question_docs = []
for chunk, chunk_id in zip(chunks, chunk_ids):
    generated = [
        line.strip()
        for line in question_generator.invoke({"passage": chunk.page_content}).split("\n")
        if line.strip()
    ]
    for question in generated:
        question_docs.append(Document(page_content=question, metadata={ID_KEY: chunk_id}))

multi_retriever.vectorstore.add_documents(question_docs)
multi_retriever.docstore.mset(list(zip(chunk_ids, chunks)))

print(f"Indexed {len(question_docs)} generated questions pointing at {len(chunks)} chunks")

sample_id = chunk_ids[1]
print("\nExample — questions generated for one chunk:")
for question_doc in [q for q in question_docs if q.metadata[ID_KEY] == sample_id][:3]:
    print(f"  - {question_doc.page_content}")

print(f"\nQ: {QUESTION}")
matched_question = multi_store.similarity_search(QUESTION, k=1)[0]
print(f"\nmatched this indexed question:\n  {matched_question.page_content}")
print(f"\nreturned the underlying chunk:\n  "
      f"{' '.join(multi_retriever.invoke(QUESTION)[0].page_content.split())[:280]}...")

# %% [markdown]
# ## 4. Self-query: infer the filter from the question
#
# The LLM reads the question, extracts a search string *and* a metadata filter, then queries with
# both. You must describe the metadata schema accurately — a wrong description yields filters
# that silently match nothing.

# %%
try:
    from langchain.chains.query_constructor.schema import AttributeInfo
    from langchain.retrievers.self_query.base import SelfQueryRetriever

    tagged_chunks = []
    for chunk in chunks:
        source = chunk.metadata["source"]
        tagged_chunks.append(
            Document(
                page_content=chunk.page_content,
                metadata={
                    "source": source,
                    "doc_type": (
                        "product" if source.startswith("product")
                        else "support" if source == "support-faq.md"
                        else "policy"
                    ),
                    "product": (
                        "Atlas R5" if "atlas" in source
                        else "Scout M2" if "scout" in source
                        else "none"
                    ),
                },
            )
        )

    self_query_store = Chroma.from_documents(
        tagged_chunks, embeddings, collection_name="m10-selfquery"
    )

    self_query_retriever = SelfQueryRetriever.from_llm(
        llm,
        self_query_store,
        document_contents="Technical documentation about Helix Robotics products and policies",
        metadata_field_info=[
            AttributeInfo(
                name="source",
                description="Filename of the source document",
                type="string",
            ),
            AttributeInfo(
                name="doc_type",
                description="One of 'product', 'support', or 'policy'",
                type="string",
            ),
            AttributeInfo(
                name="product",
                description="Which product this describes: 'Atlas R5', 'Scout M2', or 'none'",
                type="string",
            ),
        ],
        verbose=True,
    )

    print("\n" + "=" * 78)
    print("4. SELF-QUERY RETRIEVAL")
    print("=" * 78)

    for query in [
        "What does the support FAQ say about error codes?",
        "Tell me about Scout M2 charging",
        "What is the warranty on batteries?",
    ]:
        print(f"\nQ: {query}")
        for result in self_query_retriever.invoke(query)[:2]:
            print(f"  [{result.metadata['source']:<24} product={result.metadata['product']:<9}] "
                  f"{' '.join(result.page_content.split())[:56]}...")
except ImportError:
    print("\n[skipped] self-query retriever needs: pip install lark")

# %% [markdown]
# ## Side by side
#
# Same question, four retrievers, very different returned text.

# %%
print("\n" + "=" * 78)
print(f"COMPARISON — {QUESTION!r}")
print("=" * 78)

comparisons = {
    "baseline (400 chars)": baseline.page_content,
    "parent document": returned_parent.page_content,
    "sentence window": sentence_window_search(QUESTION)[0],
    "multi-vector": multi_retriever.invoke(QUESTION)[0].page_content,
}

for name, text in comparisons.items():
    print(f"\n{name}  ({len(text)} chars)")
    print(f"  {' '.join(text.split())[:180]}...")

for store_to_clean in [baseline_store, window_store, multi_store]:
    store_to_clean.delete_collection()
