# %% [markdown]
# # Module 06 — Query Transformation
#
# Rewriting, multi-query, HyDE, step-back, and decomposition — measured against a plain search.
# Read `README.md` in this folder first.

# %%
from pathlib import Path

from dotenv import load_dotenv
from langchain.retrievers.multi_query import MultiQueryRetriever
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


splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
docs = [
    Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})
    for p in sorted(find_corpus().glob("*.md"))
]
store = Chroma.from_documents(splitter.split_documents(docs), embeddings, collection_name="m06")
retriever = store.as_retriever(search_kwargs={"k": 3})


def show(label: str, chunks: list[Document]) -> None:
    print(f"\n  {label}")
    for chunk in chunks:
        print(f"    [{chunk.metadata['source']:<24}] "
              f"{' '.join(chunk.page_content.split())[:72]}...")


# A deliberately vague, conversational query — the kind real users actually type.
VAGUE = "my robot won't charge, what do I do"

# %% [markdown]
# ## Baseline: search with the raw question

# %%
print("=" * 74)
print(f"BASELINE  —  {VAGUE!r}")
print("=" * 74)
show("raw query", retriever.invoke(VAGUE))

# %% [markdown]
# ## 1. Query rewriting
#
# One LLM call turns the complaint into search vocabulary. Note the instruction to emit *only*
# the query — a chatty model that replies "Sure! Here's a better query:" will poison the
# embedding with irrelevant tokens.

# %%
rewrite_prompt = ChatPromptTemplate.from_template(
    """Rewrite the user's question as a concise search query for a technical
documentation database. Use terminology likely to appear in the docs.
Output only the query, nothing else.

Question: {question}
Search query:"""
)
rewriter = rewrite_prompt | llm | StrOutputParser()

rewritten = rewriter.invoke({"question": VAGUE}).strip()
print("\n" + "=" * 74)
print("1. QUERY REWRITING")
print("=" * 74)
print(f"\n  {VAGUE!r}\n  -> {rewritten!r}")
show("rewritten query", retriever.invoke(rewritten))

# %% [markdown]
# ## 2. Multi-query
#
# `MultiQueryRetriever` generates variations, retrieves for each, and returns the deduplicated
# union. Because it unions, it returns more than `k` — budget for that downstream.

# %%
multi_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)

print("\n" + "=" * 74)
print("2. MULTI-QUERY")
print("=" * 74)
multi_results = multi_retriever.invoke(VAGUE)
print(f"\n  union of all variations: {len(multi_results)} unique chunks "
      f"(vs {len(retriever.invoke(VAGUE))} from a single query)")
show("union", multi_results)

# %% [markdown]
# ## 3. HyDE
#
# Generate a fake answer and search with *that*. The facts in it will be wrong — Helix Robotics
# does not exist — but it is answer-shaped, which is the entire point.

# %%
hyde_prompt = ChatPromptTemplate.from_template(
    """Write a short passage from a technical manual that would answer this question.
Invent plausible specifics if you must. Write it as documentation, not as a reply.

Question: {question}
Passage:"""
)
hyde_generator = hyde_prompt | llm | StrOutputParser()

hypothetical = hyde_generator.invoke({"question": VAGUE})
print("\n" + "=" * 74)
print("3. HyDE")
print("=" * 74)
print(f"\n  hypothetical document (factually invented, structurally useful):")
print(f"    {' '.join(hypothetical.split())[:300]}...")
show("retrieved using the hypothetical document", retriever.invoke(hypothetical))

# %% [markdown]
# ## 4. Step-back prompting
#
# A narrow question that the docs answer only via a general principle. The literal query looks
# for "freezer"; the step-back query looks for the temperature rating that actually decides it.

# %%
SPECIFIC = "Can I run an Atlas R5 in a freezer warehouse at -20 degrees?"

stepback_prompt = ChatPromptTemplate.from_template(
    """Given a specific question, write a more general question whose answer would provide
the background needed to answer the specific one. Output only the question.

Specific: {question}
General:"""
)
stepback = stepback_prompt | llm | StrOutputParser()

general = stepback.invoke({"question": SPECIFIC}).strip()
print("\n" + "=" * 74)
print("4. STEP-BACK PROMPTING")
print("=" * 74)
print(f"\n  specific: {SPECIFIC!r}")
print(f"  general:  {general!r}")
show("specific query", retriever.invoke(SPECIFIC))
show("general (step-back) query", retriever.invoke(general))

# %% [markdown]
# ## 5. Decomposition
#
# A compound question embeds to the average of its parts, which can land between the relevant
# clusters rather than in any of them. Split it, retrieve per part, then answer from the merge.

# %%
COMPOUND = (
    "How do the Atlas and Scout differ in their charging docks, "
    "and how long is the battery warranty on each?"
)

decompose_prompt = ChatPromptTemplate.from_template(
    """Break this question into 2-4 standalone sub-questions, each answerable on its own.
Output one per line with no numbering or bullets.

Question: {question}
Sub-questions:"""
)
decomposer = decompose_prompt | llm | StrOutputParser()

sub_questions = [
    line.strip()
    for line in decomposer.invoke({"question": COMPOUND}).split("\n")
    if line.strip()
]

print("\n" + "=" * 74)
print("5. DECOMPOSITION")
print("=" * 74)
print(f"\n  compound: {COMPOUND!r}")
show("compound query, retrieved directly", retriever.invoke(COMPOUND))

merged: dict[str, Document] = {}
for sub in sub_questions:
    print(f"\n  sub-question: {sub!r}")
    for chunk in retriever.invoke(sub):
        key = f"{chunk.metadata['source']}:{chunk.page_content[:40]}"
        merged[key] = chunk
        print(f"    [{chunk.metadata['source']:<24}] "
              f"{' '.join(chunk.page_content.split())[:66]}...")

print(f"\n  merged unique chunks across sub-questions: {len(merged)}")

# %% [markdown]
# ## Does better retrieval produce a better answer?
#
# Retrieval quality only matters if it changes the output. Same compound question, answered
# from directly-retrieved context vs decomposed context.

# %%
answer_prompt = ChatPromptTemplate.from_template(
    """Answer using only the context. If something is missing, say so explicitly.

Context:
{context}

Question: {question}
Answer:"""
)
answer_chain = answer_prompt | llm | StrOutputParser()


def as_context(chunks) -> str:
    return "\n\n".join(
        f"[{c.metadata['source']}]\n{c.page_content}" for c in chunks
    )


print("\n" + "=" * 74)
print("ANSWER COMPARISON")
print("=" * 74)

print("\n--- from the direct compound query ---")
print(answer_chain.invoke(
    {"context": as_context(retriever.invoke(COMPOUND)), "question": COMPOUND}
))

print("\n--- from decomposed sub-questions ---")
print(answer_chain.invoke(
    {"context": as_context(list(merged.values())), "question": COMPOUND}
))

store.delete_collection()
