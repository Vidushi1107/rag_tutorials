# %% [markdown]
# # Module 11 — Conversational RAG
#
# Follow-up questions, condensation, and history that does not grow forever.
# Read `README.md` in this folder first.

# %%
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
encoder = tiktoken.get_encoding("cl100k_base")


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


docs = [
    Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})
    for p in sorted(find_corpus().glob("*.md"))
]
store = Chroma.from_documents(
    RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80).split_documents(docs),
    embeddings,
    collection_name="m11",
)
retriever = store.as_retriever(search_kwargs={"k": 3})

# %% [markdown]
# ## The failure
#
# A follow-up that is meaningless on its own. Watch what a stateless retriever does with it.

# %%
FOLLOW_UP = "How about the Scout?"

print("=" * 78)
print("NAIVE — retrieving with the raw follow-up")
print("=" * 78)
print("\n  turn 1: 'What is the maximum payload of the Atlas R5?'")
print("  turn 2: 'How about the Scout?'   <- retrieving with only this\n")
for chunk in retriever.invoke(FOLLOW_UP):
    print(f"    [{chunk.metadata['source']:<24}] "
          f"{' '.join(chunk.page_content.split())[:64]}...")
print("\nThe words 'payload', 'capacity' and 'kg' appear nowhere in the query, "
      "so nothing steers retrieval toward the specification.")

# %% [markdown]
# ## Condensation
#
# Rewrite the follow-up into a standalone question first. Note the instruction to return the
# question unchanged when it already stands alone — over-eager rewriting damages good queries.

# %%
condense_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given the conversation history and a follow-up question, rewrite the follow-up as a "
     "standalone question that makes sense without the history. Resolve pronouns and fill in "
     "implied subjects. If the question is already standalone, return it unchanged. "
     "Return only the question — never answer it."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
condenser = condense_prompt | llm | StrOutputParser()

history_pairs = [
    ("human", "What is the maximum payload of the Atlas R5?"),
    ("ai", "The Atlas R5 has a maximum payload of 450 kg."),
]

condensed = condenser.invoke({"chat_history": history_pairs, "input": FOLLOW_UP})

print("\n" + "=" * 78)
print("CONDENSED")
print("=" * 78)
print(f"\n  {FOLLOW_UP!r}\n  -> {condensed.strip()!r}\n")
for chunk in retriever.invoke(condensed):
    print(f"    [{chunk.metadata['source']:<24}] "
          f"{' '.join(chunk.page_content.split())[:64]}...")

# %% [markdown]
# ## Standalone questions should pass through untouched

# %%
print("\nPass-through check:")
for question in [
    "What is the battery warranty period?",
    "Is that covered?",
    "and the docks?",
]:
    rewritten = condenser.invoke({"chat_history": history_pairs, "input": question}).strip()
    changed = "rewritten" if rewritten.lower() != question.lower() else "unchanged"
    print(f"  [{changed:<9}] {question!r} -> {rewritten!r}")

# %% [markdown]
# ## The full chain
#
# `create_history_aware_retriever` wires condensation in front of the retriever.
# `create_retrieval_chain` joins that to the answering step. Note that the answering prompt
# receives the *original* input plus history — the rewrite is a search artefact only.

# %%
history_aware_retriever = create_history_aware_retriever(llm, retriever, condense_prompt)

answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about Helix Robotics using only the context below. "
     "If the context does not contain the answer, say so. Be concise.\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    create_stuff_documents_chain(llm, answer_prompt),
)

sessions: dict[str, BaseChatMessageHistory] = {}


def get_history(session_id: str) -> BaseChatMessageHistory:
    """Per-session isolation. In production this is Redis or a database, not a dict."""
    if session_id not in sessions:
        sessions[session_id] = InMemoryChatMessageHistory()
    return sessions[session_id]


conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    get_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# %% [markdown]
# ## A real conversation
#
# Six turns, each depending on the ones before it.

# %%
conversation = [
    "What is the maximum payload of the Atlas R5?",
    "How about the Scout?",
    "Can they share a charging dock?",
    "Why not?",
    "What error would that produce?",
    "Is that covered under warranty?",
]

config = {"configurable": {"session_id": "demo"}}

print("\n" + "=" * 78)
print("CONVERSATION")
print("=" * 78)

for turn, question in enumerate(conversation, 1):
    result = conversational_rag.invoke({"input": question}, config=config)
    sources = sorted({d.metadata["source"] for d in result["context"]})
    print(f"\n[{turn}] user: {question}")
    print(f"    bot:  {' '.join(result['answer'].split())}")
    print(f"    (retrieved from: {', '.join(sources)})")

# %% [markdown]
# ## History growth
#
# Full history is simple and gets steadily more expensive. Measure it before deciding you do not
# care.

# %%
messages = get_history("demo").messages

print("\n" + "=" * 78)
print("HISTORY COST")
print("=" * 78)
print(f"\n{'after turn':>12}  {'full history':>14}  {'windowed (last 4 msgs)':>24}")
print("-" * 56)
for turn in range(1, len(conversation) + 1):
    upto = messages[: turn * 2]
    full = sum(len(encoder.encode(m.content)) for m in upto)
    windowed = sum(len(encoder.encode(m.content)) for m in upto[-4:])
    print(f"{turn:>12}  {full:>10} tok  {windowed:>20} tok")

# %% [markdown]
# ## Summarise-and-window
#
# Compress old turns into a summary, keep recent turns verbatim. Retains long-range references at
# a fraction of the tokens.

# %%
summary_prompt = ChatPromptTemplate.from_template(
    """Summarise this conversation in 2-3 sentences. Keep specific entities, products, and
numbers that later turns might refer back to.

{conversation}

Summary:"""
)
summarizer = summary_prompt | llm | StrOutputParser()

KEEP_RECENT = 4
old, recent = messages[:-KEEP_RECENT], messages[-KEEP_RECENT:]

transcript = "\n".join(f"{m.type}: {m.content}" for m in old)
summary = summarizer.invoke({"conversation": transcript})

old_tokens = sum(len(encoder.encode(m.content)) for m in old)
summary_tokens = len(encoder.encode(summary))

print(f"\n  {len(old)} older messages: {old_tokens} tokens")
print(f"  summarised to:          {summary_tokens} tokens "
      f"({100 - summary_tokens * 100 // max(old_tokens, 1)}% smaller)")
print(f"\n  summary: {' '.join(summary.split())}")
print(f"\n  + {len(recent)} recent messages kept verbatim")

store.delete_collection()
