# %% [markdown]
# # Module 09 — Evaluating RAG
#
# Retrieval metrics and generation metrics, implemented by hand, over three pipeline configs.
# Read `README.md` in this folder first.
#
# Runtime: a couple of minutes and a few cents of API usage.

# %%
import json
from collections import defaultdict
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
judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

K = 4


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

golden = json.loads((ROOT / "data" / "eval" / "golden_set.json").read_text(encoding="utf-8"))
examples = golden["examples"]

print(f"Loaded {len(examples)} evaluation examples")
for kind, count in sorted(
    {t: sum(1 for e in examples if e["type"] == t) for t in {e["type"] for e in examples}}.items()
):
    print(f"  {kind:<18} {count}")

# %% [markdown]
# ## Three configurations to compare

# %%
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
docs = [
    Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})
    for p in sorted((ROOT / "data" / "corpus").glob("*.md"))
]
chunks = splitter.split_documents(docs)

store = Chroma.from_documents(chunks, embeddings, collection_name="m09")

dense_narrow = store.as_retriever(search_kwargs={"k": K})
dense_wide = store.as_retriever(search_kwargs={"k": 20})
sparse_wide = BM25Retriever.from_documents(chunks)
sparse_wide.k = 20
hybrid = EnsembleRetriever(retrievers=[dense_wide, sparse_wide], weights=[0.5, 0.5])

rerank_prompt = ChatPromptTemplate.from_template(
    """Rate 0-10 how well this passage answers the question.
Respond with a single integer only.

Question: {question}

Passage:
{passage}

Score:"""
)
scorer = rerank_prompt | llm | StrOutputParser()


def rerank(question: str, candidates: list[Document], top_k: int) -> list[Document]:
    scored = []
    for doc in candidates[:15]:  # cap the reranking cost
        try:
            score = float(scorer.invoke(
                {"question": question, "passage": doc.page_content}
            ).strip().split()[0])
        except (ValueError, IndexError):
            score = 0.0
        scored.append((doc, score))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _ in scored[:top_k]]


CONFIGS = {
    "dense": lambda q: dense_narrow.invoke(q),
    "hybrid": lambda q: hybrid.invoke(q)[:K],
    "hybrid+rerank": lambda q: rerank(q, hybrid.invoke(q), K),
}

# %% [markdown]
# ## Retrieval metrics
#
# No LLM involved — these are set operations over document names. Fast and free, which means you
# can run them on every change.

# %%
def hit_rate(retrieved: list[str], expected: list[str]) -> float:
    """Did any expected document appear at all?"""
    if not expected:
        return 1.0  # unanswerable: nothing to find, so nothing to miss
    return 1.0 if set(retrieved) & set(expected) else 0.0


def recall_at_k(retrieved: list[str], expected: list[str]) -> float:
    """What fraction of expected documents were found?"""
    if not expected:
        return 1.0
    return len(set(retrieved) & set(expected)) / len(set(expected))


def precision_at_k(retrieved: list[str], expected: list[str]) -> float:
    """What fraction of retrieved documents were relevant?"""
    if not retrieved:
        return 0.0
    if not expected:
        return 0.0  # anything retrieved for an unanswerable question is noise
    return len(set(retrieved) & set(expected)) / len(set(retrieved))


def reciprocal_rank(retrieved: list[str], expected: list[str]) -> float:
    """1 / position of the first correct document. Rewards ranking, not just finding."""
    if not expected:
        return 1.0
    for i, name in enumerate(retrieved, 1):
        if name in expected:
            return 1.0 / i
    return 0.0


# %% [markdown]
# ## Generation metrics (LLM-as-judge)
#
# Note the shape of these prompts: reason first, verdict second, and a constrained output
# vocabulary. Both details measurably reduce judge noise.

# %%
faithfulness_prompt = ChatPromptTemplate.from_template(
    """You are grading whether an answer is supported by its context.

Context:
{context}

Answer:
{answer}

Is every factual claim in the answer supported by the context?
First give one sentence of reasoning, then on a new line output exactly one word:
SUPPORTED, PARTIAL, or UNSUPPORTED.

If the answer states that the information is unavailable, and the context indeed does not
contain it, that counts as SUPPORTED."""
)

correctness_prompt = ChatPromptTemplate.from_template(
    """Compare a generated answer against the reference answer.

Question: {question}
Reference answer: {reference}
Generated answer: {answer}

Does the generated answer convey the same information as the reference?
Minor wording differences are fine. If the reference is "NOT_IN_CORPUS", the generated answer
is correct only if it declines to answer.

First give one sentence of reasoning, then on a new line output exactly one word:
CORRECT, PARTIAL, or INCORRECT."""
)

faithfulness_judge = faithfulness_prompt | judge | StrOutputParser()
correctness_judge = correctness_prompt | judge | StrOutputParser()

VERDICT_SCORES = {
    "SUPPORTED": 1.0, "PARTIAL": 0.5, "UNSUPPORTED": 0.0,
    "CORRECT": 1.0, "INCORRECT": 0.0,
}


def parse_verdict(response: str) -> float:
    last = response.strip().split("\n")[-1].strip().upper()
    for token, value in VERDICT_SCORES.items():
        if token in last:
            return value
    return 0.0


answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer using only the provided context. If the context does not contain the answer, "
     "reply exactly: 'The provided documents do not cover this.' Never guess."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])
answer_chain = answer_prompt | llm | StrOutputParser()

# %% [markdown]
# ## Run the evaluation

# %%
def evaluate(config_name: str, retrieve) -> dict:
    per_type = defaultdict(list)
    totals = defaultdict(list)

    for example in examples:
        retrieved_docs = retrieve(example["question"])
        retrieved_names = [d.metadata["source"] for d in retrieved_docs]
        expected = example["sources"]

        context = "\n\n".join(
            f"[{d.metadata['source']}]\n{d.page_content}" for d in retrieved_docs
        )
        answer = answer_chain.invoke(
            {"context": context, "question": example["question"]}
        )

        scores = {
            "hit_rate": hit_rate(retrieved_names, expected),
            "recall": recall_at_k(retrieved_names, expected),
            "precision": precision_at_k(retrieved_names, expected),
            "mrr": reciprocal_rank(retrieved_names, expected),
            "faithfulness": parse_verdict(
                faithfulness_judge.invoke({"context": context, "answer": answer})
            ),
            "correctness": parse_verdict(
                correctness_judge.invoke({
                    "question": example["question"],
                    "reference": example["answer"],
                    "answer": answer,
                })
            ),
        }

        for metric, value in scores.items():
            totals[metric].append(value)
            per_type[example["type"]].append((metric, value))

    return {
        "overall": {m: sum(v) / len(v) for m, v in totals.items()},
        "per_type": per_type,
    }


results = {}
for name, retrieve in CONFIGS.items():
    print(f"\nEvaluating {name}...")
    results[name] = evaluate(name, retrieve)

# %% [markdown]
# ## Results

# %%
METRICS = ["hit_rate", "recall", "precision", "mrr", "faithfulness", "correctness"]

print("\n" + "=" * 84)
print("OVERALL")
print("=" * 84)
print(f"\n{'config':<16}" + "".join(f"{m:>12}" for m in METRICS))
print("-" * 84)
for name, result in results.items():
    row = "".join(f"{result['overall'][m]:>12.3f}" for m in METRICS)
    print(f"{name:<16}{row}")

print("\n" + "=" * 84)
print("BY QUESTION TYPE  (hit_rate / correctness)")
print("=" * 84)

types = sorted({e["type"] for e in examples})
print(f"\n{'config':<16}" + "".join(f"{t:>20}" for t in types))
print("-" * 84)
for name, result in results.items():
    cells = []
    for question_type in types:
        pairs = result["per_type"][question_type]
        hits = [v for m, v in pairs if m == "hit_rate"]
        correct = [v for m, v in pairs if m == "correctness"]
        cells.append(f"{sum(hits)/len(hits):.2f} / {sum(correct)/len(correct):.2f}".rjust(20))
    print(f"{name:<16}" + "".join(cells))

print("\nRead this table for *differences between configs*, not absolute quality — "
      "14 examples is far too few for the absolute numbers to mean much.")

store.delete_collection()
