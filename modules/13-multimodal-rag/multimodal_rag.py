# %% [markdown]
# # Module 13 — Multi-modal RAG
#
# Tables and images, handled by indexing a description and returning the original.
# Read `README.md` in this folder first.
#
# Optional, for the vision section: `pip install matplotlib`

# %%
import base64
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain.retrievers import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
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

# %% [markdown]
# ## The problem, made concrete
#
# Here is what naive text extraction does to a specification table.

# %%
atlas = (CORPUS / "product-atlas-r5.md").read_text(encoding="utf-8")

table_match = re.search(r"\| Property \|.*?\n\n", atlas, re.DOTALL)
raw_table = table_match.group(0).strip()

flattened = " ".join(
    cell.strip()
    for line in raw_table.split("\n")
    for cell in line.split("|")
    if cell.strip() and not set(cell.strip()) <= {"-", " "}
)

print("=" * 78)
print("STRUCTURED (as it exists in the source)")
print("=" * 78)
print(raw_table)

print("\n" + "=" * 78)
print("FLATTENED (what naive PDF extraction typically gives you)")
print("=" * 78)
print(f"\n{flattened}\n")

cut = flattened[180:260]
print(f"Now chunk it. A boundary landing here yields:\n  ...{cut}...")
print("\nWhich number belongs to which property? The association is gone, and nothing "
      "downstream can flag that it is missing.")

# %% [markdown]
# ## Extract tables as atomic units
#
# Pull tables out before splitting, so no splitter can cut one in half. Prose and tables then get
# handled separately.

# %%
def extract_tables(text: str, source: str) -> tuple[list[Document], str]:
    """Return table documents plus the prose with tables removed."""
    tables = []
    for match in re.finditer(r"(?:^\|.*\n)+", text, re.MULTILINE):
        block = match.group(0).strip()
        if block.count("\n") >= 2:  # header, separator, at least one row
            tables.append(Document(page_content=block, metadata={"source": source,
                                                                 "kind": "table"}))
    prose = re.sub(r"(?:^\|.*\n)+", "\n[TABLE OMITTED]\n", text, flags=re.MULTILINE)
    return tables, prose


all_tables: list[Document] = []
all_prose: list[Document] = []
for path in sorted(CORPUS.glob("*.md")):
    text = path.read_text(encoding="utf-8")
    tables, prose = extract_tables(text, path.name)
    all_tables.extend(tables)
    all_prose.append(Document(page_content=prose, metadata={"source": path.name,
                                                            "kind": "prose"}))

print("\n" + "=" * 78)
print(f"EXTRACTED {len(all_tables)} TABLES")
print("=" * 78)
for table in all_tables:
    first_row = table.page_content.split("\n")[0]
    print(f"  [{table.metadata['source']:<24}] {first_row[:56]}... "
          f"({table.page_content.count(chr(10))} rows)")

# %% [markdown]
# ## Summarise tables for indexing
#
# A grid of pipes and numbers embeds poorly. A sentence describing what the table contains embeds
# well. Index the sentence, return the table.

# %%
table_summary_prompt = ChatPromptTemplate.from_template(
    """Describe what this table contains in 1-2 sentences, for a search index.
Name the entity it describes and the properties it covers. Include a few key values.

Table:
{table}

Description:"""
)
table_summarizer = table_summary_prompt | llm | StrOutputParser()

print("\n" + "=" * 78)
print("TABLE SUMMARIES")
print("=" * 78)

table_summaries = []
for table in all_tables:
    summary = table_summarizer.invoke({"table": table.page_content}).strip()
    table_summaries.append(summary)
    print(f"\n  [{table.metadata['source']}]")
    print(f"  {' '.join(summary.split())}")

# %% [markdown]
# ## One retriever holding prose, tables, and (below) images
#
# `MultiVectorRetriever` indexes the summaries and resolves matches back to the original content.

# %%
ID_KEY = "doc_id"
vector_store = Chroma(collection_name="m13", embedding_function=embeddings)
docstore = InMemoryStore()
retriever = MultiVectorRetriever(
    vectorstore=vector_store, docstore=docstore, id_key=ID_KEY
)

prose_chunks = RecursiveCharacterTextSplitter(
    chunk_size=700, chunk_overlap=100
).split_documents(all_prose)

# Prose is indexed as itself; tables are indexed by their summary.
prose_ids = [str(uuid.uuid4()) for _ in prose_chunks]
retriever.vectorstore.add_documents([
    Document(page_content=chunk.page_content, metadata={ID_KEY: doc_id, "kind": "prose"})
    for chunk, doc_id in zip(prose_chunks, prose_ids)
])
retriever.docstore.mset(list(zip(prose_ids, prose_chunks)))

table_ids = [str(uuid.uuid4()) for _ in all_tables]
retriever.vectorstore.add_documents([
    Document(page_content=summary, metadata={ID_KEY: doc_id, "kind": "table"})
    for summary, doc_id in zip(table_summaries, table_ids)
])
retriever.docstore.mset(list(zip(table_ids, all_tables)))

print(f"\nIndexed {len(prose_chunks)} prose chunks + {len(all_tables)} table summaries")

# %% [markdown]
# ## Retrieval returns the table, not the summary

# %%
QUESTION = "What are the Atlas R5 dimensions and ingress protection rating?"

print("\n" + "=" * 78)
print(f"Q: {QUESTION}")
print("=" * 78)

matched = vector_store.similarity_search(QUESTION, k=1)[0]
print(f"\nmatched this indexed text ({matched.metadata['kind']}):")
print(f"  {' '.join(matched.page_content.split())[:180]}...")

print("\nreturned this original content:")
print("  " + retriever.invoke(QUESTION)[0].page_content.replace("\n", "\n  ")[:520])

# %% [markdown]
# ## Images
#
# Same pattern with a vision model doing the describing. Generates a sample chart if matplotlib is
# available — the chart deliberately encodes data that appears nowhere in the corpus text.

# %%
IMAGES_DIR = ROOT / "data" / "images"
chart_path = IMAGES_DIR / "atlas-battery-discharge.png"

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    if not chart_path.exists():
        hours = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.5]
        laden = [100, 89, 78, 67, 56, 45, 34, 24, 15, 6, 2]
        unladen = [100, 93, 86, 79, 72, 65, 58, 51, 44, 37, 34]

        figure, axes = plt.subplots(figsize=(7, 4))
        axes.plot(hours, laden, marker="o", label="Laden (450 kg)")
        axes.plot(hours, unladen, marker="s", label="Unladen")
        axes.axhline(25, linestyle="--", color="red", label="Auto-return threshold (25%)")
        axes.set_xlabel("Hours of operation")
        axes.set_ylabel("State of charge (%)")
        axes.set_title("Atlas R5 battery discharge")
        axes.legend()
        axes.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(chart_path, dpi=100)
        plt.close(figure)
        print(f"\nGenerated sample chart: {chart_path}")

    encoded = base64.b64encode(chart_path.read_bytes()).decode("utf-8")

    vision_response = llm.invoke([
        HumanMessage(content=[
            {"type": "text",
             "text": "Describe this chart for a search index. State what it plots, the axes, "
                     "the series, and the key values a reader could look up. 3-4 sentences."},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{encoded}"}},
        ])
    ])
    description = vision_response.content

    print("\n" + "=" * 78)
    print("VISION-GENERATED DESCRIPTION")
    print("=" * 78)
    print(f"\n{' '.join(description.split())}")

    image_id = str(uuid.uuid4())
    retriever.vectorstore.add_documents([
        Document(page_content=description, metadata={ID_KEY: image_id, "kind": "image"})
    ])
    retriever.docstore.mset([(
        image_id,
        Document(
            page_content=f"[IMAGE: {chart_path.name}]\n{description}",
            metadata={"source": chart_path.name, "kind": "image"},
        ),
    )])

    IMAGE_QUESTION = "At what point does the robot automatically return to its dock?"
    print("\n" + "=" * 78)
    print(f"Q: {IMAGE_QUESTION}")
    print("=" * 78)
    for result in retriever.invoke(IMAGE_QUESTION)[:2]:
        print(f"\n  kind={result.metadata.get('kind', 'prose')} "
              f"source={result.metadata.get('source', '?')}")
        print(f"  {' '.join(result.page_content.split())[:220]}...")

    print("\nThe chart is now retrievable by its content, using the same text embedding "
          "model and the same vector store as everything else.")
except ImportError:
    print("\n[skipped] the image section needs: pip install matplotlib")
    print("          (any PNG at data/images/ would work just as well)")

# %% [markdown]
# ## PDFs
#
# Runs against any PDF you drop into `data/pdfs/`. The point is to see how much structure your
# extraction library preserves — that is where most real-world quality is won or lost.

# %%
pdf_dir = ROOT / "data" / "pdfs"
pdfs = sorted(pdf_dir.glob("*.pdf")) if pdf_dir.is_dir() else []

if not pdfs:
    print(f"\n[no PDFs found] Drop one into {pdf_dir} to try extraction.")
    print("""
Extraction library, in rough order of capability:

  pypdf         text only, no layout awareness
  pymupdf       fast, extracts embedded images too
  pdfplumber    genuine table extraction
  unstructured  layout-aware: titles, tables, figures separated
  cloud OCR     scans, forms, handwriting

Then apply this module's pattern: tables as atomic units with summaries indexed,
images described by a vision model, prose chunked normally.
""")
else:
    try:
        from langchain_community.document_loaders import PyPDFLoader

        pages = PyPDFLoader(str(pdfs[0])).load()
        print(f"\nLoaded {len(pages)} pages from {pdfs[0].name}")
        print(f"\nFirst page, first 400 chars:\n  "
              f"{' '.join(pages[0].page_content.split())[:400]}...")
        print("\nCheck whether tables survived. Usually they did not.")
    except ImportError:
        print("\n[skipped] PDF loading needs: pip install pypdf")

vector_store.delete_collection()
