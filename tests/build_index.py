import numpy as np

from app.ingestion.loader import load_pdf
from app.ingestion.chunker import chunk_text
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import VectorStore


PDF_PATH = "data/documents/test.pdf"

INDEX_PATH = "data/index/faiss.index"
METADATA_PATH = "data/index/metadata.json"


print("=" * 60)
print("BUILDING RAG INDEX")
print("=" * 60)


# ------------------------------------------------------------
# 1. LOAD PDF
# ------------------------------------------------------------

print("\nLoading PDF...")

pages = load_pdf(
    PDF_PATH
)

print(
    f"Pages loaded: {len(pages)}"
)


# ------------------------------------------------------------
# 2. CHUNK DOCUMENT
# ------------------------------------------------------------

print("\nCreating chunks...")

chunks = chunk_text(
    pages
)

print(
    f"Chunks created: {len(chunks)}"
)


# ------------------------------------------------------------
# 3. LOAD EMBEDDING MODEL
# ------------------------------------------------------------

print("\nLoading embedding model...")

embedding_model = EmbeddingModel()


# ------------------------------------------------------------
# 4. GENERATE EMBEDDINGS
# ------------------------------------------------------------

print("\nGenerating embeddings...")

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = embedding_model.encode(
    texts
)

embeddings = np.asarray(
    embeddings,
    dtype=np.float32
)

print(
    f"Embedding shape: {embeddings.shape}"
)


# ------------------------------------------------------------
# 5. CREATE VECTOR STORE
# ------------------------------------------------------------

print("\nCreating FAISS vector store...")

vector_store = VectorStore(
    dimension=embeddings.shape[1]
)

vector_store.add(
    embeddings=embeddings,
    documents=chunks
)


# ------------------------------------------------------------
# 6. SAVE INDEX
# ------------------------------------------------------------

print("\nSaving index...")

vector_store.save(
    index_path=INDEX_PATH,
    metadata_path=METADATA_PATH
)


# ------------------------------------------------------------
# 7. VERIFY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("INDEX BUILD COMPLETE")
print("=" * 60)

print(
    f"Vectors: {vector_store.index.ntotal}"
)

print(
    f"Dimension: {vector_store.index.d}"
)

print(
    f"Metadata records: {len(vector_store.documents)}"
)

print(
    f"Index: {INDEX_PATH}"
)

print(
    f"Metadata: {METADATA_PATH}"
)