from app.ingestion.loader import load_pdf
from app.ingestion.chunker import chunk_text

from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import VectorStore


PDF_PATH = "data/documents/test.pdf"

INDEX_PATH = "data/index/faiss.index"
METADATA_PATH = "data/index/metadata.json"


# --------------------------------------------------
# 1. Load PDF
# --------------------------------------------------

pages = load_pdf(PDF_PATH)

chunks = chunk_text(pages)


# --------------------------------------------------
# 2. Generate embeddings
# --------------------------------------------------

texts = [
    chunk["text"]
    for chunk in chunks
]

embedding_model = EmbeddingModel()

embeddings = embedding_model.encode(texts)


# --------------------------------------------------
# 3. Build vector store
# --------------------------------------------------

vector_store = VectorStore(
    dimension=embeddings.shape[1]
)

vector_store.add(
    embeddings=embeddings,
    documents=chunks
)


# --------------------------------------------------
# 4. Save vector store
# --------------------------------------------------

vector_store.save(
    index_path=INDEX_PATH,
    metadata_path=METADATA_PATH
)

print("=" * 60)
print("VECTOR STORE SAVED")
print("=" * 60)

print(f"FAISS index: {INDEX_PATH}")
print(f"Metadata: {METADATA_PATH}")
print(f"Vectors saved: {vector_store.index.ntotal}")


# --------------------------------------------------
# 5. Load vector store again
# --------------------------------------------------

loaded_store = VectorStore.load(
    index_path=INDEX_PATH,
    metadata_path=METADATA_PATH
)


print("\n" + "=" * 60)
print("VECTOR STORE LOADED")
print("=" * 60)

print(f"Dimension: {loaded_store.dimension}")
print(f"Vectors loaded: {loaded_store.index.ntotal}")
print(f"Metadata records: {len(loaded_store.documents)}")


# --------------------------------------------------
# 6. Search loaded index
# --------------------------------------------------

query = "How do I verify my student status?"

query_embedding = embedding_model.encode(
    [query]
)

results = loaded_store.search(
    query_embedding=query_embedding,
    top_k=3
)


print("\n" + "=" * 60)
print("SEARCH AFTER RELOADING")
print("=" * 60)

print(f"Query: {query}")


for i, result in enumerate(
    results,
    start=1
):

    print("\n" + "-" * 60)

    print(f"Result {i}")
    print(f"Score: {result['score']:.4f}")
    print(f"Chunk ID: {result['chunk_id']}")
    print(f"Page: {result['page']}")
    print(f"Source: {result['source']}")

    print("\nText:")
    print(result["text"])