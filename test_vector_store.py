from app.ingestion.loader import load_pdf
from app.ingestion.chunker import chunk_text
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import VectorStore


PDF_PATH = "data/documents/test.pdf"


# --------------------------------------------------
# 1. Load PDF
# --------------------------------------------------

pages = load_pdf(PDF_PATH)


# --------------------------------------------------
# 2. Create metadata-aware chunks
# --------------------------------------------------

chunks = chunk_text(pages)


# --------------------------------------------------
# 3. Extract text for embedding
# --------------------------------------------------

texts = [
    chunk["text"]
    for chunk in chunks
]


# --------------------------------------------------
# 4. Generate embeddings
# --------------------------------------------------

embedding_model = EmbeddingModel()

embeddings = embedding_model.encode(texts)


# --------------------------------------------------
# 5. Create vector store
# --------------------------------------------------

vector_store = VectorStore(
    dimension=embeddings.shape[1]
)


# --------------------------------------------------
# 6. Add embeddings + metadata
# --------------------------------------------------

vector_store.add(
    embeddings=embeddings,
    documents=chunks
)


# --------------------------------------------------
# 7. Search
# --------------------------------------------------

query = "How do I verify my student status?"

query_embedding = embedding_model.encode([query])

results = vector_store.search(
    query_embedding=query_embedding,
    top_k=3
)


# --------------------------------------------------
# 8. Display results
# --------------------------------------------------

print("=" * 60)
print("METADATA-AWARE VECTOR STORE TEST")
print("=" * 60)

print(f"Pages: {len(pages)}")
print(f"Chunks: {len(chunks)}")
print(f"FAISS vectors: {vector_store.index.ntotal}")

print("\nQuery:")
print(query)

for i, result in enumerate(results, start=1):

    print("\n" + "=" * 60)
    print(f"RESULT {i}")

    print(f"Score: {result['score']:.4f}")
    print(f"Chunk ID: {result['chunk_id']}")
    print(f"Page: {result['page']}")
    print(f"Source: {result['source']}")

    print("\nText:")
    print(result["text"])