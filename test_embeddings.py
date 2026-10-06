from app.ingestion.loader import load_pdf
from app.ingestion.chunker import chunk_text
from app.retrieval.embeddings import EmbeddingModel


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
# 5. Display results
# --------------------------------------------------

print("=" * 60)
print("METADATA-AWARE EMBEDDING TEST")
print("=" * 60)

print(f"Pages: {len(pages)}")
print(f"Chunks: {len(chunks)}")
print(f"Embedding shape: {embeddings.shape}")
print(f"Embedding dtype: {embeddings.dtype}")


print("\nFirst chunk metadata:")
print(f"Chunk ID: {chunks[0]['chunk_id']}")
print(f"Page: {chunks[0]['page']}")
print(f"Source: {chunks[0]['source']}")

print("\nFirst embedding:")
print(embeddings[0])