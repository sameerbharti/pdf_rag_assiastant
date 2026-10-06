from app.ingestion.loader import load_pdf
from app.ingestion.chunker import chunk_text

from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import Retriever

from app.generation.extractive_fallback import (
    extract_relevant_sentences
)


PDF_PATH = "data/documents/test.pdf"


# --------------------------------------------------
# 1. Load document
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
# 4. Retriever
# --------------------------------------------------

retriever = Retriever(
    vector_store=vector_store,
    embedding_model=embedding_model
)


# --------------------------------------------------
# 5. Query
# --------------------------------------------------

query = "How do I verify my student status?"


# --------------------------------------------------
# 6. Retrieve
# --------------------------------------------------

results = retriever.retrieve(
    query=query,
    top_k=3
)


# --------------------------------------------------
# 7. Extractive fallback
# --------------------------------------------------

answer = extract_relevant_sentences(
    query=query,
    results=results,
    max_sentences=5
)


# --------------------------------------------------
# 8. Display
# --------------------------------------------------

print("=" * 60)
print("EXTRACTIVE FALLBACK TEST")
print("=" * 60)

print("\nQUESTION:")
print(query)

print("\n" + "=" * 60)
print("FALLBACK ANSWER")
print("=" * 60)

print(answer)

print("\n" + "=" * 60)
print("SOURCES")
print("=" * 60)

for i, result in enumerate(results, start=1):

    filename = (
        result["source"]
        .replace("\\", "/")
        .split("/")[-1]
    )

    print(
        f"{i}. {filename} "
        f"(Page {result['page']}, "
        f"Chunk {result['chunk_id']})"
    )
    