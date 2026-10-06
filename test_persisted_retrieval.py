from app.retrieval.index_loader import load_vector_store
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.retriever import Retriever


# --------------------------------------------------
# 1. Load persisted FAISS index
# --------------------------------------------------

vector_store = load_vector_store()


# --------------------------------------------------
# 2. Load embedding model
# --------------------------------------------------

embedding_model = EmbeddingModel()


# --------------------------------------------------
# 3. Create retriever
# --------------------------------------------------

retriever = Retriever(
    vector_store=vector_store,
    embedding_model=embedding_model
)


# --------------------------------------------------
# 4. Query
# --------------------------------------------------

query = "How do I verify my student status?"


# --------------------------------------------------
# 5. Retrieve
# --------------------------------------------------

results = retriever.retrieve(
    query=query,
    top_k=3
)


# --------------------------------------------------
# 6. Display
# --------------------------------------------------

print("=" * 60)
print("PERSISTED RETRIEVAL TEST")
print("=" * 60)

print(f"Query: {query}")
print(f"Results: {len(results)}")


for i, result in enumerate(
    results,
    start=1
):

    filename = (
        result["source"]
        .replace("\\", "/")
        .split("/")[-1]
    )

    print("\n" + "-" * 60)

    print(f"Result {i}")
    print(f"Score: {result['score']:.4f}")
    print(f"File: {filename}")
    print(f"Page: {result['page']}")
    print(f"Chunk: {result['chunk_id']}")

    print("\nText:")
    print(result["text"])