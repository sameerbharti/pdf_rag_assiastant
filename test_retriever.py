from app.ingestion.loader import load_pdf
from app.ingestion.chunker import chunk_text

from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import Retriever


PDF_PATH = "data/documents/test.pdf"


# --------------------------------------------------
# 1. Load and chunk PDF
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
# 4. Create Retriever
# --------------------------------------------------

retriever = Retriever(
    vector_store=vector_store,
    embedding_model=embedding_model
)


# --------------------------------------------------
# 5. Query
# --------------------------------------------------

query = "How do I verify my student status?"

results = retriever.retrieve(
    query=query,
    top_k=3
)


# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

print("=" * 60)
print("RETRIEVER TEST")
print("=" * 60)

print(f"Query: {query}")
print(f"Results returned: {len(results)}")


for i, result in enumerate(results, start=1):

    print("\n" + "=" * 60)
    print(f"RESULT {i}")

    print(f"Score: {result['score']:.4f}")
    print(f"Chunk ID: {result['chunk_id']}")
    print(f"Page: {result['page']}")
    print(f"Source: {result['source']}")

    print("\nText:")
    print(result["text"])