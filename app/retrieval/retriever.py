from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> list[dict]:

        if not query or not query.strip():
            return []

        query_embedding = self.embedding_model.encode(
            [query]
        )

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        return results